import json
import math
import tempfile
import unittest

from pathlib import Path
from unittest.mock import patch

import torch
from pydantic import ValidationError

from api.schemas import TranslationRequest
from api.translation_service import translate
from api.model_loader import get_model
from api.decoding import decode, greedy_token


# ============================================================
# SENTENCES USED ONLY FOR SMOKE TESTING
# ============================================================

SENTENCES = (
    "I am happy.",
    "How are you?",
    "India is a diverse country.",
)


class TranslationTests(unittest.TestCase):

    # ========================================================
    # COMMON SMOKE TEST
    # ========================================================

    def smoke(self, name, method):
        """
        Basic translation test shared by the local models.

        Checks that:
        - translation returns a string
        - translation or warning is present
        - latency is recorded
        - special BOS token is not shown to the user
        """

        for sentence in SENTENCES:

            result = translate(
                TranslationRequest(
                    text=sentence,
                    model=name,
                    decoding=method,
                )
            )

            self.assertIsInstance(
                result.translation,
                str,
            )

            self.assertTrue(
                result.translation
                or result.warning
            )

            self.assertGreater(
                result.latency_ms,
                0,
            )

            self.assertNotIn(
                "<s>",
                result.translation,
            )


    # ========================================================
    # GRU BASELINE
    # ========================================================

    def test_gru_smoke(self):

        self.smoke(
            "gru",
            "greedy",
        )


    # ========================================================
    # GRU + BAHDANAU ATTENTION
    # ========================================================

    def test_attention_smoke_and_alignment(self):

        self.smoke(
            "attention",
            "greedy",
        )

        result = translate(
            TranslationRequest(
                text=SENTENCES[0],
                model="attention",
            )
        )

        # Attention model should return target tokens.
        self.assertTrue(
            result.target_tokens
        )

        # One attention row should correspond to
        # each displayed target token.
        self.assertEqual(
            len(result.attention_matrix),
            len(result.target_tokens),
        )

        # Validate every attention row.
        for row in result.attention_matrix:

            self.assertEqual(
                len(row),
                len(result.source_tokens),
            )

            # Every attention value must be finite
            # and between 0 and 1.
            self.assertTrue(
                all(
                    math.isfinite(value)
                    and 0 <= value <= 1
                    for value in row
                )
            )

            # Attention probabilities should sum to 1.
            self.assertAlmostEqual(
                sum(row),
                1,
                places=5,
            )


    # ========================================================
    # TRANSFORMER GREEDY
    # ========================================================

    def test_transformer_greedy(self):

        self.smoke(
            "transformer",
            "greedy",
        )


    # ========================================================
    # TRANSFORMER BEAM SEARCH
    # ========================================================

    def test_transformer_beam(self):

        self.smoke(
            "transformer",
            "beam",
        )


    # ========================================================
    # LONG INPUT HANDLING
    # ========================================================

    def test_long_input_truncation_all_models(self):

        long_text = (
            SENTENCES[0] + " "
        ) * 80

        for model in (
            "gru",
            "attention",
            "transformer",
        ):

            result = translate(
                TranslationRequest(
                    text=long_text,
                    model=model,
                    decoding="greedy",
                )
            )

            self.assertIn(
                "truncated",
                result.warning,
            )


    # ========================================================
    # INVALID REQUEST VALIDATION
    # ========================================================

    def test_empty_and_invalid_options(self):

        invalid_payloads = (

            # Empty input
            {
                "text": "",
            },

            # Whitespace input
            {
                "text": "   ",
            },

            # Invalid model
            {
                "text": "I",
                "model": "bad",
            },

            # Invalid decoding
            {
                "text": "I",
                "decoding": "bad",
            },

            # GRU must not accept beam search
            {
                "text": "I",
                "model": "gru",
                "decoding": "beam",
            },

            # Input longer than UI/API limit
            {
                "text": "x" * 20001,
            },

            # Invalid Unicode surrogate
            {
                "text": "\ud800",
            },

            # Null character
            {
                "text": "\x00",
            },
        )

        for payload in invalid_payloads:

            with self.assertRaises(
                ValidationError
            ):

                TranslationRequest(
                    **payload
                )


    # ========================================================
    # UNICODE / EDGE CASE INPUTS
    # ========================================================

    def test_unicode_emoji_punctuation_short(self):

        # Edge cases based on ordinary smoke-test inputs.
        # These are not official test-set examples.

        test_inputs = (
            "I",
            "I am happy. 🙂",
            "I am happy. हिन्दी",
            "!",
        )

        for text in test_inputs:

            result = translate(
                TranslationRequest(
                    text=text,
                    model="gru",
                )
            )

            self.assertIsInstance(
                result.translation,
                str,
            )


    # ========================================================
    # LOGGING
    # ========================================================

    def test_logging_records_and_failure_is_nonfatal(self):

        with tempfile.TemporaryDirectory() as directory:

            path = (
                Path(directory)
                / "predictions.jsonl"
            )

            # Test normal logging.
            with patch(
                "api.logging_service.LOG_PATH",
                path,
            ):

                result = translate(
                    TranslationRequest(
                        text=SENTENCES[0],
                        model="gru",
                    )
                )

            record = json.loads(
                path.read_text(
                    encoding="utf-8"
                )
            )

            self.assertEqual(
                record["translation"],
                result.translation,
            )

            self.assertEqual(
                set(record),
                {
                    "timestamp",
                    "input_text",
                    "model",
                    "decoding",
                    "translation",
                    "latency_ms",
                    "warning",
                },
            )

            # Logging failure must not stop translation.
            with (
                patch(
                    "api.logging_service.LOG_PATH",
                    Path(directory),
                ),
                self.assertLogs(
                    "api.logging_service",
                    level="ERROR",
                ),
            ):

                translate(
                    TranslationRequest(
                        text=SENTENCES[0],
                        model="gru",
                    )
                )


    # ========================================================
    # INVALID MODEL LOGITS
    # ========================================================

    def test_invalid_logits_are_rejected(self):

        # Current decoding.py uses greedy_token(),
        # which calls validate_logits() before argmax.

        invalid_logits = torch.tensor(
            [
                float("nan")
            ]
        )

        with self.assertRaises(
            RuntimeError
        ):

            greedy_token(
                invalid_logits,
                1,
            )


    # ========================================================
    # EMPTY GENERATION
    # ========================================================

    def test_empty_generation_has_warning(self):

        bundle = get_model(
            "gru"
        )

        device = next(
            bundle.model.parameters()
        ).device

        logits = torch.zeros(
            (
                1,
                bundle.target.vocab_size(),
            ),
            device=device,
        )

        # Force EOS immediately.
        logits[
            0,
            bundle.target.eos_id()
        ] = 100

        with patch.object(
            bundle.model.decoder,
            "forward",
            return_value=(
                logits,
                None,
            ),
        ):

            result = decode(
                bundle,
                SENTENCES[0],
                "greedy",
            )

        self.assertEqual(
            result["translation"],
            "",
        )

        self.assertIn(
            "no displayable",
            result["warning"],
        )


    # ========================================================
    # GENERATION LIMIT
    # ========================================================

    def test_generation_stops_at_limit_without_eos(self):

        bundle = get_model(
            "gru"
        )

        device = next(
            bundle.model.parameters()
        ).device

        logits = torch.zeros(
            (
                1,
                bundle.target.vocab_size(),
            ),
            device=device,
        )

        # Force a normal token repeatedly so EOS is never produced.
        logits[
            0,
            4
        ] = 100

        with patch.object(
            bundle.model.decoder,
            "forward",
            return_value=(
                logits,
                None,
            ),
        ) as decoder:

            result = decode(
                bundle,
                SENTENCES[0],
                "greedy",
            )

        # MAX_LEN is 64:
        # generation can therefore run for 63 decoder steps.
        self.assertEqual(
            decoder.call_count,
            63,
        )

        self.assertIn(
            "without EOS",
            result["warning"],
        )


# ============================================================
# RUN DIRECTLY
# ============================================================

if __name__ == "__main__":
    unittest.main()