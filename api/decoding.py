"""
Inference utilities for the three locally trained models.

Supported:
- GRU Baseline: greedy decoding
- GRU + Bahdanau Attention: greedy decoding + attention matrix
- Transformer From Scratch: greedy or beam search

NLLB is handled separately in nllb_service.py.
"""

import torch

from .config import (
    BEAM_SIZE,
    LENGTH_PENALTY,
)


# ============================================================
# SOURCE TOKENIZATION
# ============================================================

def encode_source(bundle, text):
    """
    Convert English text into the SentencePiece IDs expected
    by the selected locally trained model.
    """

    text = text.strip()

    if not text:
        raise ValueError(
            "Please enter an English sentence."
        )

    # Tokenize content without automatically adding BOS/EOS.
    pieces = bundle.source.encode(
        text,
        out_type=int,
    )

    warnings = []

    # Leave room for BOS and EOS.
    max_content_length = bundle.max_length - 2

    if len(pieces) > max_content_length:
        warnings.append(
            f"Input truncated from {len(pieces)} to "
            f"{max_content_length} content tokens "
            f"({bundle.max_length} tokens including BOS/EOS)."
        )

    pieces = pieces[:max_content_length]

    if bundle.source.unk_id() in pieces:
        warnings.append(
            "Input contains unknown tokenizer pieces; "
            "translation may lose information."
        )

    # Add sequence boundary tokens.
    ids = [
        bundle.source.bos_id(),
        *pieces,
        bundle.source.eos_id(),
    ]

    visible_length = len(ids)

    # GRU models were trained with padded source sequences.
    # Transformer uses masks and can use the actual source length.
    if bundle.name != "transformer":

        ids += [
            bundle.source.pad_id()
        ] * (
            bundle.max_length - len(ids)
        )

    device = next(
        bundle.model.parameters()
    ).device

    source_tensor = torch.tensor(
        [ids],
        dtype=torch.long,
        device=device,
    )

    return (
        source_tensor,
        ids[:visible_length],
        warnings,
    )


# ============================================================
# LOGIT VALIDATION
# ============================================================

def validate_logits(logits, vocab_size):
    """
    Make sure the model returned a valid target vocabulary tensor.
    """

    if logits.shape[-1] != vocab_size:
        raise RuntimeError(
            "Model output vocabulary size does not match "
            f"the target tokenizer. "
            f"Expected {vocab_size}, got {logits.shape[-1]}."
        )

    if not torch.isfinite(logits).all():
        raise RuntimeError(
            "Model produced NaN or infinite logits."
        )


def greedy_token(logits, vocab_size):
    """
    Validate logits and choose the highest-probability token.
    """

    validate_logits(
        logits,
        vocab_size,
    )

    return logits.argmax(
        dim=-1
    ).item()


# ============================================================
# TRANSFORMER BEAM SEARCH
# ============================================================

def transformer_beam_search(
    model,
    source,
    bos_id,
    eos_id,
    max_length,
    vocab_size,
):
    """
    Beam search equivalent to the final Transformer notebook.

    Configuration:
    beam size = 4
    length penalty = 0.6
    """

    encoder_output, src_mask = model.encode(
        source
    )

    # Each beam:
    # (token_sequence, cumulative_log_probability)
    beams = [
        ([bos_id], 0.0)
    ]

    completed = []

    def rank(item):
        sequence, score = item

        generated_length = max(
            1,
            len(sequence) - 1,
        )

        return (
            score
            / (
                generated_length
                ** LENGTH_PENALTY
            )
        )

    for _ in range(
        max_length - 1
    ):

        candidates = []

        for sequence, score in beams:

            # Finished beam.
            if sequence[-1] == eos_id:

                completed.append(
                    (sequence, score)
                )

                continue

            target_tensor = torch.tensor(
                [sequence],
                dtype=torch.long,
                device=source.device,
            )

            logits = model.decode(
                target_tensor,
                encoder_output,
                src_mask,
                return_attention=False,
            )

            # Last decoder position.
            logits = logits[
                0,
                -1,
                :
            ]

            validate_logits(
                logits,
                vocab_size,
            )

            log_probabilities = torch.log_softmax(
                logits,
                dim=-1,
            )

            values, token_ids = torch.topk(
                log_probabilities,
                k=min(
                    BEAM_SIZE,
                    vocab_size,
                ),
            )

            for log_probability, token_id in zip(
                values.tolist(),
                token_ids.tolist(),
            ):

                candidates.append(
                    (
                        sequence + [
                            int(token_id)
                        ],
                        score
                        + float(
                            log_probability
                        ),
                    )
                )

        # Every active beam may already have ended.
        if not candidates:
            break

        beams = sorted(
            candidates,
            key=rank,
            reverse=True,
        )[:BEAM_SIZE]

    completed.extend(
        beams
    )

    if not completed:
        return [
            bos_id
        ]

    best_sequence = max(
        completed,
        key=rank,
    )[0]

    return best_sequence


# ============================================================
# LOCAL MODEL DECODING
# ============================================================

@torch.no_grad()
def decode(bundle, text, method="greedy"):
    """
    Translate using one of the three local project models.

    GRU:
        greedy only

    Attention:
        greedy only + alignment matrix

    Transformer:
        greedy or beam
    """

    source, source_ids, warnings = encode_source(
        bundle,
        text,
    )

    model = bundle.model
    target = bundle.target

    bos_id = target.bos_id()
    eos_id = target.eos_id()
    pad_id = target.pad_id()

    vocab_size = target.vocab_size()

    # ========================================================
    # DECODING METHOD VALIDATION
    # ========================================================

    if bundle.name == "transformer":

        if method not in (
            "greedy",
            "beam",
        ):
            raise ValueError(
                "Transformer decoding must be "
                "'greedy' or 'beam'."
            )

        effective_method = method

    else:

        # GRU and Attention models use greedy decoding only.
        effective_method = "greedy"

        if method != "greedy":
            warnings.append(
                f"{bundle.name} supports greedy decoding only; "
                "greedy decoding was used."
            )

    generated = []

    attention_rows = []


    # ========================================================
    # TRANSFORMER BEAM SEARCH
    # ========================================================

    if (
        bundle.name == "transformer"
        and effective_method == "beam"
    ):

        full_sequence = transformer_beam_search(
            model=model,
            source=source,
            bos_id=bos_id,
            eos_id=eos_id,
            max_length=bundle.max_length,
            vocab_size=vocab_size,
        )

        # Remove initial BOS from generated sequence.
        generated = full_sequence[1:]


    # ========================================================
    # GREEDY DECODING
    # ========================================================

    else:

        # ----------------------------------------------------
        # PREPARE ENCODER STATE
        # ----------------------------------------------------

        if bundle.name == "gru":

            hidden = model.forward_encoder(
                source
            )

        elif bundle.name == "attention":

            (
                encoder_output,
                hidden,
                source_mask,
            ) = model.encode(
                source
            )

        elif bundle.name == "transformer":

            (
                encoder_output,
                source_mask,
            ) = model.encode(
                source
            )

        else:

            raise ValueError(
                f"Unsupported model: {bundle.name}"
            )

        sequence = [
            bos_id
        ]

        # ----------------------------------------------------
        # AUTOREGRESSIVE GENERATION
        # ----------------------------------------------------

        for _ in range(
            bundle.max_length - 1
        ):

            # ==============================
            # TRANSFORMER
            # ==============================

            if bundle.name == "transformer":

                target_tensor = torch.tensor(
                    [sequence],
                    dtype=torch.long,
                    device=source.device,
                )

                logits = model.decode(
                    target_tensor,
                    encoder_output,
                    source_mask,
                    return_attention=False,
                )

                logits = logits[
                    :,
                    -1,
                    :
                ]

            # ==============================
            # ATTENTION GRU
            # ==============================

            elif bundle.name == "attention":

                previous_token = torch.tensor(
                    [
                        sequence[-1]
                    ],
                    dtype=torch.long,
                    device=source.device,
                )

                (
                    logits,
                    hidden,
                    attention_weights,
                ) = model.decoder(
                    previous_token,
                    hidden,
                    encoder_output,
                    source_mask,
                )

            # ==============================
            # BASELINE GRU
            # ==============================

            else:

                previous_token = torch.tensor(
                    [
                        sequence[-1]
                    ],
                    dtype=torch.long,
                    device=source.device,
                )

                (
                    logits,
                    hidden,
                ) = model.decoder(
                    previous_token,
                    hidden,
                )

            # ------------------------------------------------
            # SELECT NEXT TOKEN
            # ------------------------------------------------

            next_id = greedy_token(
                logits,
                vocab_size,
            )

            sequence.append(
                next_id
            )

            generated.append(
                next_id
            )

            # ------------------------------------------------
            # STOP AT EOS
            # ------------------------------------------------

            if next_id == eos_id:
                break

            # ------------------------------------------------
            # ATTENTION ALIGNMENT
            # ------------------------------------------------

            # Store attention only for an actual displayed
            # target token, not BOS/PAD/EOS.
            if (
                bundle.name == "attention"
                and next_id not in (
                    pad_id,
                    bos_id,
                    eos_id,
                )
            ):

                attention_rows.append(
                    attention_weights[
                        0,
                        :len(source_ids),
                    ]
                    .detach()
                    .cpu()
                    .tolist()
                )


    # ========================================================
    # GENERATION WARNINGS
    # ========================================================

    if (
        not generated
        or generated[-1] != eos_id
    ):
        warnings.append(
            "Generation reached the model's maximum "
            "length without EOS."
        )


    # ========================================================
    # REMOVE SPECIAL TOKENS
    # ========================================================

    clean_ids = [
        token_id
        for token_id in generated
        if token_id not in (
            pad_id,
            bos_id,
            eos_id,
        )
    ]


    # ========================================================
    # TOKEN ID SAFETY
    # ========================================================

    for token_id in clean_ids:

        if not (
            0
            <= token_id
            < vocab_size
        ):

            raise RuntimeError(
                "Model generated an invalid token ID."
            )


    # ========================================================
    # SENTENCEPIECE DECODE
    # ========================================================

    translation = target.decode(
        clean_ids
    ).strip()


    # ========================================================
    # OUTPUT WARNINGS
    # ========================================================

    if not translation:

        warnings.append(
            "The model produced no displayable translation."
        )

    if (
        target.unk_id()
        in clean_ids
    ):

        warnings.append(
            "Output contains an unknown tokenizer piece."
        )


    # ========================================================
    # RESULT
    # ========================================================

    result = {
        "translation": translation,

        "warning": (
            " ".join(warnings)
            if warnings
            else None
        ),

        "effective_decoding":
            effective_method,
    }


    # ========================================================
    # ATTENTION RESULT
    # ========================================================

    if bundle.name == "attention":

        source_tokens = [
            bundle.source.id_to_piece(
                token_id
            )
            for token_id in source_ids
        ]

        target_tokens = [
            target.id_to_piece(
                token_id
            )
            for token_id in clean_ids
        ]

        # Defensive alignment check.
        # Each displayed target token should correspond
        # to one attention row.
        if (
            len(attention_rows)
            > len(target_tokens)
        ):
            attention_rows = attention_rows[
                :len(target_tokens)
            ]

        result.update(
            {
                "source_tokens":
                    source_tokens,

                "target_tokens":
                    target_tokens,

                "attention_matrix":
                    attention_rows,
            }
        )

    return result