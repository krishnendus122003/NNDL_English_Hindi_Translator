"""
Central translation service.

Routing:
- gru         -> our trained GRU checkpoint
- attention   -> our trained GRU + Bahdanau Attention checkpoint
- transformer -> our trained Transformer From Scratch checkpoint
- nllb        -> pretrained NLLB model

IMPORTANT:
NLLB never falls back to the scratch Transformer.
"""

from threading import Lock
from time import perf_counter

from .config import LABELS
from .decoding import decode
from .logging_service import log_prediction
from .model_loader import get_model
from .nllb_service import translate_nllb
from .schemas import (
    TranslationRequest,
    TranslationResult,
)


# ============================================================
# INFERENCE LOCK
# ============================================================

# Running several large models at exactly the same time can consume
# too much RAM/CPU on a laptop. Serialize inference requests.
_inference_lock = Lock()


# ============================================================
# TRANSLATION
# ============================================================

def translate(request: TranslationRequest):
    """
    Translate one English sentence using the model selected
    by the user.
    """

    start_time = perf_counter()

    # --------------------------------------------------------
    # CLEAN INPUT
    # --------------------------------------------------------

    text = request.text.strip()

    if not text:
        raise ValueError(
            "Please enter an English sentence."
        )

    # --------------------------------------------------------
    # VERIFY MODEL NAME
    # --------------------------------------------------------

    if request.model not in LABELS:
        raise ValueError(
            f"Unknown model: {request.model}"
        )

    # --------------------------------------------------------
    # RUN INFERENCE
    # --------------------------------------------------------

    with _inference_lock:

        # ====================================================
        # NLLB
        # ====================================================

        if request.model == "nllb":

            # This really runs NLLB.
            # There is NO fallback to Transformer.
            payload = translate_nllb(
                text=text,
                decoding=request.decoding,
            )

            actual_decoding = (
                request.decoding
            )

        # ====================================================
        # OUR LOCAL TRAINED MODELS
        # ====================================================

        else:

            bundle = get_model(
                request.model
            )

            payload = decode(
                bundle=bundle,
                text=text,
                method=request.decoding,
            )

            # decode() may change unsupported GRU "beam"
            # requests to greedy.
            actual_decoding = payload.pop(
                "effective_decoding",
                request.decoding,
            )

    # --------------------------------------------------------
    # LATENCY
    # --------------------------------------------------------

    latency_ms = round(
        (
            perf_counter()
            - start_time
        )
        * 1000,
        2,
    )

    # --------------------------------------------------------
    # BUILD API RESULT
    # --------------------------------------------------------

    result = TranslationResult(
        input_text=text,

        model=request.model,

        model_label=LABELS[
            request.model
        ],

        decoding=actual_decoding,

        latency_ms=latency_ms,

        **payload,
    )

    # --------------------------------------------------------
    # LOG PREDICTION
    # --------------------------------------------------------

    # Logging must never be responsible for model selection.
    # It only records the completed result.
    try:

        log_prediction(
            result.model_dump()
        )

    except Exception as exc:

        # Translation should still be returned even if writing
        # the log file fails.
        print(
            f"Prediction logging warning: {exc}"
        )

    return result