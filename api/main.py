"""
FastAPI backend for the English-to-Hindi Neural Translator.

Models:
1. GRU Baseline
2. GRU + Bahdanau Attention
3. Transformer From Scratch
4. NLLB — Tool-Assisted / Pretrained Reference
"""

import logging

from fastapi import (
    FastAPI,
    HTTPException,
)

from fastapi.middleware.cors import (
    CORSMiddleware,
)

from .config import LABELS

from .model_loader import (
    DEVICE,
    ModelUnavailable,
    get_available_models,
    get_loaded_models,
    load_tokenizers,
)

from .nllb_service import (
    NLLBUnavailable,
    nllb_status,
)

from .schemas import (
    TranslationRequest,
    TranslationResult,
    AttentionRequest,
    CompareRequest,
)

from .translation_service import translate


# ============================================================
# LOGGER
# ============================================================

logger = logging.getLogger(__name__)


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="English → Hindi Neural Translator API",
    description=(
        "English-to-Hindi translation using GRU, "
        "Bahdanau Attention, Transformer From Scratch "
        "and pretrained NLLB."
    ),
    version="1.0.0",
)


# ============================================================
# CORS
# ============================================================

# Frontend normally runs on:
# http://127.0.0.1:5500
# or
# http://localhost:5500

app.add_middleware(
    CORSMiddleware,

    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
    ],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],
)


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():
    """
    Basic backend status.
    """

    return {
        "status": "ok",

        "message":
            "English → Hindi Neural Translator API is running.",

        "docs": "/docs",

        "health": "/health",

        "models": "/models",
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():
    """
    Lightweight application health check.

    This endpoint does NOT force NLLB to download.
    """

    try:

        # Verify original SentencePiece tokenizers.
        normal_source, normal_target = (
            load_tokenizers(
                improved=False
            )
        )

        # Verify improved Transformer tokenizers.
        improved_source, improved_target = (
            load_tokenizers(
                improved=True
            )
        )

        tokenizer_status = {

            "gru_attention": {

                "loaded": True,

                "english_vocab":
                    normal_source.vocab_size(),

                "hindi_vocab":
                    normal_target.vocab_size(),
            },

            "transformer": {

                "loaded": True,

                "english_vocab":
                    improved_source.vocab_size(),

                "hindi_vocab":
                    improved_target.vocab_size(),
            },
        }

    except Exception as exc:

        logger.warning(
            "Tokenizer health check failed: %s",
            exc,
        )

        tokenizer_status = {
            "loaded": False,
            "error": str(exc),
        }

    return {

        "status": "ok",

        "device":
            str(DEVICE),

        "loaded_local_models":
            get_loaded_models(),

        "models":
            get_available_models(),

        "tokenizers":
            tokenizer_status,

        "nllb":
            nllb_status(),
    }


# ============================================================
# MODEL LIST
# ============================================================

@app.get("/models")
def models():
    """
    Return all selectable models and their current status.
    """

    return {
        "models":
            get_available_models()
    }


# ============================================================
# TRANSLATE
# ============================================================

@app.post(
    "/translate",
    response_model=TranslationResult,
)
def translate_endpoint(
    request: TranslationRequest,
):
    """
    Translate English text using the selected model.
    """

    try:

        return translate(
            request
        )

    except NLLBUnavailable as exc:

        # Never silently replace NLLB with Transformer.
        logger.warning(
            "NLLB translation unavailable: %s",
            exc,
        )

        raise HTTPException(
            status_code=503,
            detail=str(exc),
        )

    except ModelUnavailable as exc:

        logger.warning(
            "Requested model unavailable: %s",
            exc,
        )

        raise HTTPException(
            status_code=503,
            detail=str(exc),
        )

    except ValueError as exc:

        logger.warning(
            "Invalid translation request: %s",
            exc,
        )

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except RuntimeError as exc:

        logger.error(
            "Translation runtime error: %s",
            exc,
            exc_info=True,
        )

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )

    except Exception as exc:

        logger.exception(
            "Unexpected translation failure."
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Translation failed unexpectedly: "
                f"{exc}"
            ),
        )


# ============================================================
# ATTENTION ALIGNMENT
# ============================================================

@app.post("/attention")
def attention_endpoint(
    request: AttentionRequest,
):
    """
    Generate a translation and genuine Bahdanau attention matrix.

    This endpoint always uses:
    GRU + Bahdanau Attention
    """

    try:

        translation_request = TranslationRequest(
            text=request.text,
            model="attention",
            decoding="greedy",
        )

        result = translate(
            translation_request
        )

        if (
            result.attention_matrix is None
            or result.source_tokens is None
            or result.target_tokens is None
        ):

            raise RuntimeError(
                "Attention model did not return "
                "alignment information."
            )

        return {

            "input_text":
                result.input_text,

            "translation":
                result.translation,

            "model":
                result.model,

            "model_label":
                result.model_label,

            "source_tokens":
                result.source_tokens,

            "target_tokens":
                result.target_tokens,

            "attention_matrix":
                result.attention_matrix,

            "latency_ms":
                result.latency_ms,

            "warning":
                result.warning,
        }

    except ModelUnavailable as exc:

        logger.warning(
            "Attention model unavailable: %s",
            exc,
        )

        raise HTTPException(
            status_code=503,
            detail=str(exc),
        )

    except ValueError as exc:

        logger.warning(
            "Invalid attention request: %s",
            exc,
        )

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except RuntimeError as exc:

        logger.error(
            "Attention runtime error: %s",
            exc,
            exc_info=True,
        )

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )

    except Exception as exc:

        logger.exception(
            "Unexpected attention failure."
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Attention request failed: "
                f"{exc}"
            ),
        )


# ============================================================
# COMPARE MODELS
# ============================================================

@app.post("/compare")
def compare_endpoint(
    request: CompareRequest,
):
    """
    Translate the same English text using multiple models.

    Required project models:
    - GRU
    - GRU + Bahdanau Attention
    - Transformer From Scratch

    Optional:
    - NLLB pretrained reference

    A failure from one model does not remove successful
    translations from the other models.
    """

    results = []

    errors = {}


    # --------------------------------------------------------
    # MODEL ORDER
    # --------------------------------------------------------

    model_names = [
        "gru",
        "attention",
        "transformer",
    ]

    if request.include_nllb:

        model_names.append(
            "nllb"
        )


    # --------------------------------------------------------
    # RUN MODELS ONE BY ONE
    # --------------------------------------------------------

    for model_name in model_names:

        # GRU models support greedy decoding only.
        if model_name in (
            "gru",
            "attention",
        ):

            decoding = "greedy"

        else:

            decoding = (
                request.decoding
            )


        try:

            translation_request = (
                TranslationRequest(
                    text=request.text,
                    model=model_name,
                    decoding=decoding,
                )
            )

            result = translate(
                translation_request
            )

            results.append(
                {

                    "model":
                        result.model,

                    "model_label":
                        result.model_label,

                    "translation":
                        result.translation,

                    "decoding":
                        result.decoding,

                    "latency_ms":
                        result.latency_ms,

                    "warning":
                        result.warning,

                    "success":
                        True,
                }
            )


        except Exception as exc:

            # Keep successful model results available.
            # Failed models are reported separately.

            logger.warning(
                "Comparison model '%s' failed: %s",
                model_name,
                exc,
            )

            errors[
                model_name
            ] = str(exc)


    return {

        "input_text":
            request.text,

        "results":
            results,

        "errors":
            errors,
    }