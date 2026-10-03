"""
NLLB pretrained English -> Hindi translation service.

This model is separate from our three locally trained models.

Model:
facebook/nllb-200-distilled-600M

Source language:
English -> eng_Latn

Target language:
Hindi -> hin_Deva

IMPORTANT:
- NLLB is NOT our from-scratch Transformer.
- It is loaded lazily only when the user selects NLLB.
- No fallback to our Transformer is performed.
"""

from functools import lru_cache

import torch

from .config import (
    NLLB_MODEL_ID,
    NLLB_SOURCE_LANGUAGE,
    NLLB_TARGET_LANGUAGE,
    NLLB_MAX_INPUT_TOKENS,
    NLLB_MAX_NEW_TOKENS,
    NLLB_NUM_BEAMS,
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# CUSTOM ERROR
# ============================================================

class NLLBUnavailable(RuntimeError):
    """
    Raised when the pretrained NLLB model cannot be loaded.
    """
    pass


# ============================================================
# LOAD NLLB
# ============================================================

@lru_cache(maxsize=1)
def load_nllb():
    """
    Load the Hugging Face NLLB model only when first requested.

    The first run may download several large model files.
    Later runs normally use the Hugging Face local cache.
    """

    try:
        # Import here so the rest of the application can still work
        # even if transformers is not installed.
        from transformers import (
            AutoTokenizer,
            AutoModelForSeq2SeqLM,
        )

    except ImportError as exc:
        raise NLLBUnavailable(
            "The 'transformers' package is not installed. "
            "Install the required packages before using NLLB."
        ) from exc

    try:
        # --------------------------------------------------------
        # TOKENIZER
        # --------------------------------------------------------

        tokenizer = AutoTokenizer.from_pretrained(
            NLLB_MODEL_ID,
            src_lang=NLLB_SOURCE_LANGUAGE,
        )

        # --------------------------------------------------------
        # MODEL
        # --------------------------------------------------------

        # Use FP16 only when CUDA is available.
        # CPU uses normal float32 for reliability.
        if DEVICE.type == "cuda":

            model = AutoModelForSeq2SeqLM.from_pretrained(
                NLLB_MODEL_ID,
                torch_dtype=torch.float16,
            )

        else:

            model = AutoModelForSeq2SeqLM.from_pretrained(
                NLLB_MODEL_ID,
            )

        model = model.to(DEVICE)

        model.eval()

        # --------------------------------------------------------
        # TARGET LANGUAGE TOKEN
        # --------------------------------------------------------

        forced_bos_token_id = tokenizer.convert_tokens_to_ids(
            NLLB_TARGET_LANGUAGE
        )

        if (
            forced_bos_token_id is None
            or forced_bos_token_id < 0
        ):
            raise RuntimeError(
                "Could not find Hindi language token "
                f"{NLLB_TARGET_LANGUAGE}."
            )

        return (
            tokenizer,
            model,
            forced_bos_token_id,
        )

    except Exception as exc:
        raise NLLBUnavailable(
            "Could not load the pretrained NLLB model "
            f"'{NLLB_MODEL_ID}'. "
            "Check your internet connection, available disk space, "
            "RAM, and Hugging Face dependencies. "
            f"Original error: {exc}"
        ) from exc


# ============================================================
# TRANSLATE WITH NLLB
# ============================================================

@torch.no_grad()
def translate_nllb(text: str, decoding: str = "beam"):
    """
    Translate English text into Hindi using pretrained NLLB.

    Parameters
    ----------
    text:
        English input sentence.

    decoding:
        'greedy' or 'beam'.

    Returns
    -------
    Dictionary compatible with the application's
    TranslationResult schema.
    """

    # ------------------------------------------------------------
    # INPUT VALIDATION
    # ------------------------------------------------------------

    text = text.strip()

    if not text:
        raise ValueError(
            "Please enter an English sentence."
        )

    # ------------------------------------------------------------
    # LOAD MODEL
    # ------------------------------------------------------------

    tokenizer, model, forced_bos_token_id = load_nllb()

    warnings = []

    # ------------------------------------------------------------
    # CHECK ORIGINAL TOKEN LENGTH
    # ------------------------------------------------------------

    try:
        full_ids = tokenizer.encode(
            text,
            add_special_tokens=True,
        )

        original_length = len(full_ids)

        if original_length > NLLB_MAX_INPUT_TOKENS:
            warnings.append(
                f"NLLB input was truncated from "
                f"{original_length} to "
                f"{NLLB_MAX_INPUT_TOKENS} tokens."
            )

    except Exception:
        # Length checking is helpful but must not stop translation.
        original_length = None

    # ------------------------------------------------------------
    # TOKENIZE
    # ------------------------------------------------------------

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=NLLB_MAX_INPUT_TOKENS,
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    # ------------------------------------------------------------
    # DECODING SETTINGS
    # ------------------------------------------------------------

    if decoding == "greedy":

        num_beams = 1

    elif decoding == "beam":

        num_beams = NLLB_NUM_BEAMS

    else:

        raise ValueError(
            "NLLB decoding must be either "
            "'greedy' or 'beam'."
        )

    # ------------------------------------------------------------
    # GENERATE HINDI
    # ------------------------------------------------------------

    try:
        generated_tokens = model.generate(
            **inputs,

            forced_bos_token_id=forced_bos_token_id,

            num_beams=num_beams,

            max_new_tokens=NLLB_MAX_NEW_TOKENS,

            early_stopping=True
            if num_beams > 1
            else False,
        )

    except Exception as exc:
        raise RuntimeError(
            f"NLLB translation failed: {exc}"
        ) from exc

    # ------------------------------------------------------------
    # DECODE
    # ------------------------------------------------------------

    translation = tokenizer.batch_decode(
        generated_tokens,
        skip_special_tokens=True,
    )[0].strip()

    if not translation:
        warnings.append(
            "NLLB produced an empty translation."
        )

    # ------------------------------------------------------------
    # RESULT
    # ------------------------------------------------------------

    return {
        "translation": translation,

        "warning": (
            " ".join(warnings)
            if warnings
            else None
        ),
    }


# ============================================================
# NLLB STATUS
# ============================================================

def nllb_status():
    """
    Return lightweight NLLB runtime information.

    This does NOT force the large model to download.
    """

    try:
        import transformers

        installed = True
        version = getattr(
            transformers,
            "__version__",
            "unknown",
        )

    except ImportError:
        installed = False
        version = None

    return {
        "model": NLLB_MODEL_ID,

        "source_language": NLLB_SOURCE_LANGUAGE,

        "target_language": NLLB_TARGET_LANGUAGE,

        "device": str(DEVICE),

        "transformers_installed": installed,

        "transformers_version": version,

        "loaded": load_nllb.cache_info().currsize > 0,
    }