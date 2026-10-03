"""
Application configuration for the English-to-Hindi translator.

IMPORTANT:
- Training notebooks are not modified here.
- Trained checkpoints are not modified.
- GRU, Attention-GRU and Transformer use local checkpoints.
- NLLB is loaded separately from Hugging Face as a pretrained model.
"""

from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================

# Root folder:
# NNDL_English_Hindi_Translator/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# SentencePiece tokenizer folder
TOKENIZER_DIR = PROJECT_ROOT / "tokenizers"

# Prediction log
LOG_PATH = PROJECT_ROOT / "logs" / "predictions.jsonl"


# ============================================================
# OUR TRAINED MODEL CHECKPOINTS
# ============================================================

CHECKPOINTS = {

    # Model 1: GRU Encoder-Decoder
    "gru": (
        PROJECT_ROOT
        / "saved_models"
        / "gru"
        / "gru_seq2seq_best.pt"
    ),

    # Model 2: GRU + Bahdanau Attention
    "attention": (
        PROJECT_ROOT
        / "saved_models"
        / "gru_attention"
        / "gru_bahdanau_best.pt"
    ),

    # Model 3: Final Transformer From Scratch
    # IMPORTANT:
    # This is the selected Epoch-19 model.
    "transformer": (
        PROJECT_ROOT
        / "saved_models"
        / "transformer_1m_improved"
        / "transformer_1m_improved_best.pt"
    ),
}


# ============================================================
# MODEL DISPLAY LABELS
# ============================================================

LABELS = {
    "gru": "GRU Baseline",

    "attention": "GRU + Bahdanau Attention",

    "transformer": "Transformer From Scratch",

    "nllb": "NLLB — Tool-Assisted / Pretrained Reference",
}


# ============================================================
# NLLB PRETRAINED MODEL
# ============================================================

# This model is NOT our from-scratch Transformer.
# It is used as a pretrained/tool-assisted reference model.

NLLB_MODEL_ID = "facebook/nllb-200-distilled-600M"

# NLLB language codes
NLLB_SOURCE_LANGUAGE = "eng_Latn"
NLLB_TARGET_LANGUAGE = "hin_Deva"


# ============================================================
# DECODING SETTINGS
# ============================================================

# Final Transformer beam-search configuration
BEAM_SIZE = 4

# Best validation length penalty
LENGTH_PENALTY = 0.6


# ============================================================
# MODEL SEQUENCE LENGTH
# ============================================================

# Our trained models were designed for max sequence length 64.
# Actual checkpoint configuration is still treated as the source
# of truth when the model is loaded.
MAX_SEQUENCE_LENGTH = 64


# ============================================================
# API INPUT SAFETY
# ============================================================

# Prevent extremely large browser/API requests.
#
# NOTE:
# This does NOT mean the neural models accept 20,000 characters.
# Token-level truncation is performed separately according to each
# model's actual sequence length.
MAX_INPUT_CHARS = 20000


# ============================================================
# DEFAULT APPLICATION MODEL
# ============================================================

DEFAULT_MODEL = "transformer"


# ============================================================
# NLLB GENERATION SETTINGS
# ============================================================

# NLLB has its own Hugging Face generation system.
# Keep its settings independent from our scratch Transformer.

NLLB_MAX_INPUT_TOKENS = 256

NLLB_MAX_NEW_TOKENS = 128

NLLB_NUM_BEAMS = 4