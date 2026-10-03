"""
Model loading for the three locally trained translation models.

Local models:
1. GRU Baseline
2. GRU + Bahdanau Attention
3. Transformer From Scratch

NLLB is intentionally loaded separately in nllb_service.py because it
is a pretrained Hugging Face model rather than one of our checkpoints.

IMPORTANT:
- No training happens here.
- Checkpoints are loaded strictly.
- Trained files are never modified.
"""

from dataclasses import dataclass
from functools import lru_cache
import json
from threading import RLock

import sentencepiece as spm
import torch

from .config import (
    CHECKPOINTS,
    LABELS,
    TOKENIZER_DIR,
)

from .model_classes import (
    EncoderGRU,
    DecoderGRU,
    Seq2SeqGRU,
    EncoderGRUAttention,
    DecoderGRUAttention,
    Seq2SeqGRUAttention,
    TransformerFromScratch,
)

from .nllb_service import nllb_status


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Prevent excessive CPU thread usage for single-sentence inference.
if DEVICE.type == "cpu":
    torch.set_num_threads(
        min(4, torch.get_num_threads())
    )


# ============================================================
# CACHE
# ============================================================

_model_cache = {}

_cache_lock = RLock()


# ============================================================
# CUSTOM ERROR
# ============================================================

class ModelUnavailable(RuntimeError):
    """
    Raised when a required local model/tokenizer cannot be loaded.
    """
    pass


# ============================================================
# MODEL BUNDLE
# ============================================================

@dataclass
class ModelBundle:
    """
    Everything required to perform inference with one local model.
    """

    name: str

    model: torch.nn.Module

    source: object

    target: object

    max_length: int

    epoch: int

    checkpoint: str


# ============================================================
# TOKENIZER LOADING
# ============================================================

@lru_cache(maxsize=2)
def load_tokenizers(improved: bool = False):
    """
    Load the SentencePiece tokenizer pair.

    improved=False:
        english_bpe.model
        hindi_bpe.model

    improved=True:
        english_bpe_1m.model
        hindi_bpe_1m.model
    """

    suffix = "_1m" if improved else ""

    tokenizers = []

    for language in ("english", "hindi"):

        path = (
            TOKENIZER_DIR
            / f"{language}_bpe{suffix}.model"
        )

        if not path.is_file():
            raise ModelUnavailable(
                f"Required tokenizer is missing: {path}"
            )

        try:

            tokenizer = spm.SentencePieceProcessor(
                model_file=str(path)
            )

        except Exception as exc:
            raise ModelUnavailable(
                f"Could not load tokenizer "
                f"{path.name}: {exc}"
            ) from exc

        # ----------------------------------------------------
        # VERIFY SPECIAL TOKEN IDs
        # ----------------------------------------------------

        ids = (
            tokenizer.pad_id(),
            tokenizer.unk_id(),
            tokenizer.bos_id(),
            tokenizer.eos_id(),
        )

        if ids != (0, 1, 2, 3):
            raise ModelUnavailable(
                "Tokenizer special IDs do not match "
                f"training for {path.name}. "
                f"Expected (0, 1, 2, 3), found {ids}."
            )

        tokenizers.append(
            tokenizer
        )

    return tuple(tokenizers)


# ============================================================
# LOAD CHECKPOINT SAFELY
# ============================================================

def _read_checkpoint(path):
    """
    Load a trusted local project checkpoint onto CPU first.
    """

    try:

        checkpoint = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )

    except TypeError:
        # Compatibility with PyTorch versions where weights_only
        # is not supported.
        checkpoint = torch.load(
            path,
            map_location="cpu",
        )

    except Exception as exc:
        raise ModelUnavailable(
            f"Could not read checkpoint "
            f"{path.name}: {exc}"
        ) from exc

    if not isinstance(checkpoint, dict):
        raise ModelUnavailable(
            f"Checkpoint {path.name} "
            "does not contain the expected dictionary."
        )

    if "model_state_dict" not in checkpoint:
        raise ModelUnavailable(
            f"Checkpoint {path.name} "
            "does not contain model_state_dict."
        )

    return checkpoint


# ============================================================
# LOAD LOCAL TRAINED MODEL
# ============================================================

def _load(name: str):
    """
    Load one of the three locally trained models.

    Supported names:
        gru
        attention
        transformer
    """

    if name not in CHECKPOINTS:
        raise ValueError(
            f"Unknown local trained model: {name}"
        )

    with _cache_lock:

        # ----------------------------------------------------
        # RETURN CACHED MODEL
        # ----------------------------------------------------

        if name in _model_cache:
            return _model_cache[name]

        # ----------------------------------------------------
        # CHECKPOINT PATH
        # ----------------------------------------------------

        path = CHECKPOINTS[name]

        if not path.is_file():
            raise ModelUnavailable(
                f"Required checkpoint is missing: {path}"
            )

        # ----------------------------------------------------
        # READ CHECKPOINT
        # ----------------------------------------------------

        checkpoint = _read_checkpoint(
            path
        )

        try:

            # ==================================================
            # TRANSFORMER
            # ==================================================

            if name == "transformer":

                source, target = load_tokenizers(
                    improved=True
                )

                if "config" not in checkpoint:
                    raise ValueError(
                        "Transformer checkpoint "
                        "does not contain config."
                    )

                config = dict(
                    checkpoint["config"]
                )

                # Training checkpoint stores this name.
                if "max_seq_length" not in config:
                    raise ValueError(
                        "Transformer config does not "
                        "contain max_seq_length."
                    )

                max_length = config.pop(
                    "max_seq_length"
                )

                epoch = checkpoint.get(
                    "epoch",
                    -1,
                )

                # ------------------------------------------------
                # FINAL MODEL MUST BE EPOCH 19
                # ------------------------------------------------

                if epoch != 19:
                    raise ValueError(
                        "Wrong Transformer checkpoint. "
                        f"Expected selected Epoch 19, "
                        f"found Epoch {epoch}."
                    )

                model = TransformerFromScratch(
                    **config,
                    max_length=max_length,
                )

                checkpoint_vocabs = (
                    config["src_vocab_size"],
                    config["tgt_vocab_size"],
                )

            # ==================================================
            # GRU / ATTENTION
            # ==================================================

            else:

                source, target = load_tokenizers(
                    improved=False
                )

                if "model_config" not in checkpoint:
                    raise ValueError(
                        f"{name} checkpoint does not "
                        "contain model_config."
                    )

                config = dict(
                    checkpoint["model_config"]
                )

                # ------------------------------------------------
                # TOKENIZER CONFIG
                # ------------------------------------------------

                tokenizer_config_path = (
                    TOKENIZER_DIR
                    / "tokenizer_config.json"
                )

                if not tokenizer_config_path.is_file():
                    raise ValueError(
                        "tokenizer_config.json is missing."
                    )

                tokenizer_config = json.loads(
                    tokenizer_config_path.read_text(
                        encoding="utf-8"
                    )
                )

                max_length = config.get(
                    "max_sequence_length",
                    tokenizer_config[
                        "max_sequence_length"
                    ],
                )

                epoch = checkpoint.get(
                    "epoch",
                    -1,
                )

                common_args = {
                    "embedding_dim":
                        config["embedding_dim"],

                    "hidden_dim":
                        config["hidden_dim"],

                    "num_layers":
                        config["num_layers"],

                    "pad_id":
                        source.pad_id(),
                }

                checkpoint_vocabs = (
                    config["english_vocab_size"],
                    config["hindi_vocab_size"],
                )

                # ------------------------------------------------
                # GRU BASELINE
                # ------------------------------------------------

                if name == "gru":

                    encoder = EncoderGRU(
                        checkpoint_vocabs[0],
                        **common_args,
                    )

                    decoder = DecoderGRU(
                        checkpoint_vocabs[1],
                        **common_args,
                    )

                    model = Seq2SeqGRU(
                        encoder,
                        decoder,
                    )

                # ------------------------------------------------
                # GRU + BAHDANAU ATTENTION
                # ------------------------------------------------

                elif name == "attention":

                    encoder = EncoderGRUAttention(
                        checkpoint_vocabs[0],
                        **common_args,
                    )

                    decoder = DecoderGRUAttention(
                        checkpoint_vocabs[1],
                        attention_dim=config[
                            "attention_dim"
                        ],
                        **common_args,
                    )

                    model = Seq2SeqGRUAttention(
                        encoder,
                        decoder,
                    )

            # ==================================================
            # VERIFY VOCABULARY SIZES
            # ==================================================

            tokenizer_vocabs = (
                source.vocab_size(),
                target.vocab_size(),
            )

            if checkpoint_vocabs != tokenizer_vocabs:

                raise ValueError(
                    "Checkpoint/tokenizer vocabulary "
                    "mismatch. "
                    f"Checkpoint={checkpoint_vocabs}, "
                    f"Tokenizer={tokenizer_vocabs}"
                )

            # ==================================================
            # STRICT CHECKPOINT LOAD
            # ==================================================

            model.load_state_dict(
                checkpoint["model_state_dict"],
                strict=True,
            )

            # ==================================================
            # DEVICE + EVAL MODE
            # ==================================================

            model = model.to(
                DEVICE
            )

            model.eval()

            # ==================================================
            # CREATE MODEL BUNDLE
            # ==================================================

            bundle = ModelBundle(
                name=name,

                model=model,

                source=source,

                target=target,

                max_length=max_length,

                epoch=epoch,

                checkpoint=str(path),
            )

            # ==================================================
            # CACHE
            # ==================================================

            _model_cache[name] = bundle

            return bundle

        except ModelUnavailable:
            raise

        except Exception as exc:

            raise ModelUnavailable(
                f"Cannot load {name} from "
                f"{path.name}: {exc}"
            ) from exc


# ============================================================
# CONVENIENCE LOADERS
# ============================================================

def load_gru():
    return _load(
        "gru"
    )


def load_attention_model():
    return _load(
        "attention"
    )


def load_transformer():
    return _load(
        "transformer"
    )


# ============================================================
# GET LOCAL MODEL
# ============================================================

def get_model(name: str):
    """
    Return one of OUR locally trained models.

    NLLB is not handled here because it is loaded by
    nllb_service.py.
    """

    if name not in CHECKPOINTS:
        raise ValueError(
            f"Unknown trained local model: {name}"
        )

    return _load(
        name
    )


# ============================================================
# CACHE STATUS
# ============================================================

def get_loaded_models():
    """
    Names of local models currently loaded into memory.
    """

    with _cache_lock:
        return list(
            _model_cache.keys()
        )


# ============================================================
# AVAILABLE MODEL INFORMATION
# ============================================================

def get_available_models():
    """
    Return status information for all four selectable models.

    NLLB remains lazy:
    this function does NOT download NLLB.
    """

    items = []

    # --------------------------------------------------------
    # LOCAL TRAINED MODELS
    # --------------------------------------------------------

    for name in (
        "gru",
        "attention",
        "transformer",
    ):

        label = LABELS[name]

        suffix = (
            "_1m"
            if name == "transformer"
            else ""
        )

        tokenizer_present = all(
            (
                TOKENIZER_DIR
                / f"{language}_bpe{suffix}.model"
            ).is_file()

            for language in (
                "english",
                "hindi",
            )
        )

        checkpoint_present = (
            CHECKPOINTS[name].is_file()
        )

        available = (
            tokenizer_present
            and checkpoint_present
        )

        if name in get_loaded_models():

            status = "loaded"

        elif available:

            status = "ready"

        else:

            status = "missing required artifacts"

        items.append(
            {
                "name": name,

                "description": label,

                "available": available,

                "loaded":
                    name in get_loaded_models(),

                "status": status,

                "pretrained": False,
            }
        )

    # --------------------------------------------------------
    # NLLB PRETRAINED MODEL
    # --------------------------------------------------------

    status = nllb_status()

    transformers_installed = status[
        "transformers_installed"
    ]

    nllb_loaded = status[
        "loaded"
    ]

    if nllb_loaded:

        nllb_message = "loaded"

    elif transformers_installed:

        nllb_message = (
            "ready for lazy download/load"
        )

    else:

        nllb_message = (
            "transformers dependency missing"
        )

    items.append(
        {
            "name": "nllb",

            "description": LABELS["nllb"],

            # If transformers exists, the UI can allow selection.
            # The first translation request will download/load
            # model weights if they are not cached yet.
            "available": transformers_installed,

            "loaded": nllb_loaded,

            "status": nllb_message,

            "pretrained": True,

            "model_id":
                status["model"],

            "device":
                status["device"],
        }
    )

    return items