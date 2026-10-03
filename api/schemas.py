"""
Pydantic request/response schemas for the translation API.

Supported models:
- gru
- attention
- transformer
- nllb
"""

from typing import Literal, Optional

from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
)

from .config import MAX_INPUT_CHARS


# ============================================================
# ALLOWED VALUES
# ============================================================

ModelName = Literal[
    "gru",
    "attention",
    "transformer",
    "nllb",
]

DecodingMethod = Literal[
    "greedy",
    "beam",
]


# ============================================================
# SHARED TEXT VALIDATION
# ============================================================

def validate_english_text(value: str):
    """
    Validate user text before it reaches a model.
    """

    cleaned = value.strip()

    if not cleaned:
        raise ValueError(
            "Please enter an English sentence."
        )

    # Reject null characters.
    if "\x00" in cleaned:
        raise ValueError(
            "Input contains an invalid null character."
        )

    # Reject isolated Unicode surrogate characters.
    if any(
        0xD800 <= ord(character) <= 0xDFFF
        for character in cleaned
    ):
        raise ValueError(
            "Input contains invalid Unicode characters."
        )

    return cleaned


# ============================================================
# TRANSLATION REQUEST
# ============================================================

class TranslationRequest(BaseModel):
    """
    Request sent by the frontend to /translate.

    Behaviour:
    - GRU -> greedy by default
    - Attention -> greedy by default
    - Transformer -> beam by default
    - NLLB -> beam by default

    Explicit beam decoding is not allowed for GRU/Attention.
    """

    text: str = Field(
        ...,
        min_length=1,
        max_length=MAX_INPUT_CHARS,
        description="English text to translate.",
    )

    model: ModelName = Field(
        default="transformer"
    )

    decoding: DecodingMethod = Field(
        default="beam"
    )


    # --------------------------------------------------------
    # SET MODEL-SPECIFIC DEFAULT DECODING
    # --------------------------------------------------------

    @model_validator(mode="before")
    @classmethod
    def set_default_decoding(cls, values):
        """
        If decoding was not explicitly supplied:

        GRU / Attention -> greedy
        Transformer / NLLB -> beam
        """

        if not isinstance(values, dict):
            return values

        model = values.get(
            "model",
            "transformer",
        )

        # Only modify decoding when user/API did NOT
        # explicitly provide a decoding value.
        if "decoding" not in values:

            if model in {
                "gru",
                "attention",
            }:

                values["decoding"] = "greedy"

            else:

                values["decoding"] = "beam"

        return values


    # --------------------------------------------------------
    # TEXT VALIDATION
    # --------------------------------------------------------

    @field_validator("text")
    @classmethod
    def check_text(cls, value):

        return validate_english_text(
            value
        )


    # --------------------------------------------------------
    # MODEL / DECODING VALIDATION
    # --------------------------------------------------------

    @model_validator(mode="after")
    def validate_model_decoding(self):
        """
        GRU and GRU + Attention support greedy decoding only.
        """

        if (
            self.model in {
                "gru",
                "attention",
            }
            and self.decoding != "greedy"
        ):

            raise ValueError(
                f"{self.model} supports greedy decoding only."
            )

        return self


# ============================================================
# TRANSLATION RESULT
# ============================================================

class TranslationResult(BaseModel):
    """
    Standard translation response.

    Attention-specific fields remain optional.
    """

    input_text: str

    translation: str

    model: ModelName

    model_label: str

    decoding: DecodingMethod

    latency_ms: float

    warning: Optional[str] = None


    # GRU + Bahdanau Attention only

    source_tokens: Optional[
        list[str]
    ] = None

    target_tokens: Optional[
        list[str]
    ] = None

    attention_matrix: Optional[
        list[list[float]]
    ] = None


# ============================================================
# ATTENTION REQUEST
# ============================================================

class AttentionRequest(BaseModel):
    """
    Request for a genuine Bahdanau attention alignment.
    """

    text: str = Field(
        ...,
        min_length=1,
        max_length=MAX_INPUT_CHARS,
    )

    @field_validator("text")
    @classmethod
    def check_text(cls, value):

        return validate_english_text(
            value
        )


# ============================================================
# COMPARE REQUEST
# ============================================================

class CompareRequest(BaseModel):
    """
    Request used by Compare Models.

    Core models:
    - GRU
    - GRU + Attention
    - Transformer

    NLLB is optional.
    """

    text: str = Field(
        ...,
        min_length=1,
        max_length=MAX_INPUT_CHARS,
    )

    decoding: DecodingMethod = Field(
        default="beam"
    )

    include_nllb: bool = Field(
        default=False
    )


    @field_validator("text")
    @classmethod
    def check_text(cls, value):

        return validate_english_text(
            value
        )