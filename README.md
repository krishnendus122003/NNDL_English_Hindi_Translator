# Transora — English to Hindi Neural Translator

Transora is an English-to-Hindi Neural Machine Translation project developed for MSc Neural Networks & Deep Learning.

The project follows the model progression:

**GRU Encoder-Decoder → GRU + Bahdanau Attention → Transformer From Scratch**

A FastAPI backend serves the trained models through a responsive web interface. The application supports translation, model comparison, greedy and beam decoding, and Bahdanau attention visualization.

A pretrained NLLB model is included separately as a reference model.

---

## Project Objective

The objective is to build and compare different neural machine translation architectures for English-to-Hindi translation.

The application allows users to:

- Enter an English sentence
- Select a translation model
- Generate a Hindi translation
- Compare model outputs
- Visualize Bahdanau attention weights
- Use greedy or beam-search decoding
- Compare trained models with a pretrained NLLB reference

---

## Dataset

The project uses the **IIT Bombay English-Hindi Parallel Corpus**.

| Split | Sentence Pairs |
|---|---:|
| Original Training Set | 1,659,083 |
| Validation Set | 520 |
| Official Test Set | 2,507 |

Training data preprocessing included:

- Removing empty sentence pairs
- Removing duplicate sentence pairs
- Normalizing whitespace
- Filtering unsuitable sentence lengths

Training subsets used:

- **300,000 pairs** for the original GRU-based models
- **1,000,000 pairs** for the improved Transformer experiment

---

## Tokenization

SentencePiece BPE was used for both English and Hindi.

```text
Vocabulary Size: 16,000 per language

PAD = 0
UNK = 1
BOS = 2
EOS = 3

Maximum Sequence Length = 64 tokens
```

Long inputs are safely truncated and a warning is returned.

---

## Models

### 1. GRU Baseline

The first model is a standard sequence-to-sequence architecture using:

- GRU encoder
- GRU decoder
- Teacher forcing during training
- Greedy decoding during inference

Architecture:

```text
English Sentence
      ↓
GRU Encoder
      ↓
Context Representation
      ↓
GRU Decoder
      ↓
Hindi Translation
```

---

### 2. GRU + Bahdanau Attention

The second model extends the GRU baseline with Bahdanau Attention.

Instead of depending only on a fixed encoder representation, the decoder dynamically attends to different encoder outputs while generating each Hindi token.

Architecture:

```text
English Sentence
      ↓
GRU Encoder
      ↓
Encoder Hidden States
      ↓
Bahdanau Attention
      ↓
Context Vector
      ↓
GRU Decoder
      ↓
Hindi Translation
```

The application also displays attention weights as a heatmap.

---

### 3. Transformer From Scratch

The Transformer architecture was implemented manually.

The implementation does not use:

```python
torch.nn.Transformer
```

or:

```python
torch.nn.MultiheadAttention
```

The implementation includes:

- Token embeddings
- Sinusoidal positional encoding
- Query, Key and Value projections
- Scaled dot-product attention
- Multi-head attention
- Padding masks
- Causal masks
- Encoder layers
- Decoder layers
- Masked self-attention
- Cross-attention
- Residual connections
- Layer Normalization
- Feed-forward networks
- Vocabulary projection

### Transformer Configuration

```text
d_model = 384
Attention Heads = 8
Encoder Layers = 4
Decoder Layers = 4
Feed Forward Dimension = 1536
Dropout = 0.1
Maximum Length = 64
Selected Checkpoint = Epoch 19
```

---

## Decoding

### GRU Baseline

```text
Greedy Decoding
```

### GRU + Bahdanau Attention

```text
Greedy Decoding
```

### Transformer From Scratch

```text
Greedy Decoding
Beam Search
```

Beam-search configuration:

```text
Beam Size = 4
Length Penalty = 0.6
```

---

## Evaluation Metrics

The models were evaluated using **BLEU** and **chrF**.

### BLEU

BLEU measures token-level n-gram overlap between generated translations and reference translations.

### chrF

chrF measures character-level n-gram precision and recall.

BLEU and chrF are translation evaluation metrics and should not be interpreted as accuracy percentages.

---

## Official Test Results

Official test set size:

```text
2,507 sentence pairs
```

| Model | BLEU | chrF |
|---|---:|---:|
| GRU Baseline | 1.3797 | 14.3026 |
| GRU + Bahdanau Attention | 2.4331 | 18.9274 |
| Transformer From Scratch | 5.8019 | 28.7418 |

The trained models show progressive improvement:

```text
GRU
 ↓
GRU + Bahdanau Attention
 ↓
Transformer From Scratch
```

---

## Pretrained NLLB Reference

The project also includes the pretrained model:

```text
facebook/nllb-200-distilled-600M
```

Language codes:

```text
English: eng_Latn
Hindi: hin_Deva
```

Official test result:

| Model | BLEU | chrF |
|---|---:|---:|
| NLLB Pretrained Reference | 23.4315 | 51.0445 |

NLLB was not trained as part of this project.

It is used as a pretrained reference for comparison with the locally trained models.

The Transformer From Scratch remains a separate trained and evaluated project model.

---

## Web Application

The Transora interface supports:

- English text input
- Hindi translation output
- Model selection
- Decoding selection
- GRU Baseline
- GRU + Bahdanau Attention
- Transformer From Scratch
- NLLB reference
- Model comparison
- Attention heatmap
- Example sentences
- Copy translation
- Clear input
- Long-input warnings
- Error handling
- The frontend communicates with the FastAPI backend for translation, model comparison, attention visualization, and input validation.
---

## Attention Visualization

The attention heatmap uses the **GRU + Bahdanau Attention** model.

The source English tokens and generated Hindi tokens are displayed with their attention weights.

Each attention row represents how strongly the decoder focused on different source tokens while generating a target token.

Attention weights help visualize alignment but do not guarantee translation correctness.

---

## Backend

The backend is implemented using **FastAPI**.

### API Routes

| Route | Purpose |
|---|---|
| `GET /` | API status |
| `GET /health` | Application and tokenizer status |
| `GET /models` | Available models |
| `POST /translate` | Translate English to Hindi |
| `POST /attention` | Generate attention alignment |
| `POST /compare` | Compare model translations |

---

## Example API Request

```json
{
  "text": "Where are you going?",
  "model": "transformer",
  "decoding": "beam"
}
```

---

## Project Structure

```text
NNDL_English_Hindi_Translator/
│
├── api/
│   ├── main.py
│   ├── schemas.py
│   ├── model_loader.py
│   ├── model_classes.py
│   ├── decoding.py
│   ├── translation_service.py
│   ├── nllb_service.py
│   └── logging_service.py
│
├── app/
│   ├── index.html
│   ├── style.css
│   ├── script.js
│   ├── india-gate-bg.png
│   └── transora-logo.png
│
├── notebooks/
├── results/
├── saved_models/
├── tokenizers/
├── tests/
├── docs/
├── scripts/
│
├── requirements.txt
├── requirements-optional.txt
├── run_backend.ps1
├── run_frontend.ps1
├── .gitignore
└── README.md
```

---

## Model Checkpoints

The application uses the following trained checkpoints locally.

### GRU

```text
saved_models/gru/gru_seq2seq_best.pt
```

### GRU + Bahdanau Attention

```text
saved_models/gru_attention/gru_bahdanau_best.pt
```

### Transformer From Scratch

```text
saved_models/transformer_1m_improved/transformer_1m_improved_best.pt
```

The checkpoint files are large and are excluded from normal Git tracking.

They remain available locally and can be distributed separately when required.

---

## Tokenizers

GRU and GRU + Bahdanau Attention use:

```text
tokenizers/english_bpe.model
tokenizers/hindi_bpe.model
```

Transformer From Scratch uses:

```text
tokenizers/english_bpe_1m.model
tokenizers/hindi_bpe_1m.model
```

---

## Installation

Install the required dependencies:

```powershell
python -m pip install -r requirements.txt
```

---

## Run Backend

Open PowerShell in the project directory and run:

```powershell
.\run_backend.ps1
```

Backend:

```text
http://127.0.0.1:8000
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

---

## Run Frontend

Open another PowerShell terminal and run:

```powershell
.\run_frontend.ps1
```

Open:

```text
http://127.0.0.1:5500
```

---

## Example Sentences

```text
I am happy.
How are you?
Where are you going?
India is a diverse country.
```

---

## Automated Testing

Run:

```powershell
python -m pytest -q
```

Final verified result:

```text
................................
32 passed
```

The automated tests cover:

- API health
- Model loading
- GRU inference
- Attention inference
- Transformer greedy decoding
- Transformer beam search
- Model comparison
- Attention matrix validation
- Input validation
- Long-input handling
- Unicode handling
- Logging
- Model failure handling
- HTTP error handling
- CORS

---

## Limitations

The locally trained models have limitations including:

- Low translation quality for the GRU baseline
- Unfamiliar vocabulary
- Names and domain-specific words
- Long sentences
- Maximum sequence length of 64 tokens
- CPU inference latency

The Transformer was trained on a larger training subset than the original GRU models, so the comparison is not a completely controlled architecture-only experiment.

NLLB uses much larger multilingual pretraining and is therefore treated separately as a pretrained reference.

---

## Team

This project is completed by two students.

| Member | Responsibilities |
|---|---|
| Member A — [Name / Student ID] | Dataset preparation, preprocessing, tokenization, model training, experiments and evaluation |
| Member B — [Name / Student ID] | API development, frontend integration, testing, logging and demo preparation |

Both members contribute to the final report, presentation and project explanation.

---

## Final Status

Completed components:

- GRU Baseline
- GRU + Bahdanau Attention
- Transformer From Scratch
- SentencePiece BPE
- Greedy Decoding
- Beam Search
- Attention Visualization
- BLEU Evaluation
- chrF Evaluation
- NLLB Pretrained Reference
- FastAPI Backend
- Transora Frontend
- Model Comparison
- Input Validation
- Error Handling
- Prediction Logging
- Automated Testing

Final test status:

```text
32 / 32 tests passed
```