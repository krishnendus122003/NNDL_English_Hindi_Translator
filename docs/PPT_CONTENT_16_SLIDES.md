# Presentation content — exactly 16 slides

## Slide 1. Title

**English-to-Hindi Neural Machine Translation Chatbot**  
GRU → GRU + Bahdanau Attention → Transformer From Scratch  
MSc NNDL · [Team names / IDs] · [Institution / Supervisor]

Speaker note: Our application serves the models we trained and shows a separately labeled pretrained NLLB reference.

## Slide 2. Problem Statement & Objectives

- Translate an English sentence into Hindi despite word-order and morphology differences.
- Study recurrent encoder-decoder, additive attention and manual Transformer architectures.
- Compare completed experiments and make inference accessible through an interactive demo.
- Explain model limitations honestly; preserve trained artifacts and locked evaluation.

Speaker note: This is sentence translation with no conversational memory.

## Slide 3. Dataset & Data Split

- IIT Bombay English-Hindi parallel corpus.
- Saved raw split: 1,659,083 training, 520 validation, 2,507 test pairs.
- Fixed seed-42 training subsets: 300,000 pairs for original work, 1,000,000 for the improved Transformer.
- Use validation for selection; retain official test results without rerunning or tuning.

Visual: Three split boxes, with training branching into 300K and 1M subsets.

## Slide 4. Data Cleaning + SentencePiece BPE

- Normalize whitespace; remove empty and duplicate translation pairs.
- Retain 1–50 words in both languages during cleaning.
- Separate English/Hindi BPE vocabularies, each 16,000 pieces.
- PAD=0, UNK=1, BOS=2, EOS=3; 64-token model limit includes BOS/EOS.
- GRUs and the improved Transformer use their own verified tokenizer pairs.

Speaker note: Subword tokenization reduces whole-word sparsity but does not eliminate unknown input.

## Slide 5. Overall System Architecture

- English → SentencePiece → trained model → decoding → Hindi text.
- FastAPI validates requests and caches strict-loaded models.
- Frontend provides translation, comparison and attention alignment.
- JSONL records input, output, model, decoding, warning, latency and timestamp.
- Optional NLLB is a separate lazy pretrained branch.

Visual: Use the pipeline diagram from PROJECT_REPORT.md.

## Slide 6. GRU Encoder-Decoder

- 256-dimensional embeddings; one GRU layer; hidden dimension 512.
- Encoder final state initializes the autoregressive Hindi decoder.
- Output projection maps hidden state to 16,000 logits.
- 18,765,440 parameters; selected checkpoint Epoch 3; greedy inference.
- Fixed-state bottleneck can lose information on longer sequences.

Speaker note: Explain reset/update gates and teacher forcing versus inference.

## Slide 7. GRU + Bahdanau Attention

- Preserve all encoder states and score them against the decoder state.
- Mask padding, apply softmax and form a weighted context vector.
- Decoder consumes embedding plus context.
- Exact final projection: 1280 → 512 → 16000; hidden/attention dimensions 512.
- 20,733,569 parameters; selected checkpoint Epoch 4.

Speaker note: The checkpoint matches the later efficient decoder in the notebook.

## Slide 8. Attention Alignment

- X-axis: English SentencePiece tokens, including BOS/EOS and excluding PAD.
- Y-axis: displayed generated Hindi pieces.
- Cell intensity: genuine Bahdanau weight for that output step.
- Rows correspond exactly to generated target pieces.
- Alignment is useful for interpretation, not proof of translation correctness.

Demo visual: `docs/ui_attention.png` or show live “I am happy.” alignment.

## Slide 9. Transformer From Scratch

- Embeddings plus sinusoidal positional encoding.
- Manual Q/K/V projections and scaled dot-product attention.
- Split/concatenate eight heads; apply padding and causal masks.
- Encoder self-attention; decoder masked self-attention and cross-attention.
- Residuals, LayerNorm, FFN and vocabulary projection.
- No `nn.Transformer`, `nn.MultiheadAttention` or pretrained architecture substitute.

## Slide 10. Transformer Configuration & Training

- d_model=384; heads=8; encoder/decoder layers=4/4; FFN=1536.
- Dropout=0.1; vocabulary=16000/16000; max length=64.
- 35,012,224 parameters; 1M training pairs; selected checkpoint Epoch 19.
- Historical Adam, 4000-step warmup, label smoothing=0.1 and CUDA mixed precision where enabled.
- Final file: `transformer_1m_improved_best.pt`; fine-tuned checkpoint is not selected.

## Slide 11. Greedy vs Beam Search

- Greedy chooses one highest-scoring next token.
- Beam retains several candidate sequences; default width 4.
- Notebook ranking: cumulative log probability / length^0.6, excluding BOS from length.
- Both stop on EOS or a 63-generation-step limit.
- Beam costs more computation; it does not guarantee the best semantic translation.

Speaker note: Show both modes on a smoke sentence without claiming a new evaluation score.

## Slide 12. BLEU + chrF Evaluation

- BLEU: token n-gram overlap with brevity penalty.
- chrF: character n-gram precision/recall; useful for partial morphological matches.
- Scores are corpus metrics, not percentages of sentence accuracy.
- Scratch test: 2,507 pairs, Epoch 19, beam=4, penalty=0.6.
- Greedy validation: 4.9782 BLEU / 29.4144 chrF.
- Locked best beam validation: 6.4504 / 30.0112; historical run provenance is qualified in RESULT_PROVENANCE.md.

Speaker note: Earlier beam output differs; do not claim the inserted comparison row independently verifies its original measurement.

## Slide 13. Final Model Comparison + NLLB

| Trained by our team | BLEU | chrF |
|---|---:|---:|
| GRU | 1.3797 | 14.3026 |
| GRU + Bahdanau | 2.4331 | 18.9274 |
| Transformer From Scratch | 5.8019 | 28.7418 |

**Separate pretrained reference:** NLLB — Tool-Assisted: **23.4315 BLEU / 51.0445 chrF**.

Speaker note: NLLB was not trained by us. GRU official rows are user-locked; local GRU CSVs contain separate validation scores. Architecture and training scale both change, so this is not a controlled ablation.

## Slide 14. FastAPI + Chatbot Architecture

- `/health`, `/models`, `/translate`, `/attention`, `/compare` and root status.
- Pydantic rejects blank text, invalid models and unsupported decoding.
- Lazy strict checkpoint loading; eval mode; gradients disabled; CPU support.
- Portable paths, tokenizer checks and per-model comparison errors.
- Responsive browser UI defaults to scratch Transformer / beam.

Speaker note: API at port 8000, frontend at 5500; NLLB failure does not prevent core inference.

## Slide 15. Demo / Logging / Team Contribution

- Translate the three approved smoke sentences with all trained models.
- Show comparison, genuine heatmap, empty input and truncation warning.
- Inspect a local JSONL prediction record.
- Run automated tests and show verification evidence.
- Fill actual contributions: [Data], [GRU/Attention], [Transformer/Evaluation], [Application/Testing/Docs].

Speaker note: Do not promise a live NLLB demo unless dependencies and weights have been separately verified.

## Slide 16. Conclusion / Limitations / Future Work

- The completed project demonstrates recurrence, additive attention and a manual Transformer.
- The selected scratch model has the strongest locked scores among our trained models.
- Limitations: low absolute quality, short context, unfamiliar names, domain shifts and uneven comparisons.
- NLLB remains a separately labeled external reference.
- Future work: controlled ablations, broader data, human evaluation and longer contexts.
- Training artifacts and evaluation scores remain preserved.

Speaker note: Close with what was actually built and measured, and distinguish future proposals from completed work.
