# Final locked results

These scores are copied from completed work and the user's locked submission record. No evaluation has been rerun. See [source evidence and qualifications](RESULT_PROVENANCE.md).

## Trained by our team — official comparison

| Model | BLEU | chrF | Selected checkpoint |
|---|---:|---:|---|
| GRU Baseline | 1.3797 | 14.3026 | `saved_models/gru/gru_seq2seq_best.pt` (Epoch 3) |
| GRU + Bahdanau Attention | 2.4331 | 18.9274 | `saved_models/gru_attention/gru_bahdanau_best.pt` (Epoch 4) |
| Transformer From Scratch | 5.8019 | 28.7418 | `saved_models/transformer_1m_improved/transformer_1m_improved_best.pt` (Epoch 19) |

Final scratch Transformer: 2,507 official test samples; beam size 4; length penalty 0.6. The GRU official values are user-locked; independent official-output files for them were not found locally. Existing GRU CSVs contain validation scores instead.

## Separate pretrained/tool-assisted reference

| Model | BLEU | chrF |
|---|---:|---:|
| NLLB — Tool-Assisted / Pretrained Reference (`facebook/nllb-200-distilled-600M`) | 23.4315 | 51.0445 |

NLLB was not trained by our team. Its results must never be attributed to the scratch Transformer.

## Scratch validation, separate from test results

| Decoding | BLEU | chrF |
|---|---:|---:|
| Greedy | 4.9782 | 29.4144 |
| Locked best validation beam: size 4, penalty 0.6 | 6.4504 | 30.0112 |

Qualification: the locked beam row is inserted into the later notebook comparison table. An earlier saved beam-4 run reports 5.9770 / 30.2137. The original measurement record for the locked row is not fully established locally; see RESULT_PROVENANCE.md. Preserve both historical evidence and locked numbers.

The fine-tuned model is not the selected final scratch model. BLEU and chrF measure reference overlap, not percentage translation accuracy. Smoke translations and application tests do not produce new corpus scores.
