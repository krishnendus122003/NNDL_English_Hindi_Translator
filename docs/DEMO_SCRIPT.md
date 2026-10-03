# Demo script

## Preparation

1. Open two PowerShell terminals in the project directory. Use the Python environment containing `requirements.txt` packages.
2. Run `.\run_backend.ps1` in one terminal and `.\run_frontend.ps1` in the other. If this session's servers are already running, reuse them instead of launching duplicates.
3. Open http://127.0.0.1:8000/health and confirm `status: ok`, CPU/device information and both tokenizer pairs. Models load lazily, so an initially empty loaded-model list is normal.
4. Open http://127.0.0.1:5500. The top bar should read “API connected”. Keep the terminals open. Do not use file://.
5. NLLB is optional and unavailable without its dependency/weights. Demonstrate the three core models regardless. Do not start a large download during a timed presentation unless already prepared.

## 0:00–0:30 — Introduce the project

“We compare three English-to-Hindi neural models that we trained, moving from GRU to Bahdanau attention to a Transformer built from basic components. This interface loads the actual preserved checkpoints. NLLB is an external pretrained reference.”

## 0:30–1:15 — Default scratch Transformer

Click “I am happy.” or enter it. The default model is Transformer From Scratch and decoding is Beam Search. Click Translate. Read the actual output; the local smoke run produced “मुझे खुशी है।” but do not promise identical wording for other models or inputs. Point out the model, beam label, latency and Copy Translation button. Explain that the first request can include checkpoint loading.

## 1:15–2:00 — Compare trained models

Click Compare Models with NLLB unchecked. Identify the GRU, attention GRU and scratch Transformer cards. Explain any weak or fragmented output honestly. Change input to “How are you?” and then “India is a diverse country.” Show selected translations. Never replace weak model outputs with a handwritten translation.

## 2:00–2:45 — Greedy and beam

Select Transformer From Scratch and switch between Greedy and Beam Search. Discuss local next-token choice versus multiple candidate prefixes and length normalization. Equal outputs on a short sentence are valid; do not imply beam must improve every sentence. No BLEU or chrF is computed for the demo input.

## 2:45–3:30 — Attention heatmap

Use “I am happy.” and click Show Attention. Point out the explicit Bahdanau model attribution, English source pieces across and generated Hindi pieces down. Darker cells indicate higher weights. BOS/EOS source positions are retained, padding is not shown and target rows align with generated pieces. These are actual model weights, not an illustrative fabricated grid.

## 3:30–4:00 — Validation and failure handling

Clear the input and click Translate to show the blank-input message. Repeat “I am happy.” sufficiently many times, then translate to show the truncation warning. If demonstrating optional failure, include NLLB in comparison: a friendly unavailable message should leave the three core results visible.

## 4:00–4:45 — Research results and logs

Point out the static results table. State “Our scratch Transformer: BLEU 5.8019, chrF 28.7418. The separate pretrained NLLB: BLEU 23.4315, chrF 51.0445.” Explain the 2,507-pair scratch official test, Epoch 19 and fixed beam configuration. Refer to RESULT_PROVENANCE.md if asked about validation history.

Show the most recent prediction with:

```powershell
Get-Content -LiteralPath logs/predictions.jsonl -Tail 1
```

It contains local submitted text, output, timestamp, model, decoding, warning and latency. Do not expose unrelated private inputs.

## 4:45–5:00 — Close

“The application uses exact, strict-loaded models, preserves the completed experiments and makes their behavior visible. Translation quality remains limited, and the pretrained reference is reported separately.”

## Troubleshooting

- API offline: start the backend, inspect its console and confirm port 8000. The UI retries the health check every 30 seconds.
- Port already occupied: reuse the existing demo or stop only the known demo process. Do not terminate unrelated Python processes.
- Missing file: read the exact API error and use the mapping in README. Never swap in the older or fine-tuned Transformer.
- Slow first request: wait for CPU checkpoint loading. Core inference is cached for later requests.
- Long input: explain the 64-token model limit; the app warns instead of hiding truncation.
- Optional NLLB failure: continue with the three mandatory models; do not present cached or invented output as a live result.
- PowerShell script restriction: use the equivalent Python commands from README; no permanent execution-policy change is required.

Before submission, replace team and institution placeholders, check attribution/provenance qualifications and decide Git LFS or external storage for checkpoints. A live NLLB download, actual PPTX formatting and institutional report formatting are optional manual follow-ups, not unfinished core inference.
