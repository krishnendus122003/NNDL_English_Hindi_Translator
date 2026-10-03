# Verification report

Verified 2026-10-03. Core application and requested submission Markdown deliverables are complete. The only unresolved items are optional NLLB live inference and historical result provenance described below.

## Exact results

| Verification | Passed | Failed | Notes |
|---|---:|---:|---|
| `python -m unittest discover -s tests -v` | 32 | 0 | 0 errors, 0 skipped; final run 18.713 seconds |
| `python tests/browser_smoke.py` | 24 | 0 | Real local Edge, isolated headless context; 0 uncaught JS errors |
| Original-file integrity | 44 | 0 | SHA-256, size and modification timestamp all unchanged |
| Slide structure | 1 | 0 | Exactly Slide 1 through Slide 16, in order |
| JavaScript syntax | 1 | 0 | `node --check app/script.js` |
| PowerShell launcher syntax | 2 | 0 | Both launchers parsed without errors |

After the 24-check browser run, screenshot review found a missing space in the desktop heading. A one-space HTML correction was applied; a focused browser check confirmed the heading, refreshed desktop/mobile/attention screenshots and verified no mobile page overflow. No model or API code changed after the final 32-test run.

Evidence: `logs/test_results.log`, `docs/browser_verification.json`, `docs/ui_desktop.png`, `docs/ui_mobile.png`, `docs/ui_attention.png`. Screenshots were inspected visually. PowerShell represents native stderr from unittest as a NativeCommandError record when piped through Tee-Object; the final Python process exited 0 and unittest reported OK. This is shell output formatting, not a failed test.

## Model and tokenizer evidence

- GRU: strict load passed; Epoch 3; 18,765,440 parameters; original BPE pair.
- Bahdanau: strict load passed; Epoch 4; 20,733,569 parameters; original BPE pair; `pre_output` shape [512, 1280].
- Scratch Transformer: strict load passed; Epoch 19; 35,012,224 parameters; 1M BPE pair; exact selected file `saved_models/transformer_1m_improved/transformer_1m_improved_best.pt`.
- All four SentencePiece models load with vocabulary 16,000 and PAD/UNK/BOS/EOS IDs 0/1/2/3.
- Class ASTs match the corresponding notebook definitions, selecting the final attention decoder. No high-level Transformer modules are present.
- Isolated notebook inference functions match application greedy/beam translations on “I am happy.”; genuine attention weights match after trimming display-only rows/columns. No complete notebook cell or evaluation loop executes in these tests.

## Verified application behavior

The test suite and browser checks cover all core models on “I am happy.”, “How are you?” and “India is a diverse country.”, plus validation variants. They verify greedy and beam decoding, finite normalized attention rows and dimensions, source padding masking, target causal masking, EOS and generation limits, empty output warnings, invalid logits, unknown/unicode/emoji input handling, blank and oversized input, invalid model/decoding, model unavailability, local CORS and logging success/failure behavior.

All required HTTP endpoints respond: `/`, `/health`, `/models`, `/translate`, `/attention`, `/compare`. Startup succeeded. `/health` and the frontend returned HTTP 200 in the final live check. Model failure and optional NLLB failure are deliberately tested; they return controlled errors and leave the server/core models working. Optional-model errors can include a diagnostic traceback in backend logs; no unhandled startup or inference crash occurred.

UI checks exercised model selection, greedy/beam controls, loading completion, translation output, genuine heatmap, model attribution, comparisons, NLLB failure isolation, truncation warnings, Clear, empty-input errors, example buttons, Copy Translation, mobile overflow and simulated API error display. The backend wrote 83 prediction records at the integrity verification point, with later screenshot checks appending more records. This count is not an evaluation sample count.

## Environment and browser fallback

Existing Python 3.14.5 and torch 2.14.0+cpu worked. Core imports and inference succeeded; no Python downgrade or replacement environment was necessary. No `.venv` existed. Global packages were not changed. Playwright 1.63.0 and its dependencies were installed only under ignored `.test-tools/` for optional browser verification; the application does not need them.

The in-app Browser runtime failed before initialization with `helper_sandbox_lock_failed` / `SetNamedSecurityInfoW` error 5. Approved local Edge automation was used as the fallback, without accessing an existing user profile. CUDA was unavailable and is not claimed as tested. NLLB's `transformers` dependency is missing; no pretrained weights were downloaded, so live NLLB inference is unavailable and unverified.

## Live demo state

The backend was started as a hidden process (PID 22212), and the frontend as a hidden process (PID 4408). At final verification they serve:

- Backend: http://127.0.0.1:8000
- Health: http://127.0.0.1:8000/health
- Frontend: http://127.0.0.1:5500

These process IDs apply only to this session and can change on restart. Server logs are in `logs/backend.*.log` and `logs/frontend.*.log`. The verified background backend uses ordinary Uvicorn startup; `run_backend.ps1` adds the requested development `--reload` flag.

## Preservation and limitations

All 44 original files remain byte-for-byte identical with unchanged size and mtime. `notebooks/`, `saved_models/`, `tokenizers/` and `results/` were read only. `data/` was absent and was not recreated. No training, model experiment, official evaluation, checkpoint replacement, tokenization training, renaming, deletion or Git commit occurred.

Historical provenance limitations remain in `RESULT_PROVENANCE.md`: the locked beam-validation row is inserted in a later notebook comparison while an earlier saved beam run differs; independent local official-output files for the user-locked GRU official values were not found. No claim is made that these original measurement records were recovered. These limitations do not block verified inference or alter locked scores.

Manual submission actions: fill genuine team/institution details; locate the historical run records if available; choose checkpoint distribution using Git LFS or external storage before publishing. Optional: install/verify NLLB separately. No mandatory core implementation remains blocked.
