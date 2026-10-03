# Result provenance and unresolved evidence

All cell numbers below are zero-based. Reading notebook outputs does not re-execute them.

| Claim | Local evidence | Status |
|---|---|---|
| Final scratch checkpoint is Epoch 19 | Actual checkpoint metadata; improved notebook cell 56 | Verified |
| Scratch official BLEU 5.8019, chrF 28.7418; beam 4, penalty 0.6 | Improved notebook cell 57 saved output | Verified against locked request |
| NLLB official BLEU 23.4315, chrF 51.0445 | Pretrained notebook cell 4 saved output | Verified against locked request |
| Scratch greedy validation 4.9782 / 29.4144 | Improved notebook cell 54 saved output | Verified against locked request |
| Best beam validation 6.4504 / 30.0112 | Improved notebook cell 60 comparison table and user-locked result | Recorded, with qualification below |
| GRU official 1.3797 / 14.3026; Attention official 2.4331 / 18.9274 | User's locked instructions | User-supplied; separate official evaluation output not found locally |

## Beam validation discrepancy

Cell 55 prints BLEU 5.9770 and chrF 30.2137 for beam 4 / penalty 0.6. Cell 56 then explicitly reloads the Epoch-19 best checkpoint. Cell 60 evaluates beam sizes 5 and 6, but inserts the beam-4 result 6.4504 / 30.0112 as literal values rather than computing that row in the cell. Its saved summary selects that inserted beam-4 row.

The later reload means these outputs need not describe the same in-memory model, but the local notebook alone does not establish the original measured run that produced 6.4504 / 30.0112. That explanation is a possibility, not a verified reconciliation. The locked result remains exactly as supplied. No measurement is fabricated, overwritten, re-scored, or tuned. Final official test results and checkpoint selection have independent local support and are unaffected.

The original run record for the locked beam-validation row remains a manual provenance item for the project team. No official test rerun is needed or permitted for this application work.

## Validation CSVs are different experiments/splits

Existing GRU CSV: validation BLEU 1.6036412716065567, chrF 15.688910807270299. Existing Attention CSV: validation BLEU 3.241286018073338, chrF 20.62411438409161. These are explicitly validation fields, not substitutes for the locked official results. They remain unchanged.
