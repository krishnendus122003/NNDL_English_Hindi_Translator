# Workspace audit

Date: 2026-10-03 (Asia/Calcutta). Inspection completed before application implementation.

## Initial complete folder structure

```

notebooks/

notebooks\01_Data_Verification_and_EDA.ipynb (342950 bytes)

notebooks\02_Data_Cleaning.ipynb (72773 bytes)

notebooks\03_SentencePiece_BPE.ipynb (44428 bytes)

notebooks\04_GRU_Seq2Seq.ipynb (153870 bytes)

notebooks\05_GRU_Bahdanau_Attention.ipynb (226817 bytes)

notebooks\06_Transformer_From_Scratch.ipynb (144148 bytes)

notebooks\07_Decoding_and_Evaluation.ipynb (13365 bytes)

notebooks\07_Transformer_1M_Improved.ipynb (279742 bytes)

notebooks\08_Pretrained_Tool_Assisted_Translation.ipynb (223270 bytes)

results/

results\attention_maps/

results\attention_maps\attention_map_i_am_happy.png (110571 bytes)

results\attention_maps\attention_map_i_am_happy_final.png (126035 bytes)

results\attention_maps\attention_map_i_am_happy_fixed_font.png (64892 bytes)

results\metrics/

results\metrics\dataset_quality_summary.csv (168 bytes)

results\metrics\gru_attention_training_history.csv (445 bytes)

results\metrics\gru_attention_validation_metrics.csv (187 bytes)

results\metrics\gru_attention_validation_predictions.csv (283125 bytes)

results\metrics\gru_seq2seq_training_history.csv (129 bytes)

results\metrics\gru_seq2seq_validation_metrics.csv (185 bytes)

results\metrics\gru_validation_predictions.csv (278405 bytes)

results\metrics\tokenizer_length_statistics.csv (243 bytes)

results\metrics\transformer_from_scratch_training_history.csv (1339 bytes)

results\metrics\transformer_training_history.csv (616 bytes)

results\plots/

results\plots\english_sentence_length_distribution.png (80272 bytes)

results\plots\gru_seq2seq_loss_curve.png (119737 bytes)

results\plots\hindi_sentence_length_distribution.png (80491 bytes)

saved_models/

saved_models\gru/

saved_models\gru\gru_seq2seq_best.pt (225199829 bytes)

saved_models\gru_attention/

saved_models\gru_attention\gru_bahdanau_best.pt (248827109 bytes)

saved_models\transformer/

saved_models\transformer\transformer_best.pt (214206196 bytes)

saved_models\transformer\transformer_from_scratch_best.pt (214242615 bytes)

saved_models\transformer_1m_improved/

saved_models\transformer_1m_improved\transformer_1m_improved_best.pt (420473632 bytes)

saved_models\transformer_1m_improved\transformer_1m_resume.pt (420470431 bytes)

tokenizers/

tokenizers\english_bpe.model (503956 bytes)

tokenizers\english_bpe.vocab (236566 bytes)

tokenizers\english_bpe_1m.model (503959 bytes)

tokenizers\english_bpe_1m.vocab (236566 bytes)

tokenizers\english_training_corpus.txt (21366685 bytes)

tokenizers\english_training_corpus_1m.txt (71320039 bytes)

tokenizers\hindi_bpe.model (623633 bytes)

tokenizers\hindi_bpe.vocab (356250 bytes)

tokenizers\hindi_bpe_1m.model (622671 bytes)

tokenizers\hindi_bpe_1m.vocab (355282 bytes)

tokenizers\hindi_training_corpus.txt (52294445 bytes)

tokenizers\hindi_training_corpus_1m.txt (174548239 bytes)

tokenizers\tokenizer_config.json (374 bytes)

```

## Environment

Python: 3.14.5 (tags/v3.14.5:5607950, May 10 2026, 10:43:50) [MSC v.1944 64 bit (AMD64)]; executable: C:\Users\Lenovo\AppData\Local\Programs\Python\Python314\python.exe

Active virtual environment: none. No .venv exists.

Python launcher also lists 3.12. Core imports succeed on 3.14; no incompatibility or replacement environment required.

CPU torch build; CUDA available: False

Windows sandbox process startup failed with SetNamedSecurityInfoW error 5. Approved shell execution used.

No Git repository or AGENTS.md found.

## Packages

- torch: 2.14.0

- sentencepiece: 0.2.2

- fastapi: 0.141.1

- uvicorn: 0.54.0

- pydantic: 2.13.5

- numpy: 2.5.2

- transformers: MISSING

- pytest: MISSING

- httpx: MISSING

## Missing application and optional source files

api/, app/, tests/, logs/, docs/, README, requirements and run scripts were absent. No existing application logic to reuse.

data/ and the optional fine-tuned checkpoint are absent; neither is needed for inference. Required inference files all exist. Only mounted filesystem drive is C:; no mounted Google Drive found. No source files need copying.

Separate final official-result CSV files are absent locally; use locked user results and notebook outputs, never overwrite validation CSVs.

## Checkpoint evidence (CPU, weights_only=True)

### saved_models\gru\gru_seq2seq_best.pt

Top keys: epoch, model_state_dict, optimizer_state_dict, validation_loss, model_config

```json

{
  "epoch": 3,
  "validation_loss": 6.664642202495732,
  "model_config": {
    "embedding_dim": 256,
    "hidden_dim": 512,
    "num_layers": 1,
    "english_vocab_size": 16000,
    "hindi_vocab_size": 16000,
    "max_sequence_length": 64
  }
}

```

State prefixes: decoder, encoder

```

encoder.embedding.weight: [16000, 256]

encoder.gru.weight_ih_l0: [1536, 256]

encoder.gru.weight_hh_l0: [1536, 512]

encoder.gru.bias_ih_l0: [1536]

encoder.gru.bias_hh_l0: [1536]

decoder.embedding.weight: [16000, 256]

decoder.gru.weight_ih_l0: [1536, 256]

decoder.gru.weight_hh_l0: [1536, 512]

decoder.gru.bias_ih_l0: [1536]

decoder.gru.bias_hh_l0: [1536]

decoder.output_layer.weight: [16000, 512]

decoder.output_layer.bias: [16000]

```

### saved_models\gru_attention\gru_bahdanau_best.pt

Top keys: epoch, model_state_dict, optimizer_state_dict, validation_loss, model_config

```json

{
  "epoch": 4,
  "validation_loss": 6.520715561180102,
  "model_config": {
    "english_vocab_size": 16000,
    "hindi_vocab_size": 16000,
    "embedding_dim": 256,
    "hidden_dim": 512,
    "attention_dim": 512,
    "num_layers": 1,
    "pad_id": 0,
    "batch_size": 128
  }
}

```

State prefixes: decoder, encoder

```

encoder.embedding.weight: [16000, 256]

encoder.gru.weight_ih_l0: [1536, 256]

encoder.gru.weight_hh_l0: [1536, 512]

encoder.gru.bias_ih_l0: [1536]

encoder.gru.bias_hh_l0: [1536]

decoder.embedding.weight: [16000, 256]

decoder.attention.W_encoder.weight: [512, 512]

decoder.attention.W_encoder.bias: [512]

decoder.attention.W_decoder.weight: [512, 512]

decoder.attention.W_decoder.bias: [512]

decoder.attention.V.weight: [1, 512]

decoder.attention.V.bias: [1]

decoder.gru.weight_ih_l0: [1536, 768]

decoder.gru.weight_hh_l0: [1536, 512]

decoder.gru.bias_ih_l0: [1536]

decoder.gru.bias_hh_l0: [1536]

decoder.pre_output.weight: [512, 1280]

decoder.pre_output.bias: [512]

decoder.output_layer.weight: [16000, 512]

decoder.output_layer.bias: [16000]

```

### saved_models\transformer\transformer_best.pt

Top keys: epoch, model_state_dict, optimizer_state_dict, validation_loss, model_config

```json

{
  "epoch": 10,
  "validation_loss": 4.517586517333984,
  "model_config": {
    "src_vocab_size": 16000,
    "tgt_vocab_size": 16000,
    "d_model": 256,
    "num_heads": 8,
    "num_encoder_layers": 3,
    "num_decoder_layers": 3,
    "dim_feedforward": 1024,
    "dropout": 0.1,
    "pad_id": 0,
    "batch_size": 128
  }
}

```

State prefixes: output_layer, positional_encoding, src_embedding, tgt_embedding, transformer

```

src_embedding.weight: [16000, 256]

tgt_embedding.weight: [16000, 256]

positional_encoding.positional_encoding: [1, 64, 256]

transformer.encoder.layers.0.self_attn.in_proj_weight: [768, 256]

transformer.encoder.layers.0.self_attn.in_proj_bias: [768]

transformer.encoder.layers.0.self_attn.out_proj.weight: [256, 256]

transformer.encoder.layers.0.self_attn.out_proj.bias: [256]

transformer.encoder.layers.0.linear1.weight: [1024, 256]

transformer.encoder.layers.0.linear1.bias: [1024]

transformer.encoder.layers.0.linear2.weight: [256, 1024]

transformer.encoder.layers.0.linear2.bias: [256]

transformer.encoder.layers.0.norm1.weight: [256]

transformer.encoder.layers.0.norm1.bias: [256]

transformer.encoder.layers.0.norm2.weight: [256]

transformer.encoder.layers.0.norm2.bias: [256]

transformer.encoder.layers.1.self_attn.in_proj_weight: [768, 256]

transformer.encoder.layers.1.self_attn.in_proj_bias: [768]

transformer.encoder.layers.1.self_attn.out_proj.weight: [256, 256]

transformer.encoder.layers.1.self_attn.out_proj.bias: [256]

transformer.encoder.layers.1.linear1.weight: [1024, 256]

transformer.encoder.layers.1.linear1.bias: [1024]

transformer.encoder.layers.1.linear2.weight: [256, 1024]

transformer.encoder.layers.1.linear2.bias: [256]

transformer.encoder.layers.1.norm1.weight: [256]

transformer.encoder.layers.1.norm1.bias: [256]

transformer.encoder.layers.1.norm2.weight: [256]

transformer.encoder.layers.1.norm2.bias: [256]

transformer.encoder.layers.2.self_attn.in_proj_weight: [768, 256]

transformer.encoder.layers.2.self_attn.in_proj_bias: [768]

transformer.encoder.layers.2.self_attn.out_proj.weight: [256, 256]

transformer.encoder.layers.2.self_attn.out_proj.bias: [256]

transformer.encoder.layers.2.linear1.weight: [1024, 256]

transformer.encoder.layers.2.linear1.bias: [1024]

transformer.encoder.layers.2.linear2.weight: [256, 1024]

transformer.encoder.layers.2.linear2.bias: [256]

transformer.encoder.layers.2.norm1.weight: [256]

transformer.encoder.layers.2.norm1.bias: [256]

transformer.encoder.layers.2.norm2.weight: [256]

transformer.encoder.layers.2.norm2.bias: [256]

transformer.encoder.norm.weight: [256]

transformer.encoder.norm.bias: [256]

transformer.decoder.layers.0.self_attn.in_proj_weight: [768, 256]

transformer.decoder.layers.0.self_attn.in_proj_bias: [768]

transformer.decoder.layers.0.self_attn.out_proj.weight: [256, 256]

transformer.decoder.layers.0.self_attn.out_proj.bias: [256]

transformer.decoder.layers.0.multihead_attn.in_proj_weight: [768, 256]

transformer.decoder.layers.0.multihead_attn.in_proj_bias: [768]

transformer.decoder.layers.0.multihead_attn.out_proj.weight: [256, 256]

transformer.decoder.layers.0.multihead_attn.out_proj.bias: [256]

transformer.decoder.layers.0.linear1.weight: [1024, 256]

transformer.decoder.layers.0.linear1.bias: [1024]

transformer.decoder.layers.0.linear2.weight: [256, 1024]

transformer.decoder.layers.0.linear2.bias: [256]

transformer.decoder.layers.0.norm1.weight: [256]

transformer.decoder.layers.0.norm1.bias: [256]

transformer.decoder.layers.0.norm2.weight: [256]

transformer.decoder.layers.0.norm2.bias: [256]

transformer.decoder.layers.0.norm3.weight: [256]

transformer.decoder.layers.0.norm3.bias: [256]

transformer.decoder.layers.1.self_attn.in_proj_weight: [768, 256]

transformer.decoder.layers.1.self_attn.in_proj_bias: [768]

transformer.decoder.layers.1.self_attn.out_proj.weight: [256, 256]

transformer.decoder.layers.1.self_attn.out_proj.bias: [256]

transformer.decoder.layers.1.multihead_attn.in_proj_weight: [768, 256]

transformer.decoder.layers.1.multihead_attn.in_proj_bias: [768]

transformer.decoder.layers.1.multihead_attn.out_proj.weight: [256, 256]

transformer.decoder.layers.1.multihead_attn.out_proj.bias: [256]

transformer.decoder.layers.1.linear1.weight: [1024, 256]

transformer.decoder.layers.1.linear1.bias: [1024]

transformer.decoder.layers.1.linear2.weight: [256, 1024]

transformer.decoder.layers.1.linear2.bias: [256]

transformer.decoder.layers.1.norm1.weight: [256]

transformer.decoder.layers.1.norm1.bias: [256]

transformer.decoder.layers.1.norm2.weight: [256]

transformer.decoder.layers.1.norm2.bias: [256]

transformer.decoder.layers.1.norm3.weight: [256]

transformer.decoder.layers.1.norm3.bias: [256]

transformer.decoder.layers.2.self_attn.in_proj_weight: [768, 256]

transformer.decoder.layers.2.self_attn.in_proj_bias: [768]

transformer.decoder.layers.2.self_attn.out_proj.weight: [256, 256]

transformer.decoder.layers.2.self_attn.out_proj.bias: [256]

transformer.decoder.layers.2.multihead_attn.in_proj_weight: [768, 256]

transformer.decoder.layers.2.multihead_attn.in_proj_bias: [768]

transformer.decoder.layers.2.multihead_attn.out_proj.weight: [256, 256]

transformer.decoder.layers.2.multihead_attn.out_proj.bias: [256]

transformer.decoder.layers.2.linear1.weight: [1024, 256]

transformer.decoder.layers.2.linear1.bias: [1024]

transformer.decoder.layers.2.linear2.weight: [256, 1024]

transformer.decoder.layers.2.linear2.bias: [256]

transformer.decoder.layers.2.norm1.weight: [256]

transformer.decoder.layers.2.norm1.bias: [256]

transformer.decoder.layers.2.norm2.weight: [256]

transformer.decoder.layers.2.norm2.bias: [256]

transformer.decoder.layers.2.norm3.weight: [256]

transformer.decoder.layers.2.norm3.bias: [256]

transformer.decoder.norm.weight: [256]

transformer.decoder.norm.bias: [256]

output_layer.weight: [16000, 256]

output_layer.bias: [16000]

```

### saved_models\transformer\transformer_from_scratch_best.pt

Top keys: epoch, validation_loss, random_seed, model_state_dict, optimizer_state_dict, scaler_state_dict, architecture, model_config

```json

{
  "epoch": 20,
  "validation_loss": 4.3022888660430905,
  "random_seed": 42,
  "architecture": "TransformerFromScratch",
  "model_config": {
    "src_vocab_size": 16000,
    "tgt_vocab_size": 16000,
    "d_model": 256,
    "num_heads": 8,
    "num_encoder_layers": 3,
    "num_decoder_layers": 3,
    "dim_feedforward": 1024,
    "dropout": 0.1,
    "pad_id": 0,
    "max_length": 64,
    "batch_size": 128
  }
}

```

State prefixes: decoder_layers, encoder_layers, output_layer, positional_encoding, src_embedding, tgt_embedding

```

src_embedding.weight: [16000, 256]

tgt_embedding.weight: [16000, 256]

positional_encoding.positional_encoding: [1, 64, 256]

encoder_layers.0.self_attention.W_q.weight: [256, 256]

encoder_layers.0.self_attention.W_q.bias: [256]

encoder_layers.0.self_attention.W_k.weight: [256, 256]

encoder_layers.0.self_attention.W_k.bias: [256]

encoder_layers.0.self_attention.W_v.weight: [256, 256]

encoder_layers.0.self_attention.W_v.bias: [256]

encoder_layers.0.self_attention.W_o.weight: [256, 256]

encoder_layers.0.self_attention.W_o.bias: [256]

encoder_layers.0.feed_forward.linear1.weight: [1024, 256]

encoder_layers.0.feed_forward.linear1.bias: [1024]

encoder_layers.0.feed_forward.linear2.weight: [256, 1024]

encoder_layers.0.feed_forward.linear2.bias: [256]

encoder_layers.0.norm1.weight: [256]

encoder_layers.0.norm1.bias: [256]

encoder_layers.0.norm2.weight: [256]

encoder_layers.0.norm2.bias: [256]

encoder_layers.1.self_attention.W_q.weight: [256, 256]

encoder_layers.1.self_attention.W_q.bias: [256]

encoder_layers.1.self_attention.W_k.weight: [256, 256]

encoder_layers.1.self_attention.W_k.bias: [256]

encoder_layers.1.self_attention.W_v.weight: [256, 256]

encoder_layers.1.self_attention.W_v.bias: [256]

encoder_layers.1.self_attention.W_o.weight: [256, 256]

encoder_layers.1.self_attention.W_o.bias: [256]

encoder_layers.1.feed_forward.linear1.weight: [1024, 256]

encoder_layers.1.feed_forward.linear1.bias: [1024]

encoder_layers.1.feed_forward.linear2.weight: [256, 1024]

encoder_layers.1.feed_forward.linear2.bias: [256]

encoder_layers.1.norm1.weight: [256]

encoder_layers.1.norm1.bias: [256]

encoder_layers.1.norm2.weight: [256]

encoder_layers.1.norm2.bias: [256]

encoder_layers.2.self_attention.W_q.weight: [256, 256]

encoder_layers.2.self_attention.W_q.bias: [256]

encoder_layers.2.self_attention.W_k.weight: [256, 256]

encoder_layers.2.self_attention.W_k.bias: [256]

encoder_layers.2.self_attention.W_v.weight: [256, 256]

encoder_layers.2.self_attention.W_v.bias: [256]

encoder_layers.2.self_attention.W_o.weight: [256, 256]

encoder_layers.2.self_attention.W_o.bias: [256]

encoder_layers.2.feed_forward.linear1.weight: [1024, 256]

encoder_layers.2.feed_forward.linear1.bias: [1024]

encoder_layers.2.feed_forward.linear2.weight: [256, 1024]

encoder_layers.2.feed_forward.linear2.bias: [256]

encoder_layers.2.norm1.weight: [256]

encoder_layers.2.norm1.bias: [256]

encoder_layers.2.norm2.weight: [256]

encoder_layers.2.norm2.bias: [256]

decoder_layers.0.self_attention.W_q.weight: [256, 256]

decoder_layers.0.self_attention.W_q.bias: [256]

decoder_layers.0.self_attention.W_k.weight: [256, 256]

decoder_layers.0.self_attention.W_k.bias: [256]

decoder_layers.0.self_attention.W_v.weight: [256, 256]

decoder_layers.0.self_attention.W_v.bias: [256]

decoder_layers.0.self_attention.W_o.weight: [256, 256]

decoder_layers.0.self_attention.W_o.bias: [256]

decoder_layers.0.cross_attention.W_q.weight: [256, 256]

decoder_layers.0.cross_attention.W_q.bias: [256]

decoder_layers.0.cross_attention.W_k.weight: [256, 256]

decoder_layers.0.cross_attention.W_k.bias: [256]

decoder_layers.0.cross_attention.W_v.weight: [256, 256]

decoder_layers.0.cross_attention.W_v.bias: [256]

decoder_layers.0.cross_attention.W_o.weight: [256, 256]

decoder_layers.0.cross_attention.W_o.bias: [256]

decoder_layers.0.feed_forward.linear1.weight: [1024, 256]

decoder_layers.0.feed_forward.linear1.bias: [1024]

decoder_layers.0.feed_forward.linear2.weight: [256, 1024]

decoder_layers.0.feed_forward.linear2.bias: [256]

decoder_layers.0.norm1.weight: [256]

decoder_layers.0.norm1.bias: [256]

decoder_layers.0.norm2.weight: [256]

decoder_layers.0.norm2.bias: [256]

decoder_layers.0.norm3.weight: [256]

decoder_layers.0.norm3.bias: [256]

decoder_layers.1.self_attention.W_q.weight: [256, 256]

decoder_layers.1.self_attention.W_q.bias: [256]

decoder_layers.1.self_attention.W_k.weight: [256, 256]

decoder_layers.1.self_attention.W_k.bias: [256]

decoder_layers.1.self_attention.W_v.weight: [256, 256]

decoder_layers.1.self_attention.W_v.bias: [256]

decoder_layers.1.self_attention.W_o.weight: [256, 256]

decoder_layers.1.self_attention.W_o.bias: [256]

decoder_layers.1.cross_attention.W_q.weight: [256, 256]

decoder_layers.1.cross_attention.W_q.bias: [256]

decoder_layers.1.cross_attention.W_k.weight: [256, 256]

decoder_layers.1.cross_attention.W_k.bias: [256]

decoder_layers.1.cross_attention.W_v.weight: [256, 256]

decoder_layers.1.cross_attention.W_v.bias: [256]

decoder_layers.1.cross_attention.W_o.weight: [256, 256]

decoder_layers.1.cross_attention.W_o.bias: [256]

decoder_layers.1.feed_forward.linear1.weight: [1024, 256]

decoder_layers.1.feed_forward.linear1.bias: [1024]

decoder_layers.1.feed_forward.linear2.weight: [256, 1024]

decoder_layers.1.feed_forward.linear2.bias: [256]

decoder_layers.1.norm1.weight: [256]

decoder_layers.1.norm1.bias: [256]

decoder_layers.1.norm2.weight: [256]

decoder_layers.1.norm2.bias: [256]

decoder_layers.1.norm3.weight: [256]

decoder_layers.1.norm3.bias: [256]

decoder_layers.2.self_attention.W_q.weight: [256, 256]

decoder_layers.2.self_attention.W_q.bias: [256]

decoder_layers.2.self_attention.W_k.weight: [256, 256]

decoder_layers.2.self_attention.W_k.bias: [256]

decoder_layers.2.self_attention.W_v.weight: [256, 256]

decoder_layers.2.self_attention.W_v.bias: [256]

decoder_layers.2.self_attention.W_o.weight: [256, 256]

decoder_layers.2.self_attention.W_o.bias: [256]

decoder_layers.2.cross_attention.W_q.weight: [256, 256]

decoder_layers.2.cross_attention.W_q.bias: [256]

decoder_layers.2.cross_attention.W_k.weight: [256, 256]

decoder_layers.2.cross_attention.W_k.bias: [256]

decoder_layers.2.cross_attention.W_v.weight: [256, 256]

decoder_layers.2.cross_attention.W_v.bias: [256]

decoder_layers.2.cross_attention.W_o.weight: [256, 256]

decoder_layers.2.cross_attention.W_o.bias: [256]

decoder_layers.2.feed_forward.linear1.weight: [1024, 256]

decoder_layers.2.feed_forward.linear1.bias: [1024]

decoder_layers.2.feed_forward.linear2.weight: [256, 1024]

decoder_layers.2.feed_forward.linear2.bias: [256]

decoder_layers.2.norm1.weight: [256]

decoder_layers.2.norm1.bias: [256]

decoder_layers.2.norm2.weight: [256]

decoder_layers.2.norm2.bias: [256]

decoder_layers.2.norm3.weight: [256]

decoder_layers.2.norm3.bias: [256]

output_layer.weight: [16000, 256]

output_layer.bias: [16000]

```

### saved_models\transformer_1m_improved\transformer_1m_improved_best.pt

Top keys: epoch, model_state_dict, optimizer_state_dict, scheduler_state_dict, best_val_loss, config

```json

{
  "epoch": 19,
  "best_val_loss": 4.574004120296902,
  "config": {
    "d_model": 384,
    "num_heads": 8,
    "num_encoder_layers": 4,
    "num_decoder_layers": 4,
    "d_ff": 1536,
    "dropout": 0.1,
    "max_seq_length": 64,
    "src_vocab_size": 16000,
    "tgt_vocab_size": 16000,
    "pad_id": 0
  }
}

```

State prefixes: decoder_layers, encoder_layers, output_layer, positional_encoding, src_embedding, tgt_embedding

```

src_embedding.weight: [16000, 384]

tgt_embedding.weight: [16000, 384]

positional_encoding.pe: [1, 64, 384]

encoder_layers.0.self_attention.W_q.weight: [384, 384]

encoder_layers.0.self_attention.W_q.bias: [384]

encoder_layers.0.self_attention.W_k.weight: [384, 384]

encoder_layers.0.self_attention.W_k.bias: [384]

encoder_layers.0.self_attention.W_v.weight: [384, 384]

encoder_layers.0.self_attention.W_v.bias: [384]

encoder_layers.0.self_attention.W_o.weight: [384, 384]

encoder_layers.0.self_attention.W_o.bias: [384]

encoder_layers.0.feed_forward.linear1.weight: [1536, 384]

encoder_layers.0.feed_forward.linear1.bias: [1536]

encoder_layers.0.feed_forward.linear2.weight: [384, 1536]

encoder_layers.0.feed_forward.linear2.bias: [384]

encoder_layers.0.norm1.weight: [384]

encoder_layers.0.norm1.bias: [384]

encoder_layers.0.norm2.weight: [384]

encoder_layers.0.norm2.bias: [384]

encoder_layers.1.self_attention.W_q.weight: [384, 384]

encoder_layers.1.self_attention.W_q.bias: [384]

encoder_layers.1.self_attention.W_k.weight: [384, 384]

encoder_layers.1.self_attention.W_k.bias: [384]

encoder_layers.1.self_attention.W_v.weight: [384, 384]

encoder_layers.1.self_attention.W_v.bias: [384]

encoder_layers.1.self_attention.W_o.weight: [384, 384]

encoder_layers.1.self_attention.W_o.bias: [384]

encoder_layers.1.feed_forward.linear1.weight: [1536, 384]

encoder_layers.1.feed_forward.linear1.bias: [1536]

encoder_layers.1.feed_forward.linear2.weight: [384, 1536]

encoder_layers.1.feed_forward.linear2.bias: [384]

encoder_layers.1.norm1.weight: [384]

encoder_layers.1.norm1.bias: [384]

encoder_layers.1.norm2.weight: [384]

encoder_layers.1.norm2.bias: [384]

encoder_layers.2.self_attention.W_q.weight: [384, 384]

encoder_layers.2.self_attention.W_q.bias: [384]

encoder_layers.2.self_attention.W_k.weight: [384, 384]

encoder_layers.2.self_attention.W_k.bias: [384]

encoder_layers.2.self_attention.W_v.weight: [384, 384]

encoder_layers.2.self_attention.W_v.bias: [384]

encoder_layers.2.self_attention.W_o.weight: [384, 384]

encoder_layers.2.self_attention.W_o.bias: [384]

encoder_layers.2.feed_forward.linear1.weight: [1536, 384]

encoder_layers.2.feed_forward.linear1.bias: [1536]

encoder_layers.2.feed_forward.linear2.weight: [384, 1536]

encoder_layers.2.feed_forward.linear2.bias: [384]

encoder_layers.2.norm1.weight: [384]

encoder_layers.2.norm1.bias: [384]

encoder_layers.2.norm2.weight: [384]

encoder_layers.2.norm2.bias: [384]

encoder_layers.3.self_attention.W_q.weight: [384, 384]

encoder_layers.3.self_attention.W_q.bias: [384]

encoder_layers.3.self_attention.W_k.weight: [384, 384]

encoder_layers.3.self_attention.W_k.bias: [384]

encoder_layers.3.self_attention.W_v.weight: [384, 384]

encoder_layers.3.self_attention.W_v.bias: [384]

encoder_layers.3.self_attention.W_o.weight: [384, 384]

encoder_layers.3.self_attention.W_o.bias: [384]

encoder_layers.3.feed_forward.linear1.weight: [1536, 384]

encoder_layers.3.feed_forward.linear1.bias: [1536]

encoder_layers.3.feed_forward.linear2.weight: [384, 1536]

encoder_layers.3.feed_forward.linear2.bias: [384]

encoder_layers.3.norm1.weight: [384]

encoder_layers.3.norm1.bias: [384]

encoder_layers.3.norm2.weight: [384]

encoder_layers.3.norm2.bias: [384]

decoder_layers.0.self_attention.W_q.weight: [384, 384]

decoder_layers.0.self_attention.W_q.bias: [384]

decoder_layers.0.self_attention.W_k.weight: [384, 384]

decoder_layers.0.self_attention.W_k.bias: [384]

decoder_layers.0.self_attention.W_v.weight: [384, 384]

decoder_layers.0.self_attention.W_v.bias: [384]

decoder_layers.0.self_attention.W_o.weight: [384, 384]

decoder_layers.0.self_attention.W_o.bias: [384]

decoder_layers.0.cross_attention.W_q.weight: [384, 384]

decoder_layers.0.cross_attention.W_q.bias: [384]

decoder_layers.0.cross_attention.W_k.weight: [384, 384]

decoder_layers.0.cross_attention.W_k.bias: [384]

decoder_layers.0.cross_attention.W_v.weight: [384, 384]

decoder_layers.0.cross_attention.W_v.bias: [384]

decoder_layers.0.cross_attention.W_o.weight: [384, 384]

decoder_layers.0.cross_attention.W_o.bias: [384]

decoder_layers.0.feed_forward.linear1.weight: [1536, 384]

decoder_layers.0.feed_forward.linear1.bias: [1536]

decoder_layers.0.feed_forward.linear2.weight: [384, 1536]

decoder_layers.0.feed_forward.linear2.bias: [384]

decoder_layers.0.norm1.weight: [384]

decoder_layers.0.norm1.bias: [384]

decoder_layers.0.norm2.weight: [384]

decoder_layers.0.norm2.bias: [384]

decoder_layers.0.norm3.weight: [384]

decoder_layers.0.norm3.bias: [384]

decoder_layers.1.self_attention.W_q.weight: [384, 384]

decoder_layers.1.self_attention.W_q.bias: [384]

decoder_layers.1.self_attention.W_k.weight: [384, 384]

decoder_layers.1.self_attention.W_k.bias: [384]

decoder_layers.1.self_attention.W_v.weight: [384, 384]

decoder_layers.1.self_attention.W_v.bias: [384]

decoder_layers.1.self_attention.W_o.weight: [384, 384]

decoder_layers.1.self_attention.W_o.bias: [384]

decoder_layers.1.cross_attention.W_q.weight: [384, 384]

decoder_layers.1.cross_attention.W_q.bias: [384]

decoder_layers.1.cross_attention.W_k.weight: [384, 384]

decoder_layers.1.cross_attention.W_k.bias: [384]

decoder_layers.1.cross_attention.W_v.weight: [384, 384]

decoder_layers.1.cross_attention.W_v.bias: [384]

decoder_layers.1.cross_attention.W_o.weight: [384, 384]

decoder_layers.1.cross_attention.W_o.bias: [384]

decoder_layers.1.feed_forward.linear1.weight: [1536, 384]

decoder_layers.1.feed_forward.linear1.bias: [1536]

decoder_layers.1.feed_forward.linear2.weight: [384, 1536]

decoder_layers.1.feed_forward.linear2.bias: [384]

decoder_layers.1.norm1.weight: [384]

decoder_layers.1.norm1.bias: [384]

decoder_layers.1.norm2.weight: [384]

decoder_layers.1.norm2.bias: [384]

decoder_layers.1.norm3.weight: [384]

decoder_layers.1.norm3.bias: [384]

decoder_layers.2.self_attention.W_q.weight: [384, 384]

decoder_layers.2.self_attention.W_q.bias: [384]

decoder_layers.2.self_attention.W_k.weight: [384, 384]

decoder_layers.2.self_attention.W_k.bias: [384]

decoder_layers.2.self_attention.W_v.weight: [384, 384]

decoder_layers.2.self_attention.W_v.bias: [384]

decoder_layers.2.self_attention.W_o.weight: [384, 384]

decoder_layers.2.self_attention.W_o.bias: [384]

decoder_layers.2.cross_attention.W_q.weight: [384, 384]

decoder_layers.2.cross_attention.W_q.bias: [384]

decoder_layers.2.cross_attention.W_k.weight: [384, 384]

decoder_layers.2.cross_attention.W_k.bias: [384]

decoder_layers.2.cross_attention.W_v.weight: [384, 384]

decoder_layers.2.cross_attention.W_v.bias: [384]

decoder_layers.2.cross_attention.W_o.weight: [384, 384]

decoder_layers.2.cross_attention.W_o.bias: [384]

decoder_layers.2.feed_forward.linear1.weight: [1536, 384]

decoder_layers.2.feed_forward.linear1.bias: [1536]

decoder_layers.2.feed_forward.linear2.weight: [384, 1536]

decoder_layers.2.feed_forward.linear2.bias: [384]

decoder_layers.2.norm1.weight: [384]

decoder_layers.2.norm1.bias: [384]

decoder_layers.2.norm2.weight: [384]

decoder_layers.2.norm2.bias: [384]

decoder_layers.2.norm3.weight: [384]

decoder_layers.2.norm3.bias: [384]

decoder_layers.3.self_attention.W_q.weight: [384, 384]

decoder_layers.3.self_attention.W_q.bias: [384]

decoder_layers.3.self_attention.W_k.weight: [384, 384]

decoder_layers.3.self_attention.W_k.bias: [384]

decoder_layers.3.self_attention.W_v.weight: [384, 384]

decoder_layers.3.self_attention.W_v.bias: [384]

decoder_layers.3.self_attention.W_o.weight: [384, 384]

decoder_layers.3.self_attention.W_o.bias: [384]

decoder_layers.3.cross_attention.W_q.weight: [384, 384]

decoder_layers.3.cross_attention.W_q.bias: [384]

decoder_layers.3.cross_attention.W_k.weight: [384, 384]

decoder_layers.3.cross_attention.W_k.bias: [384]

decoder_layers.3.cross_attention.W_v.weight: [384, 384]

decoder_layers.3.cross_attention.W_v.bias: [384]

decoder_layers.3.cross_attention.W_o.weight: [384, 384]

decoder_layers.3.cross_attention.W_o.bias: [384]

decoder_layers.3.feed_forward.linear1.weight: [1536, 384]

decoder_layers.3.feed_forward.linear1.bias: [1536]

decoder_layers.3.feed_forward.linear2.weight: [384, 1536]

decoder_layers.3.feed_forward.linear2.bias: [384]

decoder_layers.3.norm1.weight: [384]

decoder_layers.3.norm1.bias: [384]

decoder_layers.3.norm2.weight: [384]

decoder_layers.3.norm2.bias: [384]

decoder_layers.3.norm3.weight: [384]

decoder_layers.3.norm3.bias: [384]

output_layer.weight: [16000, 384]

output_layer.bias: [16000]

```

### saved_models\transformer_1m_improved\transformer_1m_resume.pt

Top keys: epoch, model_state_dict, optimizer_state_dict, scheduler_state_dict, scaler_state_dict, best_val_loss, training_history, config

```json

{
  "epoch": 20,
  "best_val_loss": 4.574004120296902,
  "training_history": [
    {
      "epoch": 1,
      "train_loss": 5.424491438323974,
      "val_loss": 5.335837682088216,
      "learning_rate": 0.00040401695763926724
    },
    {
      "epoch": 2,
      "train_loss": 4.688316267700196,
      "val_loss": 5.072683652242024,
      "learning_rate": 0.00028716744302911015
    },
    {
      "epoch": 3,
      "train_loss": 4.466410226242066,
      "val_loss": 4.945814291636149,
      "learning_rate": 0.00023487943144345202
    },
    {
      "epoch": 4,
      "train_loss": 4.3397172612457275,
      "val_loss": 4.86248074637519,
      "learning_rate": 0.00020358900230161078
    },
    {
      "epoch": 5,
      "train_loss": 4.254877309875488,
      "val_loss": 4.813407791985406,
      "learning_rate": 0.00018219096756209196
    },
    {
      "epoch": 6,
      "train_loss": 4.19073752558899,
      "val_loss": 4.777731736501058,
      "learning_rate": 0.00016637498969190507
    },
    {
      "epoch": 7,
      "train_loss": 4.1402623567657475,
      "val_loss": 4.7484020127190485,
      "learning_rate": 0.0001540717999697857
    },
    {
      "epoch": 8,
      "train_loss": 4.097703725463867,
      "val_loss": 4.705128351847331,
      "learning_rate": 0.000144147993196148
    },
    {
      "epoch": 9,
      "train_loss": 4.062792940582275,
      "val_loss": 4.698535389370388,
      "learning_rate": 0.00013592385567123854
    },
    {
      "epoch": 10,
      "train_loss": 4.034036792297363,
      "val_loss": 4.67630508210924,
      "learning_rate": 0.0001289637432404737
    },
    {
      "epoch": 11,
      "train_loss": 4.008436656021118,
      "val_loss": 4.664959695604113,
      "learning_rate": 0.00012297385009494676
    },
    {
      "epoch": 12,
      "train_loss": 3.9864407405853273,
      "val_loss": 4.651791148715549,
      "learning_rate": 0.00011774787134166651
    },
    {
      "epoch": 13,
      "train_loss": 3.9661173480529786,
      "val_loss": 4.628948052724202,
      "learning_rate": 0.000113136117634923
    },
    {
      "epoch": 14,
      "train_loss": 3.9479024957733153,
      "val_loss": 4.623740196228027,
      "learning_rate": 0.00010902698762773835
    },
    {
      "epoch": 15,
      "train_loss": null,
      "val_loss": 4.607582463158502,
      "learning_rate": 0.00010530865639567127
    },
    {
      "epoch": 16,
      "train_loss": 3.915820560913086,
      "val_loss": 4.607229073842366,
      "learning_rate": 0.0001019707477212475
    },
    {
      "epoch": 17,
      "train_loss": 3.9020390037994384,
      "val_loss": 4.599395275115967,
      "learning_rate": 9.893136121034427e-05
    },
    {
      "epoch": 18,
      "train_loss": 3.8884393787231444,
      "val_loss": 4.586509545644124,
      "learning_rate": 9.614849839792197e-05
    },
    {
      "epoch": 19,
      "train_loss": 3.8760783842315676,
      "val_loss": 4.574004120296902,
      "learning_rate": 9.35879931746837e-05
    },
    {
      "epoch": 20,
      "train_loss": 3.8650640102386475,
      "val_loss": 4.574785232543945,
      "learning_rate": 9.122172860064785e-05
    }
  ],
  "config": {
    "d_model": 384,
    "num_heads": 8,
    "num_encoder_layers": 4,
    "num_decoder_layers": 4,
    "d_ff": 1536,
    "dropout": 0.1,
    "max_seq_length": 64,
    "src_vocab_size": 16000,
    "tgt_vocab_size": 16000,
    "pad_id": 0
  }
}

```

State prefixes: decoder_layers, encoder_layers, output_layer, positional_encoding, src_embedding, tgt_embedding

```

src_embedding.weight: [16000, 384]

tgt_embedding.weight: [16000, 384]

positional_encoding.pe: [1, 64, 384]

encoder_layers.0.self_attention.W_q.weight: [384, 384]

encoder_layers.0.self_attention.W_q.bias: [384]

encoder_layers.0.self_attention.W_k.weight: [384, 384]

encoder_layers.0.self_attention.W_k.bias: [384]

encoder_layers.0.self_attention.W_v.weight: [384, 384]

encoder_layers.0.self_attention.W_v.bias: [384]

encoder_layers.0.self_attention.W_o.weight: [384, 384]

encoder_layers.0.self_attention.W_o.bias: [384]

encoder_layers.0.feed_forward.linear1.weight: [1536, 384]

encoder_layers.0.feed_forward.linear1.bias: [1536]

encoder_layers.0.feed_forward.linear2.weight: [384, 1536]

encoder_layers.0.feed_forward.linear2.bias: [384]

encoder_layers.0.norm1.weight: [384]

encoder_layers.0.norm1.bias: [384]

encoder_layers.0.norm2.weight: [384]

encoder_layers.0.norm2.bias: [384]

encoder_layers.1.self_attention.W_q.weight: [384, 384]

encoder_layers.1.self_attention.W_q.bias: [384]

encoder_layers.1.self_attention.W_k.weight: [384, 384]

encoder_layers.1.self_attention.W_k.bias: [384]

encoder_layers.1.self_attention.W_v.weight: [384, 384]

encoder_layers.1.self_attention.W_v.bias: [384]

encoder_layers.1.self_attention.W_o.weight: [384, 384]

encoder_layers.1.self_attention.W_o.bias: [384]

encoder_layers.1.feed_forward.linear1.weight: [1536, 384]

encoder_layers.1.feed_forward.linear1.bias: [1536]

encoder_layers.1.feed_forward.linear2.weight: [384, 1536]

encoder_layers.1.feed_forward.linear2.bias: [384]

encoder_layers.1.norm1.weight: [384]

encoder_layers.1.norm1.bias: [384]

encoder_layers.1.norm2.weight: [384]

encoder_layers.1.norm2.bias: [384]

encoder_layers.2.self_attention.W_q.weight: [384, 384]

encoder_layers.2.self_attention.W_q.bias: [384]

encoder_layers.2.self_attention.W_k.weight: [384, 384]

encoder_layers.2.self_attention.W_k.bias: [384]

encoder_layers.2.self_attention.W_v.weight: [384, 384]

encoder_layers.2.self_attention.W_v.bias: [384]

encoder_layers.2.self_attention.W_o.weight: [384, 384]

encoder_layers.2.self_attention.W_o.bias: [384]

encoder_layers.2.feed_forward.linear1.weight: [1536, 384]

encoder_layers.2.feed_forward.linear1.bias: [1536]

encoder_layers.2.feed_forward.linear2.weight: [384, 1536]

encoder_layers.2.feed_forward.linear2.bias: [384]

encoder_layers.2.norm1.weight: [384]

encoder_layers.2.norm1.bias: [384]

encoder_layers.2.norm2.weight: [384]

encoder_layers.2.norm2.bias: [384]

encoder_layers.3.self_attention.W_q.weight: [384, 384]

encoder_layers.3.self_attention.W_q.bias: [384]

encoder_layers.3.self_attention.W_k.weight: [384, 384]

encoder_layers.3.self_attention.W_k.bias: [384]

encoder_layers.3.self_attention.W_v.weight: [384, 384]

encoder_layers.3.self_attention.W_v.bias: [384]

encoder_layers.3.self_attention.W_o.weight: [384, 384]

encoder_layers.3.self_attention.W_o.bias: [384]

encoder_layers.3.feed_forward.linear1.weight: [1536, 384]

encoder_layers.3.feed_forward.linear1.bias: [1536]

encoder_layers.3.feed_forward.linear2.weight: [384, 1536]

encoder_layers.3.feed_forward.linear2.bias: [384]

encoder_layers.3.norm1.weight: [384]

encoder_layers.3.norm1.bias: [384]

encoder_layers.3.norm2.weight: [384]

encoder_layers.3.norm2.bias: [384]

decoder_layers.0.self_attention.W_q.weight: [384, 384]

decoder_layers.0.self_attention.W_q.bias: [384]

decoder_layers.0.self_attention.W_k.weight: [384, 384]

decoder_layers.0.self_attention.W_k.bias: [384]

decoder_layers.0.self_attention.W_v.weight: [384, 384]

decoder_layers.0.self_attention.W_v.bias: [384]

decoder_layers.0.self_attention.W_o.weight: [384, 384]

decoder_layers.0.self_attention.W_o.bias: [384]

decoder_layers.0.cross_attention.W_q.weight: [384, 384]

decoder_layers.0.cross_attention.W_q.bias: [384]

decoder_layers.0.cross_attention.W_k.weight: [384, 384]

decoder_layers.0.cross_attention.W_k.bias: [384]

decoder_layers.0.cross_attention.W_v.weight: [384, 384]

decoder_layers.0.cross_attention.W_v.bias: [384]

decoder_layers.0.cross_attention.W_o.weight: [384, 384]

decoder_layers.0.cross_attention.W_o.bias: [384]

decoder_layers.0.feed_forward.linear1.weight: [1536, 384]

decoder_layers.0.feed_forward.linear1.bias: [1536]

decoder_layers.0.feed_forward.linear2.weight: [384, 1536]

decoder_layers.0.feed_forward.linear2.bias: [384]

decoder_layers.0.norm1.weight: [384]

decoder_layers.0.norm1.bias: [384]

decoder_layers.0.norm2.weight: [384]

decoder_layers.0.norm2.bias: [384]

decoder_layers.0.norm3.weight: [384]

decoder_layers.0.norm3.bias: [384]

decoder_layers.1.self_attention.W_q.weight: [384, 384]

decoder_layers.1.self_attention.W_q.bias: [384]

decoder_layers.1.self_attention.W_k.weight: [384, 384]

decoder_layers.1.self_attention.W_k.bias: [384]

decoder_layers.1.self_attention.W_v.weight: [384, 384]

decoder_layers.1.self_attention.W_v.bias: [384]

decoder_layers.1.self_attention.W_o.weight: [384, 384]

decoder_layers.1.self_attention.W_o.bias: [384]

decoder_layers.1.cross_attention.W_q.weight: [384, 384]

decoder_layers.1.cross_attention.W_q.bias: [384]

decoder_layers.1.cross_attention.W_k.weight: [384, 384]

decoder_layers.1.cross_attention.W_k.bias: [384]

decoder_layers.1.cross_attention.W_v.weight: [384, 384]

decoder_layers.1.cross_attention.W_v.bias: [384]

decoder_layers.1.cross_attention.W_o.weight: [384, 384]

decoder_layers.1.cross_attention.W_o.bias: [384]

decoder_layers.1.feed_forward.linear1.weight: [1536, 384]

decoder_layers.1.feed_forward.linear1.bias: [1536]

decoder_layers.1.feed_forward.linear2.weight: [384, 1536]

decoder_layers.1.feed_forward.linear2.bias: [384]

decoder_layers.1.norm1.weight: [384]

decoder_layers.1.norm1.bias: [384]

decoder_layers.1.norm2.weight: [384]

decoder_layers.1.norm2.bias: [384]

decoder_layers.1.norm3.weight: [384]

decoder_layers.1.norm3.bias: [384]

decoder_layers.2.self_attention.W_q.weight: [384, 384]

decoder_layers.2.self_attention.W_q.bias: [384]

decoder_layers.2.self_attention.W_k.weight: [384, 384]

decoder_layers.2.self_attention.W_k.bias: [384]

decoder_layers.2.self_attention.W_v.weight: [384, 384]

decoder_layers.2.self_attention.W_v.bias: [384]

decoder_layers.2.self_attention.W_o.weight: [384, 384]

decoder_layers.2.self_attention.W_o.bias: [384]

decoder_layers.2.cross_attention.W_q.weight: [384, 384]

decoder_layers.2.cross_attention.W_q.bias: [384]

decoder_layers.2.cross_attention.W_k.weight: [384, 384]

decoder_layers.2.cross_attention.W_k.bias: [384]

decoder_layers.2.cross_attention.W_v.weight: [384, 384]

decoder_layers.2.cross_attention.W_v.bias: [384]

decoder_layers.2.cross_attention.W_o.weight: [384, 384]

decoder_layers.2.cross_attention.W_o.bias: [384]

decoder_layers.2.feed_forward.linear1.weight: [1536, 384]

decoder_layers.2.feed_forward.linear1.bias: [1536]

decoder_layers.2.feed_forward.linear2.weight: [384, 1536]

decoder_layers.2.feed_forward.linear2.bias: [384]

decoder_layers.2.norm1.weight: [384]

decoder_layers.2.norm1.bias: [384]

decoder_layers.2.norm2.weight: [384]

decoder_layers.2.norm2.bias: [384]

decoder_layers.2.norm3.weight: [384]

decoder_layers.2.norm3.bias: [384]

decoder_layers.3.self_attention.W_q.weight: [384, 384]

decoder_layers.3.self_attention.W_q.bias: [384]

decoder_layers.3.self_attention.W_k.weight: [384, 384]

decoder_layers.3.self_attention.W_k.bias: [384]

decoder_layers.3.self_attention.W_v.weight: [384, 384]

decoder_layers.3.self_attention.W_v.bias: [384]

decoder_layers.3.self_attention.W_o.weight: [384, 384]

decoder_layers.3.self_attention.W_o.bias: [384]

decoder_layers.3.cross_attention.W_q.weight: [384, 384]

decoder_layers.3.cross_attention.W_q.bias: [384]

decoder_layers.3.cross_attention.W_k.weight: [384, 384]

decoder_layers.3.cross_attention.W_k.bias: [384]

decoder_layers.3.cross_attention.W_v.weight: [384, 384]

decoder_layers.3.cross_attention.W_v.bias: [384]

decoder_layers.3.cross_attention.W_o.weight: [384, 384]

decoder_layers.3.cross_attention.W_o.bias: [384]

decoder_layers.3.feed_forward.linear1.weight: [1536, 384]

decoder_layers.3.feed_forward.linear1.bias: [1536]

decoder_layers.3.feed_forward.linear2.weight: [384, 1536]

decoder_layers.3.feed_forward.linear2.bias: [384]

decoder_layers.3.norm1.weight: [384]

decoder_layers.3.norm1.bias: [384]

decoder_layers.3.norm2.weight: [384]

decoder_layers.3.norm2.bias: [384]

decoder_layers.3.norm3.weight: [384]

decoder_layers.3.norm3.bias: [384]

output_layer.weight: [16000, 384]

output_layer.bias: [16000]

```

## Tokenizers

- english_bpe.model: vocab=16000, PAD=0, UNK=1, BOS=2, EOS=3

- english_bpe_1m.model: vocab=16000, PAD=0, UNK=1, BOS=2, EOS=3

- hindi_bpe.model: vocab=16000, PAD=0, UNK=1, BOS=2, EOS=3

- hindi_bpe_1m.model: vocab=16000, PAD=0, UNK=1, BOS=2, EOS=3

## Architecture resolution

GRU: notebook 04 classes. Attention: notebook 05 final DecoderGRUAttention (cell 26, zero-based), including pre_output [512,1280]; earlier direct projection is superseded and does not match checkpoint. Transformer: notebook 07_Transformer_1M_Improved classes, cells 8-13; checkpoint config maps max_seq_length to constructor max_length. Epoch 19 confirmed.

Attention maximum length comes from tokenizer_config.json (64), as notebook cell 4 does. GRU length is also in checkpoint metadata.

All original files have size, mtime and SHA-256 recorded in SOURCE_MANIFEST.json. No original files modified.