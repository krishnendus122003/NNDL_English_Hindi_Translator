# Viva guide

## 30-second explanation

“Our project translates English into Hindi using three neural models that we trained: a GRU encoder-decoder, a GRU with Bahdanau attention and a Transformer implemented from scratch. We use SentencePiece subwords and compare greedy and beam decoding. The selected scratch Transformer is Epoch 19 and has locked official scores of BLEU 5.8019 and chrF 28.7418. FastAPI and a browser interface demonstrate translation, comparison and genuine attention. NLLB is a separate pretrained reference, not our trained model.”

## Two-minute architecture explanation

“We first convert the English sentence into SentencePiece IDs, add BOS and EOS, and bound the input to the model's 64-token capacity. The GRU encoder processes embeddings sequentially and passes its final hidden state to a Hindi decoder. The decoder predicts the next token until EOS. Its weakness is compressing the source into one state.

Bahdanau attention keeps all encoder states. For each target step, it compares the decoder state with each source state, masks padding and normalizes scores. The resulting context vector is a weighted sum of encoder states. Our saved decoder combines the output state, context and embedding, then uses a 1280-to-512 projection before predicting over 16,000 Hindi pieces. The displayed heatmap comes from these actual attention weights.

The Transformer replaces recurrence with attention. We implemented projections, head splitting, scaled dot products, masks and head combination manually. Sinusoidal positions supply order information. Four encoder layers build source memory; four decoder layers use causal self-attention and cross-attention to that memory. Residuals, LayerNorm and feed-forward layers complete each block. The model dimension is 384, with eight heads and FFN width 1536.

At inference we use greedy search or beam search with width four and length penalty 0.6. The API strictly loads the exact Epoch-19 checkpoint and matching 1M tokenizers. It never retrains, substitutes an architecture or hides incompatible weights. NLLB is labeled separately because it is externally pretrained.”

## Five-minute full explanation

“The project asks how increasingly expressive neural architectures handle English-to-Hindi translation. Hindi has different typical word order and morphology, so simple word replacement is insufficient. We use the IIT Bombay parallel corpus, which the saved EDA records as 1,659,083 training pairs, 520 validation pairs and 2,507 test pairs. Cleaning normalizes whitespace, removes empty and duplicate pairs and filters training sentences to 1–50 words in each language. Fixed seed-42 subsets support the original 300K experiments and improved 1M Transformer.

SentencePiece BPE breaks text into reusable subword pieces. We have English and Hindi vocabularies of 16,000 pieces. PAD, UNK, BOS and EOS are IDs zero through three. The original GRU models and improved Transformer have explicit tokenizer mappings. Inputs have a 64-token limit including BOS and EOS. The application reports when content exceeds 62 pieces rather than silently truncating it.

The baseline is an encoder-decoder GRU with 256-dimensional embeddings and a 512-dimensional hidden state. The encoder's final state initializes the decoder. At inference, each predicted token becomes the next input. During training, teacher forcing can instead supply the reference token; the baseline notebook uses ratio 0.5. A fixed final encoder state creates an information bottleneck, which is one reason long sequences can be difficult.

The Bahdanau model keeps all encoder outputs and dynamically attends to relevant positions. Additive scoring combines encoder states with the current decoder state. Source padding receives no attention. The normalized weighted sum becomes a context vector. Our actual trained decoder uses a 1280-to-512 pre-output projection, tanh and the vocabulary projection. This matters because the notebook contains an earlier decoder variant that does not match the checkpoint. The heatmap shows real weights aligned with the generated Hindi pieces; it is not a fabricated explanation or a guarantee of correctness.

The final Transformer is implemented manually. Embeddings are scaled and combined with sinusoidal positional encoding. Attention uses Q, K and V projections, eight heads and scaled dot products. Padding masks hide padding; the causal mask hides future target tokens. Decoder cross-attention accesses the encoder memory. Residual connections, LayerNorm and feed-forward layers support representation and optimization. We use four encoder layers and four decoder layers, d_model 384, FFN 1536 and dropout 0.1. The selected model has 35,012,224 parameters and is the preserved Epoch-19 checkpoint, not the fine-tuned alternative.

Greedy decoding chooses the best next token immediately. Beam search keeps several candidate prefixes and ranks cumulative log probability with the notebook's length normalization. Our default is width four and penalty 0.6. Beam search explores alternatives but costs more computation and cannot fix all modeling errors. Generation is bounded and stops on EOS.

The locked official scores progress from GRU BLEU 1.3797 and chrF 14.3026, through attention 2.4331 and 18.9274, to scratch Transformer 5.8019 and 28.7418. BLEU is token n-gram overlap with a brevity penalty; chrF uses character n-gram precision and recall. Neither is percentage accuracy. The NLLB reference scores 23.4315 and 51.0445, but it uses external pretraining and was not trained by our team. Also, training scale changed between our models, so we cannot attribute all improvement solely to architecture. We document a historical beam-validation provenance discrepancy without rerunning evaluation or changing locked scores.

Finally, FastAPI validates requests, loads models strictly and caches them in evaluation mode. The browser offers translation, comparisons, warnings and attention. Logging appends local prediction records without interrupting translations on a logging error. Tests verify actual checkpoints, masks, inference, validation and API behavior with smoke sentences only. The result is an educational, reproducible local demonstration with preserved training work, not a claim of production-quality translation.”

## Concepts to explain confidently

| Concept | Explanation tied to this project |
|---|---|
| NLP | Natural language processing: computational handling of human language. Translation is one NLP task. |
| NMT | Neural machine translation: learn a conditional distribution over target sequences given a source sequence. |
| Parallel corpus | Aligned source/target sentence pairs conveying corresponding meanings. Alignment quality matters. |
| SentencePiece | A tokenizer that learns pieces directly from text and can represent word boundaries without requiring a separate word tokenizer. |
| BPE | Byte-pair encoding starts with small units and repeatedly merges frequent adjacent units; learned subwords balance vocabulary size and sequence length. |
| PAD | Fills batch sequences to a common length. Mask it in attention and ignore it in the training loss. |
| UNK | Represents content that cannot be represented by the available tokenizer vocabulary. Subwords reduce but do not guarantee removal of unknowns. |
| BOS | Signals the beginning of the sequence and starts autoregressive decoding. |
| EOS | Signals that generation should stop. It is retained for source boundary information but removed from displayed text. |
| Embeddings | Learned dense vectors indexed by token ID; close vector directions can encode useful statistical relationships. |
| RNN | A recurrent network updates a hidden state across ordered time steps. It can struggle to preserve long-range gradients. |
| GRU | A gated recurrent unit uses reset and update gates to control candidate state computation and state retention. |
| Reset gate | Controls how much of the previous hidden information enters the candidate state. |
| Update gate | Controls interpolation between the previous state and the candidate state. In PyTorch's convention, larger z retains more old state. |
| Encoder | Converts the source sequence into a representation: final state, full recurrent states or Transformer memory. |
| Decoder | Predicts the target sequence conditioned on source information and previous target tokens. |
| Hidden state | An internal learned vector summarizing information available at a recurrent time step. |
| Teacher forcing | During training, provide a gold previous token rather than always feeding a model prediction. Reference tokens are unavailable at inference. |
| Cross entropy | Negative log likelihood of the target class; penalizes low probability on the correct next token. PAD is ignored. |
| Bahdanau attention | Additive learned scoring between decoder state and source states, followed by masked softmax. |
| Context vector | Weighted sum of source states, using the current target step's attention weights. |
| Alignment | A distribution connecting a target-generation step to source positions; soft and learned rather than a hard dictionary match. |
| Transformer | An architecture built from attention and feed-forward blocks without recurrent state updates across token positions. |
| Positional encoding | Supplies sequence order, which bare content-based self-attention does not inherently identify. This project uses sine/cosine functions. |
| Q/K/V | Queries describe what a position seeks, keys support matching and values supply the information aggregated after matching. They are learned projections. |
| Scaled dot-product attention | `softmax(QKᵀ / sqrt(d_k) + mask) V`; scaling moderates logit magnitude. Masks exclude forbidden positions before softmax. |
| Multi-head attention | Several smaller attention spaces run in parallel and are concatenated/projected. Eight heads here have 48 dimensions each. |
| Padding mask | Prevents attention to artificial PAD positions. It is not the same as the causal mask. |
| Causal mask | Allows target position t to see only permitted preceding/current positions, preventing future-token leakage. |
| Masked self-attention | Decoder attention whose queries, keys and values come from target states, constrained by causal and padding masks. |
| Cross-attention | Decoder queries attend to keys/values formed from source encoder memory. |
| FFN | A position-wise nonlinear feed-forward transformation: 384 → 1536 → 384 with ReLU and dropout here. |
| Residual connection | Adds a block's input back to its transformed output, helping preserve information and gradient flow. |
| LayerNorm | Normalizes features within each token representation, with learned scale/bias. It is not batch normalization. |
| Adam | An optimizer using moving estimates of first and second gradient moments to adapt updates. It is unnecessary for inference. |
| Warmup schedule | Gradually raises the learning rate at the start, then decays it; the improved notebook uses 4,000 warmup steps. |
| Label smoothing | Replaces a completely one-hot target distribution with a softened distribution; coefficient 0.1 in improved training. |
| Mixed precision | Combines lower precision operations with selected full-precision operations to reduce memory/compute; training can use gradient scaling. |
| Gradient clipping | Bounds gradient norm to reduce excessively large updates. It operates during training, not decoding. |
| Greedy decoding | Chooses one locally best next token; fast but cannot revisit earlier alternatives. |
| Beam search | Keeps a fixed-width set of candidate prefixes to explore more likely sequences. It is approximate search. |
| Length penalty | Adjusts ranking to reduce the tendency of cumulative log probability to favor short sequences; our notebook divides by length^0.6. |
| BLEU | Corpus token n-gram precision with a brevity penalty; sensitive to wording and tokenization. |
| chrF | Character n-gram F-score combining precision and recall; gives partial credit for related character sequences. |
| FastAPI | Python HTTP API framework used to validate requests and expose inference services; it is not a neural model. |
| NLLB | No Language Left Behind; here the externally pretrained `facebook/nllb-200-distilled-600M` reference, using eng_Latn → hin_Deva. |

## Equations and shapes

For a PyTorch-style GRU, omitting biases for readability:

```text
r_t = sigmoid(W_ir x_t + W_hr h_(t-1))
z_t = sigmoid(W_iz x_t + W_hz h_(t-1))
n_t = tanh(W_in x_t + r_t ⊙ (W_hn h_(t-1)))
h_t = (1-z_t) ⊙ n_t + z_t ⊙ h_(t-1)
```

Some textbooks swap which term is multiplied by the update gate. State the convention before describing the gate. Actual PyTorch candidate-state biases are applied within their respective projections.

For attention, encoder states are `[batch, source_length, 512]`, decoder hidden is `[batch, 512]`, weights are `[batch, source_length]` and context is `[batch, 512]`. In the manual Transformer, head tensors are `[batch, 8, length, 48]`. Scores have shape `[batch, 8, query_length, key_length]`.

## Common questions and answers

**What did your team train?** The GRU, Bahdanau GRU and scratch Transformer. NLLB is a pretrained external reference. Fill real individual contributions before presenting.

**What makes your Transformer “from scratch”?** We wrote attention projections, scaling, masks, head operations and encoder/decoder blocks using basic PyTorch layers and tensors. “From scratch” does not mean we implemented automatic differentiation or matrix multiplication ourselves.

**How do you prove the application loads the real models?** Checkpoint configuration and tensor shapes match notebook-derived classes; `load_state_dict(strict=True)` passes. The selected Transformer checkpoint records Epoch 19. A class AST test checks fidelity to notebook definitions.

**Why not strict=False?** It could conceal missing or extra learned parameters and leave layers incorrectly initialized. We surface mismatches instead of accepting an approximate architecture.

**Why do two Bahdanau decoder definitions exist?** The notebook developed an earlier direct projection and a later efficient pre-output projection. The checkpoint contains `decoder.pre_output` and its tensor dimensions match the later class, which is what the application uses.

**Why 64 tokens?** It is the trained configuration and positional-buffer capacity. The app keeps 62 content pieces plus BOS/EOS and warns when truncation is necessary. It does not extend the model's context capacity.

**Are BPE pieces words?** Not necessarily. A word can split into multiple pieces, and the visible whitespace marker is part of SentencePiece's representation. This is why heatmap axes can contain fragments.

**Why is PAD masking necessary if PAD embedding is zero?** A zero embedding can still flow through biased projections and attention normalization. Masking explicitly prevents padded source positions receiving probability.

**Why does attention help GRU translation?** It supplies direct access to source-position states at every decoder step instead of relying only on one compressed final state. Improvement is empirical, not guaranteed for every sentence.

**Can attention prove why a translation is correct?** No. It shows internal alignment weights. High attention to a token does not establish causal explanation or semantic correctness.

**How are training and inference different?** Training uses reference targets and loss gradients to update parameters. Inference uses model predictions, eval mode and disabled gradients. Dropout is disabled by eval mode; no_grad alone would not disable it.

**Why divide by sqrt(d_k)?** Dot products tend to grow in magnitude with key dimension. Scaling reduces extreme softmax logits and helps optimization.

**Can the Transformer decode all Hindi tokens in parallel?** Training can compute losses for shifted known target positions together using masks. Autoregressive inference still generates the next token from its prefix step by step.

**Does increasing beam width always improve BLEU?** No. It can favor model biases, alter length and increase latency. Width four and penalty 0.6 are preserved selections, not tuned in this app.

**Why are the scores low?** Possible factors include limited training scale, capacity, vocabulary, length bounds, domain variation and valid wording differences from the reference. Do not claim a diagnosed cause without an experiment.

**Is NLLB a fair architecture comparison?** It is a useful external reference but uses different and much broader pretraining. Also, our improved Transformer uses more training pairs than original models. Neither comparison isolates architecture alone.

**What is the exact final scratch score?** BLEU 5.8019 and chrF 28.7418 on 2,507 official test pairs, Epoch 19, beam four and penalty 0.6. The 23.4315 BLEU score belongs only to NLLB.

**Why do local GRU CSVs show different scores?** Their fields are validation_bleu and validation_chrf. The official GRU values are preserved from the user's locked record; independent official outputs were not located in this copy.

**Why do two beam-validation results differ?** The earlier notebook run prints 5.9770/30.2137. A later table inserts 6.4504/30.0112 after a checkpoint reload elsewhere. The original measurement record for the inserted row is not established locally. We document that qualification and preserve the locked value without fabricating a reconciliation.

**Why not rerun the test set to verify?** This completion task explicitly preserves the finished evaluation and forbids official reruns and test-set tuning. Application verification uses only independent smoke inputs.

**What happens if input is invalid or a model is missing?** Schema validation returns 422, unavailable model errors return 503 and comparison reports errors per model. Long valid input is truncated with a warning; unexpected failures are logged and return a controlled error.

**What if logging fails?** Translation still returns. The backend records the logging exception; prediction logging is not allowed to break inference.

**Will NLLB download at startup?** No. Imports and weights are lazy and occur only when requested. It remains unavailable locally without the optional dependency; other models still work.

**What would you improve next?** Propose controlled training comparisons, wider domain coverage, human evaluation and longer-context research. Keep proposals distinct from implemented changes and avoid using the official test set for tuning.
