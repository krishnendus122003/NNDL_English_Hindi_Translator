"""Exact class ASTs from training notebooks; no training cells execute.
See docs/MODEL_PROVENANCE.md for source cells.
"""

import math

import torch

from torch import nn

class EncoderGRU(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1, pad_id=0):
        super().__init__()
        self.pad_id = pad_id
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=pad_id)
        self.gru = nn.GRU(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)

    def forward(self, source):
        lengths = (source != self.pad_id).sum(dim=1)
        lengths = lengths.cpu()
        embedded = self.embedding(source)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        _, hidden = self.gru(packed_embedded)
        return hidden

class DecoderGRU(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1, pad_id=0):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=pad_id)
        self.gru = nn.GRU(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_token, hidden):
        input_token = input_token.unsqueeze(1)
        embedded = self.embedding(input_token)
        output, hidden = self.gru(embedded, hidden)
        output = output.squeeze(1)
        prediction = self.output_layer(output)
        return (prediction, hidden)

class Seq2SeqGRU(nn.Module):

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward_encoder(self, source):
        """
        Encode the English input sentence
        and return the final hidden state.
        """
        hidden = self.encoder(source)
        return hidden

class BahdanauAttention(nn.Module):

    def __init__(self, hidden_dim, attention_dim):
        super().__init__()
        self.W_encoder = nn.Linear(hidden_dim, attention_dim)
        self.W_decoder = nn.Linear(hidden_dim, attention_dim)
        self.V = nn.Linear(attention_dim, 1)

    def forward(self, decoder_hidden, encoder_outputs, source_mask):
        """
        decoder_hidden:
            [batch_size, hidden_dim]

        encoder_outputs:
            [batch_size, source_length, hidden_dim]

        source_mask:
            [batch_size, source_length]
        """
        decoder_hidden = decoder_hidden.unsqueeze(1)
        energy = torch.tanh(self.W_encoder(encoder_outputs) + self.W_decoder(decoder_hidden))
        scores = self.V(energy).squeeze(2)
        scores = scores.masked_fill(source_mask == 0, float('-inf'))
        attention_weights = torch.softmax(scores, dim=1)
        context_vector = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return (context_vector, attention_weights)

class EncoderGRUAttention(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1, pad_id=0):
        super().__init__()
        self.pad_id = pad_id
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=pad_id)
        self.gru = nn.GRU(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)

    def forward(self, source):
        lengths = (source != self.pad_id).sum(dim=1)
        lengths_cpu = lengths.cpu()
        embedded = self.embedding(source)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, lengths_cpu, batch_first=True, enforce_sorted=False)
        packed_outputs, hidden = self.gru(packed_embedded)
        encoder_outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True, total_length=source.size(1))
        source_mask = source != self.pad_id
        return (encoder_outputs, hidden, source_mask)

class DecoderGRUAttention(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, attention_dim, num_layers=1, pad_id=0):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=pad_id)
        self.attention = BahdanauAttention(hidden_dim=hidden_dim, attention_dim=attention_dim)
        self.gru = nn.GRU(input_size=embedding_dim + hidden_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.pre_output = nn.Linear(hidden_dim + hidden_dim + embedding_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_token, hidden, encoder_outputs, source_mask):
        embedded = self.embedding(input_token)
        decoder_hidden = hidden[-1]
        context_vector, attention_weights = self.attention(decoder_hidden, encoder_outputs, source_mask)
        gru_input = torch.cat([embedded, context_vector], dim=1).unsqueeze(1)
        output, hidden = self.gru(gru_input, hidden)
        output = output.squeeze(1)
        combined = torch.cat([output, context_vector, embedded], dim=1)
        combined = torch.tanh(self.pre_output(combined))
        prediction = self.output_layer(combined)
        return (prediction, hidden, attention_weights)

class Seq2SeqGRUAttention(nn.Module):

    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def encode(self, source):
        """
        Run the encoder and return:
        1. encoder outputs for every source token
        2. final encoder hidden state
        3. source mask for PAD handling
        """
        encoder_outputs, hidden, source_mask = self.encoder(source)
        return (encoder_outputs, hidden, source_mask)

class PositionalEncoding(nn.Module):
    """
    Adds positional information to token embeddings
    using sine and cosine functions.
    """

    def __init__(self, d_model, max_length=64):
        super().__init__()
        position = torch.arange(max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_length, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x shape:
        [batch_size, sequence_length, d_model]
        """
        seq_length = x.size(1)
        return x + self.pe[:, :seq_length]

class MultiHeadAttention(nn.Module):
    """
    Manual Multi-Head Attention implementation.

    We do NOT use nn.MultiheadAttention.
    This class performs:
    1. Linear projections for Q, K, V
    2. Split into multiple heads
    3. Scaled dot-product attention
    4. Masking
    5. Concatenate heads
    6. Final linear projection
    """

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, 'd_model must be divisible by num_heads'
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x):
        """
        Input shape:
        [batch_size, seq_len, d_model]

        Output shape:
        [batch_size, num_heads, seq_len, head_dim]
        """
        batch_size, seq_len, _ = x.size()
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def combine_heads(self, x):
        """
        Converts:
        [batch, heads, seq_len, head_dim]

        back into:
        [batch, seq_len, d_model]
        """
        batch_size, _, seq_len, _ = x.size()
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, self.d_model)

    def forward(self, query, key, value, mask=None, return_attention=False):
        Q = self.split_heads(self.W_q(query))
        K = self.split_heads(self.W_k(key))
        V = self.split_heads(self.W_v(value))
        original_dtype = Q.dtype
        with torch.amp.autocast(device_type=query.device.type, enabled=False):
            Q_float = Q.float()
            K_float = K.float()
            V_float = V.float()
            scores = torch.matmul(Q_float, K_float.transpose(-2, -1))
            scores = scores / math.sqrt(self.head_dim)
            if mask is not None:
                scores = scores.masked_fill(~mask.bool(), -1000000000.0)
            attention_weights = torch.softmax(scores, dim=-1)
            attention_weights = self.dropout(attention_weights)
            context = torch.matmul(attention_weights, V_float)
        context = context.to(original_dtype)
        context = self.combine_heads(context)
        output = self.W_o(context)
        if return_attention:
            return (output, attention_weights)
        return output

class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network used inside
    each Transformer encoder and decoder layer.

    Flow:
    d_model -> d_ff -> ReLU -> Dropout -> d_model
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x

class EncoderLayer(nn.Module):
    """
    One Transformer Encoder Layer.

    Structure:
    1. Multi-Head Self-Attention
    2. Residual Connection + LayerNorm
    3. Feed-Forward Network
    4. Residual Connection + LayerNorm
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)
        self.feed_forward = FeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        attention_output = self.self_attention(query=x, key=x, value=x, mask=src_mask)
        x = self.norm1(x + self.dropout1(attention_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        return x

class DecoderLayer(nn.Module):
    """
    One Transformer Decoder Layer.

    Structure:
    1. Masked Multi-Head Self-Attention
    2. Residual Connection + LayerNorm
    3. Cross-Attention with Encoder Output
    4. Residual Connection + LayerNorm
    5. Feed-Forward Network
    6. Residual Connection + LayerNorm
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)
        self.cross_attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads, dropout=dropout)
        self.feed_forward = FeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, encoder_output, tgt_mask=None, src_mask=None, return_cross_attention=False):
        self_attention_output = self.self_attention(query=x, key=x, value=x, mask=tgt_mask)
        x = self.norm1(x + self.dropout1(self_attention_output))
        if return_cross_attention:
            cross_attention_output, cross_attention_weights = self.cross_attention(query=x, key=encoder_output, value=encoder_output, mask=src_mask, return_attention=True)
        else:
            cross_attention_output = self.cross_attention(query=x, key=encoder_output, value=encoder_output, mask=src_mask)
        x = self.norm2(x + self.dropout2(cross_attention_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout3(ff_output))
        if return_cross_attention:
            return (x, cross_attention_weights)
        return x

class TransformerFromScratch(nn.Module):
    """
    Complete English-to-Hindi Transformer built manually.

    Includes:
    - Source embedding
    - Target embedding
    - Positional encoding
    - Encoder stack
    - Decoder stack
    - Padding masks
    - Causal target mask
    - Final vocabulary projection

    Does NOT use nn.Transformer.
    """

    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_encoder_layers, num_decoder_layers, d_ff, dropout, max_length, pad_id):
        super().__init__()
        self.d_model = d_model
        self.pad_id = pad_id
        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_id)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_id)
        self.positional_encoding = PositionalEncoding(d_model=d_model, max_length=max_length)
        self.embedding_dropout = nn.Dropout(dropout)
        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=dropout) for _ in range(num_encoder_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model=d_model, num_heads=num_heads, d_ff=d_ff, dropout=dropout) for _ in range(num_decoder_layers)])
        self.output_layer = nn.Linear(d_model, tgt_vocab_size)

    def create_src_mask(self, src):
        """
        src shape:
        [batch_size, src_len]

        Output shape:
        [batch_size, 1, 1, src_len]
        """
        return (src != self.pad_id).unsqueeze(1).unsqueeze(2)

    def create_tgt_mask(self, tgt):
        """
        Combines:
        1. Target padding mask
        2. Causal mask

        Prevents the decoder from seeing future tokens.
        """
        batch_size, tgt_len = tgt.shape
        padding_mask = (tgt != self.pad_id).unsqueeze(1).unsqueeze(2)
        causal_mask = torch.tril(torch.ones(tgt_len, tgt_len, device=tgt.device, dtype=torch.bool))
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(1)
        return padding_mask & causal_mask

    def encode(self, src):
        src_mask = self.create_src_mask(src)
        src_emb = self.src_embedding(src)
        src_emb = src_emb * math.sqrt(self.d_model)
        src_emb = self.positional_encoding(src_emb)
        x = self.embedding_dropout(src_emb)
        for layer in self.encoder_layers:
            x = layer(x, src_mask=src_mask)
        return (x, src_mask)

    def decode(self, tgt, encoder_output, src_mask, return_attention=False):
        tgt_mask = self.create_tgt_mask(tgt)
        tgt_emb = self.tgt_embedding(tgt)
        tgt_emb = tgt_emb * math.sqrt(self.d_model)
        tgt_emb = self.positional_encoding(tgt_emb)
        x = self.embedding_dropout(tgt_emb)
        final_cross_attention = None
        for i, layer in enumerate(self.decoder_layers):
            if return_attention and i == len(self.decoder_layers) - 1:
                x, final_cross_attention = layer(x=x, encoder_output=encoder_output, tgt_mask=tgt_mask, src_mask=src_mask, return_cross_attention=True)
            else:
                x = layer(x=x, encoder_output=encoder_output, tgt_mask=tgt_mask, src_mask=src_mask)
        logits = self.output_layer(x)
        if return_attention:
            return (logits, final_cross_attention)
        return logits

    def forward(self, src, tgt, return_attention=False):
        encoder_output, src_mask = self.encode(src)
        return self.decode(tgt=tgt, encoder_output=encoder_output, src_mask=src_mask, return_attention=return_attention)
