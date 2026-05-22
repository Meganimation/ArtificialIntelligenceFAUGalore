import numpy as np

# -----------------------------
# Encoder only
# -----------------------------

def softmax(x, axis=-1):
    # ‘’’
    # Compute softmax value of x
    # ‘’’
    x_shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def layer_norm(x, eps=1e-5):
    """
    x: [B, T, D]
    B = batch size
    T = sequence length
    D = model dimension
    returns LN(x) with per-token normalization over D
    """

    # ‘’
    # Compute mu (mean value)
    # Compute var (variance)
    # return (x-mu)/np.sqrt(var+eps)
    # ‘’’
    mu = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps)

def positional_encoding(T, D):
    """
    Sinusoidal positional encoding: [T, D]
    """
    pe = np.zeros((T, D), dtype=np.float64)
    pos = np.arange(T)[:, None]
    i = np.arange(D)[None, :]
    angle_rates = 1.0 / np.power(10000.0, (2 * (i // 2)) / D)
    angles = pos * angle_rates
    pe[:, 0::2] = np.sin(angles[:, 0::2])
    pe[:, 1::2] = np.cos(angles[:, 1::2])
    return pe

# -----------------------------
# Encoder (1 layer, 1 head)
# -----------------------------
class EncoderLayer:
    def __init__(self, vocab_size, d_model=32, d_ff=64, seed=0):
        rng = np.random.default_rng(seed)
        self.vocab_size = vocab_size
        self.D = d_model
        self.d_ff = d_ff

        # Token embedding table
        self.E = rng.normal(0, 0.02, size=(vocab_size, d_model))

        # Single-head self-attention projections
        self.Wq = rng.normal(0, 0.02, size=(d_model, d_model))
        self.Wk = rng.normal(0, 0.02, size=(d_model, d_model))
        self.Wv = rng.normal(0, 0.02, size=(d_model, d_model))
        self.Wo = rng.normal(0, 0.02, size=(d_model, d_model))

        # Feed-forward network
        self.W1 = rng.normal(0, 0.02, size=(d_model, d_ff))
        self.b1 = np.zeros((d_ff,))
        self.W2 = rng.normal(0, 0.02, size=(d_ff, d_model))
        self.b2 = np.zeros((d_model,))

    def self_attention(self, x, attn_mask=None):
        """
        x: [B, T, D]
        attn_mask: optional [T, T] where masked positions are -inf (additive)
        """
        B, T, D = x.shape
        scale = 1.0 / np.sqrt(D)

        # Q=XWq (linear transformation)
        # ‘’’
        # Compute Q, K, V
        # Q = matrix multiplication @ between x and Wq
        # Compute K and V similarly.
        # Compute scores using Q,K,V divided by scale
        # ‘’’
        Q = x @ self.Wq
        K = x @ self.Wk
        V = x @ self.Wv
        scores = (Q @ np.transpose(K, (0, 2, 1))) * scale

        # Necessary for Padding
        # if input sequences are different, we use PAD
        if attn_mask is not None:
            scores = scores + attn_mask[None, :, :]   # broadcast to [B,T,T]

        A = softmax(scores, axis=-1)        # [B,T,T]
        C = A @ V                                     # [B,T,D]
        out = C @ self.Wo                         # [B,T,D]
        return out, A

    def feed_forward(self, x):
        """
        x: [B,T,D]
        """
        h = x @ self.W1 + self.b1     # [B,T,d_ff]
        h = np.maximum(0, h)          # ReLU
        y = h @ self.W2 + self.b2     # [B,T,D]
        return y

    def forward(self, tokens, pe=None, attn_mask=None):
        """
        tokens: [B, T] integer token ids
        pe: optional [T, D] positional encoding
        returns: output [B,T,D], attention weights [B,T,T]
        """
        B, T = tokens.shape

        # 1) Token embedding (+ optional position)
        x = self.E[tokens]                         # [B,T,D]
        if pe is not None:
            x = x + pe[None, :, :]                 # [B,T,D]

        # 2) Self-attention block with pre-LN + residual
        x_ln = layer_norm(x)
        attn_out, A = self.self_attention(x_ln, attn_mask=attn_mask)
        x = x + attn_out                           # residual

        # 3) FFN block with pre-LN + residual
        # ‘’’
        # Compute x_ln = layer normalized x
        # Compute ffn_out using feed_forward(x_ln)
        # Compute residual value x = x + ffn_out (residual connection)
        # ‘’’
        x_ln = layer_norm(x)
        ffn_out = self.feed_forward(x_ln)
        x = x + ffn_out

        return x, A

vocab_size = 50
B, T, D = 2, 6, 32

model = EncoderLayer(vocab_size=vocab_size, d_model=D, d_ff=64, seed=0)
tokens = np.array([[1, 5, 2, 9, 9, 3],
                   [4, 4, 7, 1, 0, 2]], dtype=np.int64)

pe = positional_encoding(T, D)   # optional
out, attn = model.forward(tokens, pe=pe)

print("out shape:", out.shape)   # (B, T, D)
print("attn shape:", attn.shape) # (B, T, T)
print("attn[0] row sums:", attn[0].sum(axis=-1))  # should be ~1 each row
