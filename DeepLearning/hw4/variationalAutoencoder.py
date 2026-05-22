
import numpy as np

# ---------- helpers ----------
def relu(x):
    # ‘’’
    # YOUR WORK HERE
    # ‘’’

def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)

# ---------- VAE Encoder (MLP) ----------
class VAEEncoder:
    """
    Encoder: x -> h -> (mu, logvar) and sample z via reparameterization method
    Forward-pass only.
    """
    def __init__(self, input_dim, hidden_dim, latent_dim, seed=0):
        rng = np.random.default_rng(seed)

        # He initialization for ReLU layer, small init for heads
        self.W1 = rng.normal(0, np.sqrt(2.0 / input_dim), size=(input_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim)

        self.W_mu = rng.normal(0, 0.02, size=(hidden_dim, latent_dim))
        self.b_mu = np.zeros(latent_dim)

        self.W_lv = rng.normal(0, 0.02, size=(hidden_dim, latent_dim))
        self.b_lv = np.zeros(latent_dim)

    def forward(self, x, rng=None):
        """
        x: [B, D] (batch of input vectors)
        returns:
          mu:     [B, Z]
          logvar: [B, Z]
          z:      [B, Z]
        """
        if rng is None:
            rng = np.random.default_rng()

        # ‘’’
        # YOUR WORK HERE
        # Compute h
        # Compute mu and logvar
        # ‘’’

        eps = rng.normal(size=mu.shape)
        # ‘’’
        # YOUR WORK HERE
        # Reparameterization method
        # Compute z using mu, logvar, and eps
        # Return mu, logvar, and z
        # ‘’’
        return mu, logvar, z

B, D = 4, 10      # batch size, input dim
H, Z = 16, 3      # hidden dim, latent dim

rng = np.random.default_rng(123)
x = rng.normal(size=(B, D))

enc = VAEEncoder(input_dim=D, hidden_dim=H, latent_dim=Z, seed=42)
mu, logvar, z = enc.forward(x, rng=rng)

print("x shape     :", x.shape)
print("mu shape    :", mu.shape)
print("logvar shape:", logvar.shape)
print("z shape     :", z.shape)
print("\nmu:\n", mu)
print("\nlogvar:\n", logvar)
print("\nz sample:\n", z)
