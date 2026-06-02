import pickle
import matplotlib.pyplot as plt
import numpy as np

PKL_FILE      = "pinn_causal.pkl"
OUT_FILE_LOSS = "training_loss.png"
OUT_FILE_W    = "causal_weight.png"

with open(PKL_FILE, "rb") as f:
    ckpt = pickle.load(f)

loss_log = np.array(ckpt['loss_log'])
W_log    = np.array(ckpt['W_log'])
config   = ckpt['config']

n_checkpoints = len(loss_log)
checkpoint_iters = np.arange(n_checkpoints) * 500   # one checkpoint every 500 iters

eps_schedule  = config['epsilon_schedule']
n_iter_per_eps = config['n_iter_per_eps']
eps_boundaries = [i * n_iter_per_eps // 500 for i in range(1, len(eps_schedule))]

print(f"Loaded {PKL_FILE}")
print(f"  Checkpoints:  {n_checkpoints}")
print(f"  Total iters:  {checkpoint_iters[-1]:,}")
print(f"  Final loss:   {loss_log[-1]:.3e}")
print(f"  Final W_min:  {W_log[-1]:.4f}")
print(f"  ε schedule:   {eps_schedule}")

fig1, ax1 = plt.subplots(figsize=(7, 4.5))
ax1.semilogy(checkpoint_iters, loss_log, lw=1.2)
for b, eps in zip(eps_boundaries, eps_schedule[1:]):
    ax1.axvline(b * 500, color='gray', linestyle=':', alpha=0.6)
    ax1.text(b * 500, ax1.get_ylim()[1] * 0.5,
             f' ε={eps}', color='gray', fontsize=9, va='top')
ax1.set_xlabel('iteration')
ax1.set_ylabel('total loss')
ax1.set_title('Training loss')
ax1.grid(alpha=0.3)

fig1.tight_layout()
fig1.savefig(OUT_FILE_LOSS, dpi=120)
print(f"\nSaved {OUT_FILE_LOSS}")

fig2, ax2 = plt.subplots(figsize=(7, 4.5))
ax2.plot(checkpoint_iters, W_log, lw=1.2)
ax2.axhline(0.99, color='red', linestyle='--', alpha=0.6, label='target W=0.99')
for b, eps in zip(eps_boundaries, eps_schedule[1:]):
    ax2.axvline(b * 500, color='gray', linestyle=':', alpha=0.6)
    ax2.text(b * 500, 0.05, f' ε={eps}', color='gray', fontsize=9)
ax2.set_xlabel('iteration')
ax2.set_ylabel('min causal weight  W_min')
ax2.set_title('Causal weight progression')
ax2.set_ylim(-0.05, 1.05)
ax2.legend(loc='lower right')
ax2.grid(alpha=0.3)

fig2.tight_layout()
fig2.savefig(OUT_FILE_W, dpi=120)
print(f"Saved {OUT_FILE_W}")
