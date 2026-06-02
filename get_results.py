import pickle
import numpy as np
import jax.numpy as jnp
from jax import random, vmap
import pyvista as pv
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import sys
import os

from pinn import MLP


L_X         = 2.0
L_Y         = 0.5
CYL_CX      = 0.5
CYL_CY      = 0.25
CYL_R       = 0.1

PINN_FILE        = "pinn_causal.pkl"
VTK_FILE         = "flow/VTK/flow_0_5/internal.vtu"
COMPARISON_TIME  = 0.5

N_SUBSAMPLE = 20000
SEED        = 42

OUT_PLOT = "comparison_t0p5_{name}.png"


print(f"Loading CFD snapshot: {VTK_FILE}")
mesh = pv.read(VTK_FILE)
centers = mesh.cell_centers()
xyz = np.asarray(centers.points)
x_cfd_all = xyz[:, 0]
y_cfd_all = xyz[:, 1]

U_field = np.asarray(mesh.cell_data.get_array('U'))
u_cfd_all = U_field[:, 0]
v_cfd_all = U_field[:, 1]
p_cfd_all = np.asarray(mesh.cell_data.get_array('p'))

outside_cyl = (x_cfd_all - CYL_CX)**2 + (y_cfd_all - CYL_CY)**2 > CYL_R**2
x_all = x_cfd_all[outside_cyl]
y_all = y_cfd_all[outside_cyl]
u_all = u_cfd_all[outside_cyl]
v_all = v_cfd_all[outside_cyl]
p_all = p_cfd_all[outside_cyl]

print(f"  CFD cells outside cylinder: {len(x_all)}")
print(f"  u range: [{u_all.min():.3f}, {u_all.max():.3f}]")
print(f"  v range: [{v_all.min():.3f}, {v_all.max():.3f}]")
print(f"  p range: [{p_all.min():.3f}, {p_all.max():.3f}]")


rng = np.random.default_rng(SEED)
N = min(N_SUBSAMPLE, len(x_all))
idx = rng.choice(len(x_all), size=N, replace=False)
idx = np.sort(idx)

x_pts  = x_all[idx]
y_pts  = y_all[idx]
u_cfd  = u_all[idx]
v_cfd  = v_all[idx]
p_cfd  = p_all[idx]

print(f"\nSubsampled to {N} points (seed={SEED})")


print(f"\nLoading PINN: {PINN_FILE}")
with open(PINN_FILE, 'rb') as f:
    ckpt = pickle.load(f)

cfg = ckpt['config']
params = ckpt['params']

_, apply = MLP(
    cfg['layers'],
    consts_key=random.PRNGKey(cfg['consts_key_seed']),
    rff_n=cfg['rff_n'],
    sigma=cfg['sigma'],
    M_t=cfg['M_t'],
)

def u_net(params, t, x, y):
    return apply(params, jnp.array([t, x, y]))[0]
def v_net(params, t, x, y):
    return apply(params, jnp.array([t, x, y]))[1]
def p_net(params, t, x, y):
    return apply(params, jnp.array([t, x, y]))[2]

u_fn = vmap(u_net, in_axes=(None, 0, 0, 0))
v_fn = vmap(v_net, in_axes=(None, 0, 0, 0))
p_fn = vmap(p_net, in_axes=(None, 0, 0, 0))

t_arr = jnp.full(N, COMPARISON_TIME)
x_arr = jnp.array(x_pts)
y_arr = jnp.array(y_pts)

u_pinn = np.asarray(u_fn(params, t_arr, x_arr, y_arr))
v_pinn = np.asarray(v_fn(params, t_arr, x_arr, y_arr))
p_pinn = np.asarray(p_fn(params, t_arr, x_arr, y_arr))

print(f"\nPINN prediction at t={COMPARISON_TIME}:")
print(f"  u range: [{u_pinn.min():.3f}, {u_pinn.max():.3f}]")
print(f"  v range: [{v_pinn.min():.3f}, {v_pinn.max():.3f}]")
print(f"  p range: [{p_pinn.min():.3f}, {p_pinn.max():.3f}]")


def rel_l2(pred, true):
    num = np.linalg.norm(pred - true)
    den = np.linalg.norm(true)
    return num / den if den > 1e-12 else num

err_u = rel_l2(u_pinn, u_cfd)
err_v = rel_l2(v_pinn, v_cfd)
err_p = rel_l2(p_pinn, p_cfd)

print(f"\nRelative L² errors at t={COMPARISON_TIME}:")
print(f"  u: {err_u:.4e}")
print(f"  v: {err_v:.4e}")
print(f"  p: {err_p:.4e}")


fields_cfd  = [u_cfd,  v_cfd,  p_cfd]
fields_pinn = [u_pinn, v_pinn, p_pinn]
names       = ['u', 'v', 'p']
errors      = [err_u, err_v, err_p]


def style(ax):
    ax.add_patch(Circle((CYL_CX, CYL_CY), CYL_R,
                        fill=True, facecolor='lightgray',
                        edgecolor='black', linewidth=1.2, zorder=10))
    ax.set_aspect('equal')
    ax.set_xlim(0, L_X)
    ax.set_ylim(0, L_Y)


for cfd, pinn, name, err in zip(fields_cfd, fields_pinn, names, errors):
    fig, axes = plt.subplots(1, 3, figsize=(16, 3.2))

    vmax = np.max(np.abs(cfd))

    sc = axes[0].scatter(x_pts, y_pts, c=cfd, s=4, cmap='RdBu_r',
                         vmin=-vmax, vmax=vmax)
    style(axes[0])
    axes[0].set_title(f'CFD  {name}(x,y,t={COMPARISON_TIME})')
    plt.colorbar(sc, ax=axes[0], fraction=0.025, pad=0.02)

    sc = axes[1].scatter(x_pts, y_pts, c=pinn, s=4, cmap='RdBu_r',
                         vmin=-vmax, vmax=vmax)
    style(axes[1])
    axes[1].set_title(f'PINN  {name}(x,y,t={COMPARISON_TIME})')
    plt.colorbar(sc, ax=axes[1], fraction=0.025, pad=0.02)

    abs_err = np.abs(pinn - cfd)
    sc = axes[2].scatter(x_pts, y_pts, c=abs_err, s=4, cmap='magma')
    style(axes[2])
    axes[2].set_title(f'|PINN − CFD|  {name}   (rel L² = {err:.3e})')
    plt.colorbar(sc, ax=axes[2], fraction=0.025, pad=0.02)

    plt.tight_layout()
    out = OUT_PLOT.format(name=name)
    plt.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Saved {out}")
