import numpy as np
import pyvista as pv
from pathlib import Path

L_X         = 2.0
L_Y         = 0.5
CYL_CX      = 0.5
CYL_CY      = 0.25
CYL_R       = 0.1

VTK_FILE      = "flow/VTK/flow_0_0/internal.vtu"
SNAPSHOT_TIME = 0.0
OUT_FILE      = "ic_t0.npy"

N_SUBSAMPLE = 5000      
SEED        = 42        

print(f"Loading {VTK_FILE} ...")
mesh = pv.read(VTK_FILE)
print(f"  Type:  {type(mesh).__name__}")
print(f"  Cells: {mesh.n_cells}")
print(f"  Points: {mesh.n_points}")
print(f"  Cell data arrays:  {list(mesh.cell_data.keys())}")
print(f"  Point data arrays: {list(mesh.point_data.keys())}")

centers = mesh.cell_centers()
xyz = np.asarray(centers.points)
x = xyz[:, 0]
y = xyz[:, 1]
z = xyz[:, 2]

U_field = np.asarray(mesh.cell_data.get_array('U'))    # shape (N_cells, 3)
u = U_field[:, 0]
v = U_field[:, 1]
p = np.asarray(mesh.cell_data.get_array('p'))          # shape (N_cells,)

outside_cyl = (x - CYL_CX)**2 + (y - CYL_CY)**2 > CYL_R**2
x = x[outside_cyl]
y = y[outside_cyl]
u = u[outside_cyl]
v = v[outside_cyl]
p = p[outside_cyl]

if N_SUBSAMPLE is not None and N_SUBSAMPLE < len(x):
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(x), size=N_SUBSAMPLE, replace=False)
    idx = np.sort(idx)            # sorting is optional, helps cache locality
    x = x[idx]
    y = y[idx]
    u = u[idx]
    v = v[idx]
    p = p[idx]

ic = {
    't0': SNAPSHOT_TIME,
    'x':  x.astype(np.float32),
    'y':  y.astype(np.float32),
    'u':  u.astype(np.float32),
    'v':  v.astype(np.float32),
    'p':  p.astype(np.float32),
}
np.save(OUT_FILE, ic, allow_pickle=True)
