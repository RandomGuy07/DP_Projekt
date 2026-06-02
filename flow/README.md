# Pipe Flow — Turbulent k-ω SST + Sinusoidal Inlet
# Re = 50,000 | U_mean = 1.0 m/s | nu = 1e-5 | H = 0.5 m

## Key parameters
- Inlet: U(t) = 1.0 + 0.5*sin(2π*t)  [±50% oscillation at 1 Hz]
- Cylinder: D = 0.2 m at (0.5, 0.25)
- Turbulence: k-ω SST with wall functions (y+ target: 30–100)
- k_inlet  = 0.00375 m²/s²
- ω_inlet  = 3.04 1/s

## Workflow
1. gmshToFoam pipe_obstacle.msh
2. Edit constant/polyMesh/boundary → set frontAndBack to type empty
3. cp -r initial 0
4. pimpleFoam 2>&1 | tee log.pimpleFoam
5. paraFoam

## For PINNs
- Fields to export: U, p, k, omega at each writeInterval
- foamToVTK → converts all timesteps to VTK format for easy loading in Python
- Use numpy or pyvista to load VTK files as training data
