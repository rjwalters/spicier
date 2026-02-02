# spicier-solver

Linear and nonlinear solvers for the Spicier circuit simulator.

## Features

- **DC analysis** - Operating point (.OP) and DC sweep (.DC)
- **AC analysis** - Small-signal frequency response (.AC)
- **Transient analysis** - Time-domain simulation (.TRAN)
- **Newton-Raphson** - Nonlinear solver with convergence aids
- **Sparse solvers** - faer-based LU with symbolic caching
- **GMRES** - Iterative solver with Jacobi preconditioning

## Analysis Types

```rust
use spicier_solver::{solve_dc, solve_ac, solve_transient};
use spicier_solver::{AcParams, AcSweepType, TransientParams};

// DC operating point
let dc_result = solve_dc(&stamper)?;

// AC sweep (decade, 10 points, 1Hz to 1MHz)
let ac_params = AcParams {
    sweep_type: AcSweepType::Decade,
    num_points: 10,
    fstart: 1.0,
    fstop: 1e6,
};
let ac_result = solve_ac(&ac_stamper, &ac_params)?;

// Transient (1us step, 1ms duration)
let tran_params = TransientParams {
    tstep: 1e-6,
    tstop: 1e-3,
    ..Default::default()
};
let tran_result = solve_transient(&tran_stamper, caps, inds, &tran_params, &dc)?;
```

## License

Licensed under either of Apache License, Version 2.0 or MIT license at your option.
