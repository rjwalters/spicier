# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-02-01

Initial release of Spicier, a high-performance SPICE circuit simulator written in Rust.

### Added

#### Core Functionality
- **Circuit representation** - MNA (Modified Nodal Analysis) matrix structures, node management, and circuit graph
- **Netlist parsing** - Full SPICE netlist parser supporting elements R, C, L, V, I, D, M, E, G, F, H, B, X
- **DC analysis** - Operating point (.OP) and DC sweep (.DC) with nested sweep support
- **AC analysis** - Small-signal frequency response (.AC) with linear, decade, and octave sweeps
- **Transient analysis** - Time-domain simulation (.TRAN) with multiple integration methods

#### Device Models
- **Passive elements** - Resistor, Capacitor, Inductor
- **Independent sources** - Voltage source, Current source with DC and time-varying waveforms
- **Controlled sources** - VCVS (E), VCCS (G), CCCS (F), CCVS (H)
- **Semiconductor devices** - Diode (Shockley model), MOSFET Level 1 (NMOS/PMOS)
- **Behavioral sources** - B elements with expression-based voltage/current
- **Time-varying sources** - PULSE, SIN, PWL waveforms

#### Solver Features
- **Newton-Raphson iteration** - For nonlinear DC and transient analysis
- **Convergence aids** - Source stepping, Gmin stepping, voltage limiting
- **Sparse solver** - faer-based sparse LU factorization with symbolic caching
- **GMRES iterative solver** - With Jacobi preconditioning for large circuits
- **Automatic solver selection** - Based on circuit size

#### Integration Methods
- Backward Euler (1st order, A-stable)
- Trapezoidal (2nd order, A-stable)
- TR-BDF2 (2nd order, L-stable composite method)
- Adaptive timestep control with LTE estimation

#### Hierarchical Design
- **.SUBCKT/.ENDS** - Subcircuit definitions with nested instantiation
- **.MODEL** - Device model definitions for diodes and MOSFETs
- **.IC** - Initial conditions with UIC option

#### Output Control
- **.PRINT** - Variable selection for DC, AC, and TRAN output
- Multiple output formats including V(node), I(device), VM, VP, VDB

#### Performance Features
- **SIMD acceleration** - AVX2/AVX-512 kernels for dot products and matvec
- **Batched device evaluation** - SoA layouts for vectorized nonlinear evaluation
- **Parallel matrix assembly** - Thread-local triplet accumulation
- **Parameter sweeps** - Monte Carlo, corner analysis, and linear sweep generators

#### Compute Backends
- **CPU backend** - SIMD-accelerated dense operators
- **CUDA backend** - cuBLAS-based GPU operators (optional)
- **Metal/WebGPU backend** - WGSL compute shaders (optional)

#### Validation
- **Cross-simulator comparison** - spicier-validate crate for comparing with ngspice
- **Golden data validation** - JSON-based reference data for regression testing
- **320+ unit and integration tests**

### Crates

- `spicier-core` - Circuit graph, MNA matrices, node management
- `spicier-solver` - Linear/nonlinear solvers, DC/AC/transient analysis
- `spicier-devices` - Device models, stamps, waveforms
- `spicier-parser` - SPICE netlist lexer and parser
- `spicier-simd` - SIMD-accelerated numerical kernels
- `spicier-backend-cpu` - CPU dense operators with SIMD
- `spicier-backend-cuda` - CUDA/cuBLAS operators (optional)
- `spicier-backend-metal` - Metal/WebGPU operators (optional)
- `spicier-cli` - Command-line interface
- `spicier-validate` - Cross-simulator validation tool

[0.1.0]: https://github.com/rwalters/spicier/releases/tag/v0.1.0
