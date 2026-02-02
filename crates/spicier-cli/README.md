# spicier-cli

Command-line interface for the Spicier circuit simulator.

## Installation

```bash
cargo install spicier-cli
```

## Usage

```bash
# Run DC operating point analysis
spicier circuit.sp

# Verbose output
spicier -v circuit.sp

# Show help
spicier --help
```

## Supported Analysis

The CLI automatically detects analysis commands in the netlist:

- `.OP` - DC operating point
- `.DC` - DC sweep (single or nested)
- `.AC` - AC small-signal analysis
- `.TRAN` - Transient time-domain simulation

## Example

```spice
* RC Low-pass Filter
V1 1 0 DC 0 AC 1
R1 1 2 1k
C1 2 0 1u
.AC DEC 10 1 100k
.END
```

```bash
$ spicier rc_filter.sp
AC Analysis: 1.000e0 Hz to 1.000e5 Hz (31 points)

Freq          V(2) Mag(dB)   V(2) Phase(deg)
1.000e+00     -0.00         -0.06
...
```

## License

Licensed under either of Apache License, Version 2.0 or MIT license at your option.
