# Golden Data Format

This directory contains pre-computed reference data from ngspice for validation testing.

## File Format

Each `.json` file contains golden data for one or more test circuits:

```json
{
  "generator": "ngspice-43",
  "generated_at": "2026-02-01T12:00:00Z",
  "circuits": [
    {
      "name": "voltage_divider",
      "description": "Simple resistive voltage divider",
      "netlist": "V1 1 0 DC 10\nR1 1 2 1k\nR2 2 0 1k",
      "analysis": {
        "type": "dc_op",
        "results": {
          "V(1)": 10.0,
          "V(2)": 5.0,
          "I(V1)": -0.005
        },
        "tolerances": {
          "voltage": 1e-6,
          "current": 1e-9
        }
      }
    }
  ]
}
```

## Analysis Types

### `dc_op` - DC Operating Point
```json
{
  "type": "dc_op",
  "results": {
    "V(node)": value,
    "I(source)": value
  }
}
```

### `dc_sweep` - DC Sweep
```json
{
  "type": "dc_sweep",
  "source": "V1",
  "sweep": { "start": 0, "stop": 10, "step": 1 },
  "results": [
    { "V1": 0, "V(2)": 0 },
    { "V1": 1, "V(2)": 0.5 }
  ]
}
```

### `ac` - AC Analysis
```json
{
  "type": "ac",
  "sweep": { "type": "dec", "points": 10, "fstart": 1, "fstop": 1e6 },
  "results": [
    { "freq": 1.0, "V(2)": { "mag_db": -0.001, "phase_deg": -0.036 } },
    { "freq": 10.0, "V(2)": { "mag_db": -0.043, "phase_deg": -0.36 } }
  ]
}
```

### `tran` - Transient Analysis
```json
{
  "type": "tran",
  "params": { "tstop": 1e-3, "tstep": 1e-6 },
  "results": [
    { "time": 0, "V(2)": 0 },
    { "time": 1e-6, "V(2)": 0.00632 }
  ]
}
```

## Generating Golden Data

Golden data can be generated using ngspice:

```bash
ngspice -b circuit.sp -o output.raw
# Then extract values using ngspice scripting or external tools
```

## Tolerance Guidelines

- DC voltage: 1e-6 V (1 µV) for linear circuits, 1e-3 V (1 mV) for nonlinear
- DC current: 1e-9 A (1 nA) for most circuits
- AC magnitude: 0.01 dB
- AC phase: 0.1 degrees
- Transient voltage: 1e-4 V relative to signal amplitude
