# spicier-devices

Device models and MNA stamps for the Spicier circuit simulator.

## Supported Devices

- **Passive elements** - Resistor, Capacitor, Inductor
- **Independent sources** - Voltage source, Current source
- **Controlled sources** - VCVS (E), VCCS (G), CCCS (F), CCVS (H)
- **Semiconductors** - Diode (Shockley), MOSFET Level 1 (NMOS/PMOS)
- **Behavioral** - B elements with expression-based V/I

## Waveforms

Time-varying source waveforms for transient analysis:
- **PULSE** - Periodic pulse with rise/fall times
- **SIN** - Sinusoidal with optional damping
- **PWL** - Piecewise linear arbitrary waveform

## Usage

```rust
use spicier_devices::{Resistor, Capacitor, VoltageSource, Diode};

let r1 = Resistor::new(Some(0), Some(1), 1000.0); // 1k resistor
let c1 = Capacitor::new(Some(1), None, 1e-6);     // 1uF capacitor
let v1 = VoltageSource::new(Some(0), None, 5.0);  // 5V source
let d1 = Diode::new(Some(1), None);               // Diode with defaults
```

## License

Licensed under either of Apache License, Version 2.0 or MIT license at your option.
