//! BSIM4 model parameters.
//!
//! This module defines the parameter struct for the BSIM4 model (Level 54).
//! BSIM4 extends BSIM3 with quantum mechanical effects, stress modeling,
//! and gate tunneling current.

use super::super::level1::MosfetType;

/// BSIM4 model parameters.
///
/// This "BSIM4-lite" covers the essential physics for SKY130 PDK support:
/// - Core BSIM3-compatible DC parameters
/// - Quantum mechanical corrections (poly depletion, inversion quantization)
/// - Stress effects (layout-dependent SA/SB/SD)
/// - Gate tunneling current (IGIDL, IGISL)
#[derive(Debug, Clone)]
pub struct Bsim4Params {
    // ========================================
    // Geometry Parameters
    // ========================================
    /// Gate oxide thickness (m). Default: 4.1e-9 (SKY130)
    pub toxe: f64,
    /// Physical oxide thickness for tunneling (m). Default: 4.1e-9
    pub toxp: f64,
    /// Oxide thickness for mobility model (m). Default: 4.1e-9
    pub toxm: f64,
    /// Channel length reduction per side (m). Default: 0.0
    pub lint: f64,
    /// Channel width reduction per side (m). Default: 0.0
    pub wint: f64,
    /// Minimum channel length for BSIM4 (m). Default: 0.0
    pub lmin: f64,
    /// Minimum channel width for BSIM4 (m). Default: 0.0
    pub wmin: f64,

    // ========================================
    // Threshold Voltage Parameters
    // ========================================
    /// Zero-bias threshold voltage (V). Default: 0.4 (NMOS), -0.4 (PMOS)
    pub vth0: f64,
    /// First-order body effect coefficient (V^0.5). Default: 0.5
    pub k1: f64,
    /// Second-order body effect coefficient. Default: 0.0
    pub k2: f64,
    /// Short-channel effect coefficient 0. Default: 2.2
    pub dvt0: f64,
    /// Short-channel effect coefficient 1. Default: 0.53
    pub dvt1: f64,
    /// Body-bias coefficient for SCE. Default: -0.032
    pub dvt2: f64,
    /// Lateral non-uniform doping parameter (m). Default: 1.74e-7
    pub nlx: f64,
    /// Threshold voltage offset in subthreshold (V). Default: -0.11
    pub voff: f64,
    /// Subthreshold swing factor. Default: 1.0
    pub nfactor: f64,
    /// Minimum drain-source conductance. Default: 0.0
    pub vfb: f64,

    // ========================================
    // Width Effect Parameters
    // ========================================
    /// Narrow width threshold voltage coefficient. Default: 0.0
    pub k3: f64,
    /// Body effect coefficient for narrow width. Default: 0.0
    pub k3b: f64,
    /// Narrow width parameter (m). Default: 0.0
    pub w0: f64,
    /// Narrow width SCE coefficient 0. Default: 0.0
    pub dvt0w: f64,
    /// Narrow width SCE coefficient 1. Default: 0.0
    pub dvt1w: f64,
    /// Narrow width SCE body coefficient. Default: 0.0
    pub dvt2w: f64,

    // ========================================
    // Mobility Parameters
    // ========================================
    /// Low-field mobility (cm^2/V-s). Default: 670 (NMOS), 250 (PMOS)
    pub u0: f64,
    /// First-order mobility degradation (m/V). Default: 2.25e-9
    pub ua: f64,
    /// Second-order mobility degradation (m/V)^2. Default: 5.87e-19
    pub ub: f64,
    /// Body-bias mobility degradation (m/V^2). Default: -4.65e-11
    pub uc: f64,
    /// Saturation velocity (m/s). Default: 1.5e5
    pub vsat: f64,
    /// Mobility degradation exponent selector. Default: 0 (BSIM3 style)
    pub mobmod: i32,
    /// Minimum channel length for velocity overshoot. Default: 0.0
    pub eu: f64,

    // ========================================
    // Output Conductance Parameters
    // ========================================
    /// Channel length modulation parameter. Default: 1.3
    pub pclm: f64,
    /// DIBL coefficient 1. Default: 0.39
    pub pdiblc1: f64,
    /// DIBL coefficient 2. Default: 0.0086
    pub pdiblc2: f64,
    /// DIBL output resistance parameter. Default: 0.56
    pub drout: f64,
    /// Effective Vds smoothing parameter (V). Default: 0.01
    pub delta: f64,
    /// Body effect on DIBL output resistance. Default: 0.0
    pub pdiblcb: f64,
    /// DIBL to Rout parameter. Default: 0.0
    pub fprout: f64,
    /// DIBL multiplier for Vth. Default: 1.0
    pub pvag: f64,

    // ========================================
    // DIBL Parameters
    // ========================================
    /// DIBL coefficient. Default: 0.08
    pub eta0: f64,
    /// Body-bias DIBL coefficient (1/V). Default: -0.07
    pub etab: f64,
    /// DIBL coefficient in subthreshold. Default: 0.56
    pub dsub: f64,

    // ========================================
    // Substrate Current Parameters
    // ========================================
    /// Substrate current impact ionization coefficient. Default: 0.0
    pub alpha0: f64,
    /// Substrate current exponent. Default: 30.0
    pub beta0: f64,

    // ========================================
    // Parasitic Resistance
    // ========================================
    /// Source/drain resistance per width (ohm-um). Default: 200
    pub rdsw: f64,
    /// Zero-bias drain resistance. Default: 0.0
    pub rd: f64,
    /// Zero-bias source resistance. Default: 0.0
    pub rs: f64,
    /// Width dependence of Rds. Default: 0.0
    pub prwb: f64,
    /// Gate-bias dependence of Rds. Default: 0.0
    pub prwg: f64,
    /// Source/drain sheet resistance. Default: 0.0
    pub rsh: f64,

    // ========================================
    // Process Parameters
    // ========================================
    /// Channel doping concentration (cm^-3). Default: 1.7e17
    pub nch: f64,
    /// Gate doping concentration (cm^-3). Default: 1e20
    pub ngate: f64,
    /// Substrate doping concentration (cm^-3). Default: 1e15
    pub nsub: f64,
    /// Vertical doping profile characteristic length (m). Default: 0.0
    pub xt: f64,
    /// Doping profile parameter. Default: 1.0
    pub ndep: f64,

    // ========================================
    // Quantum Mechanical Effect Parameters (BSIM4-specific)
    // ========================================
    /// Quantum mechanical correction factor for Vth. Default: 0.0
    /// When nonzero, adds QM correction to threshold voltage:
    /// dVth_QM ≈ -(qme1 / Cox) * (eps_si * Vt / (q * Nch))^(1/3)
    pub qme1: f64,
    /// Second QM correction parameter. Default: 0.0
    pub qme2: f64,
    /// Third QM correction parameter. Default: 0.0
    pub qme3: f64,
    /// Poly depletion effect flag: 0=off, 1=on. Default: 1
    pub polymod: i32,
    /// Poly gate N-type doping. Default: 1e20
    pub ngate_poly: f64,

    // ========================================
    // Stress Effect Parameters (BSIM4-specific)
    // ========================================
    /// Reference distance between OD edge and gate (m). Default: 1e-6
    pub saref: f64,
    /// Reference distance between gate and OD edge on other side (m). Default: 1e-6
    pub sbref: f64,
    /// Instance SA parameter (m). Default: 0.0 (no stress)
    pub sa: f64,
    /// Instance SB parameter (m). Default: 0.0 (no stress)
    pub sb: f64,
    /// Instance SD parameter (m). Default: 0.0 (no stress)
    pub sd: f64,
    /// Mobility stress coefficient 1. Default: 0.0
    pub ku0: f64,
    /// Vth stress coefficient 1. Default: 0.0
    pub kvth0: f64,
    /// Stress temp coefficient for ku0. Default: 0.0
    pub stk2: f64,
    /// Stress temp coefficient for kvth0. Default: 0.0
    pub stheta: f64,

    // ========================================
    // Gate Tunneling Current (IGIDL/IGISL)
    // ========================================
    /// IGIDL coefficient. Default: 0.0 (disabled)
    pub agidl: f64,
    /// IGIDL exponential coefficient. Default: 0.8
    pub bgidl: f64,
    /// IGIDL reference voltage. Default: 0.3
    pub cgidl: f64,
    /// IGIDL body bias coefficient. Default: 0.0
    pub egidl: f64,

    // ========================================
    // Capacitance Parameters
    // ========================================
    /// Gate-source overlap capacitance per unit width (F/m). Default: 0.0
    pub cgso: f64,
    /// Gate-drain overlap capacitance per unit width (F/m). Default: 0.0
    pub cgdo: f64,
    /// Gate-bulk overlap capacitance per unit length (F/m). Default: 0.0
    pub cgbo: f64,
    /// Zero-bias bulk-drain junction capacitance per unit area (F/m^2). Default: 5e-4
    pub cj: f64,
    /// Zero-bias bulk-drain sidewall capacitance per unit length (F/m). Default: 5e-10
    pub cjsw: f64,
    /// Zero-bias gate-edge sidewall capacitance per unit length (F/m). Default: 0.0
    pub cjswg: f64,
    /// Bulk junction bottom grading coefficient. Default: 0.5
    pub mj: f64,
    /// Bulk junction sidewall grading coefficient. Default: 0.33
    pub mjsw: f64,
    /// Bulk junction gate-side sidewall grading coefficient. Default: 0.33
    pub mjswg: f64,
    /// Bulk junction built-in potential (V). Default: 1.0
    pub pb: f64,
    /// Bulk junction sidewall built-in potential (V). Default: 1.0
    pub pbsw: f64,
    /// Bulk junction gate-side sidewall built-in potential (V). Default: 1.0
    pub pbswg: f64,
    /// Capacitance model selector. Default: 0
    pub capmod: i32,

    // ========================================
    // Temperature Parameters
    // ========================================
    /// Nominal temperature for parameter extraction (K). Default: 300.15
    pub tnom: f64,
    /// First-order Vth temperature coefficient (V/K). Default: -0.11
    pub kt1: f64,
    /// Body-bias Vth temperature coefficient (V/K). Default: 0.0
    pub kt1l: f64,
    /// Second-order Vth temperature coefficient (V/K^2). Default: 0.022
    pub kt2: f64,
    /// Mobility temperature exponent. Default: -1.5
    pub ute: f64,
    /// UA temperature coefficient (m/V/K). Default: 4.31e-9
    pub ua1: f64,
    /// UB temperature coefficient ((m/V)^2/K). Default: -7.61e-18
    pub ub1: f64,
    /// UC temperature coefficient (m/V^2/K). Default: -5.6e-11
    pub uc1: f64,
    /// Saturation velocity temperature coefficient (m/s/K). Default: 3.3e2
    pub at: f64,
    /// RDSW temperature coefficient (ohm-um/K). Default: 0.0
    pub prt: f64,

    // ========================================
    // Instance Parameters (set per device)
    // ========================================
    /// Channel width (m). Default: 1e-6
    pub w: f64,
    /// Channel length (m). Default: 100e-9
    pub l: f64,
    /// Number of fingers. Default: 1
    pub nf: f64,
    /// Source diffusion area (m^2). Default: 0.0 (auto-calculated if zero)
    pub as_: f64,
    /// Drain diffusion area (m^2). Default: 0.0 (auto-calculated if zero)
    pub ad: f64,
    /// Source diffusion perimeter (m). Default: 0.0 (auto-calculated if zero)
    pub ps: f64,
    /// Drain diffusion perimeter (m). Default: 0.0 (auto-calculated if zero)
    pub pd: f64,
    /// Number of drain/source diffusion squares. Default: 0.0
    pub nrd: f64,
    /// Number of drain/source diffusion squares. Default: 0.0
    pub nrs: f64,
    /// Multiplier. Default: 1.0
    pub mult: f64,
    /// Device type (set by model type)
    pub mos_type: MosfetType,
}

impl Bsim4Params {
    /// Physical constants used in BSIM4 calculations.
    pub const Q: f64 = 1.602176634e-19; // Elementary charge (C)
    pub const KB: f64 = 1.380649e-23; // Boltzmann constant (J/K)
    pub const EPS_SI: f64 = 1.03594e-10; // Permittivity of Si (F/m) = 11.7 * eps0
    pub const EPS_OX: f64 = 3.45314e-11; // Permittivity of SiO2 (F/m) = 3.9 * eps0
    pub const NI: f64 = 1.45e10; // Intrinsic carrier concentration at 300K (cm^-3)
    pub const T_NOM: f64 = 300.15; // Nominal temperature (K)
    pub const HBAR: f64 = 1.054571817e-34; // Reduced Planck constant (J-s)
    pub const M_EFF: f64 = 9.11e-31; // Electron effective mass (kg) * 0.19 (Si)

    /// Create default NMOS BSIM4 parameters.
    pub fn nmos_default() -> Self {
        Self {
            // Geometry
            toxe: 4.1e-9,
            toxp: 4.1e-9,
            toxm: 4.1e-9,
            lint: 0.0,
            wint: 0.0,
            lmin: 0.0,
            wmin: 0.0,

            // Threshold voltage
            vth0: 0.4,
            k1: 0.5,
            k2: 0.0,
            dvt0: 2.2,
            dvt1: 0.53,
            dvt2: -0.032,
            nlx: 1.74e-7,
            voff: -0.11,
            nfactor: 1.0,
            vfb: -1.0,

            // Width effects
            k3: 0.0,
            k3b: 0.0,
            w0: 0.0,
            dvt0w: 0.0,
            dvt1w: 0.0,
            dvt2w: 0.0,

            // Mobility
            u0: 670.0,
            ua: 2.25e-9,
            ub: 5.87e-19,
            uc: -4.65e-11,
            vsat: 1.5e5,
            mobmod: 0,
            eu: 1.67,

            // Output conductance
            pclm: 1.3,
            pdiblc1: 0.39,
            pdiblc2: 0.0086,
            drout: 0.56,
            delta: 0.01,
            pdiblcb: 0.0,
            fprout: 0.0,
            pvag: 1.0,

            // DIBL
            eta0: 0.08,
            etab: -0.07,
            dsub: 0.56,

            // Substrate current
            alpha0: 0.0,
            beta0: 30.0,

            // Parasitic resistance
            rdsw: 200.0,
            rd: 0.0,
            rs: 0.0,
            prwb: 0.0,
            prwg: 0.0,
            rsh: 0.0,

            // Process
            nch: 1.7e17,
            ngate: 1e20,
            nsub: 1e15,
            xt: 0.0,
            ndep: 1.0,

            // Quantum mechanical effects
            qme1: 0.0,
            qme2: 0.0,
            qme3: 0.0,
            polymod: 1,
            ngate_poly: 1e20,

            // Stress effects (disabled by default)
            saref: 1e-6,
            sbref: 1e-6,
            sa: 0.0,
            sb: 0.0,
            sd: 0.0,
            ku0: 0.0,
            kvth0: 0.0,
            stk2: 0.0,
            stheta: 0.0,

            // Gate tunneling current (disabled by default)
            agidl: 0.0,
            bgidl: 0.8,
            cgidl: 0.3,
            egidl: 0.0,

            // Capacitances
            cgso: 0.0,
            cgdo: 0.0,
            cgbo: 0.0,
            cj: 5e-4,
            cjsw: 5e-10,
            cjswg: 0.0,
            mj: 0.5,
            mjsw: 0.33,
            mjswg: 0.33,
            pb: 1.0,
            pbsw: 1.0,
            pbswg: 1.0,
            capmod: 0,

            // Temperature
            tnom: 300.15,
            kt1: -0.11,
            kt1l: 0.0,
            kt2: 0.022,
            ute: -1.5,
            ua1: 4.31e-9,
            ub1: -7.61e-18,
            uc1: -5.6e-11,
            at: 3.3e2,
            prt: 0.0,

            // Instance (defaults)
            w: 1e-6,
            l: 100e-9,
            nf: 1.0,
            as_: 0.0,
            ad: 0.0,
            ps: 0.0,
            pd: 0.0,
            nrd: 0.0,
            nrs: 0.0,
            mult: 1.0,
            mos_type: MosfetType::Nmos,
        }
    }

    /// Create default PMOS BSIM4 parameters.
    pub fn pmos_default() -> Self {
        Self {
            // Geometry
            toxe: 4.1e-9,
            toxp: 4.1e-9,
            toxm: 4.1e-9,
            lint: 0.0,
            wint: 0.0,
            lmin: 0.0,
            wmin: 0.0,

            // Threshold voltage (negative for PMOS)
            vth0: -0.4,
            k1: 0.5,
            k2: 0.0,
            dvt0: 2.2,
            dvt1: 0.53,
            dvt2: -0.032,
            nlx: 1.74e-7,
            voff: -0.11,
            nfactor: 1.0,
            vfb: -1.0,

            // Width effects
            k3: 0.0,
            k3b: 0.0,
            w0: 0.0,
            dvt0w: 0.0,
            dvt1w: 0.0,
            dvt2w: 0.0,

            // Mobility (lower for PMOS)
            u0: 250.0,
            ua: 2.25e-9,
            ub: 5.87e-19,
            uc: -4.65e-11,
            vsat: 1.0e5,
            mobmod: 0,
            eu: 1.67,

            // Output conductance
            pclm: 1.3,
            pdiblc1: 0.39,
            pdiblc2: 0.0086,
            drout: 0.56,
            delta: 0.01,
            pdiblcb: 0.0,
            fprout: 0.0,
            pvag: 1.0,

            // DIBL
            eta0: 0.08,
            etab: -0.07,
            dsub: 0.56,

            // Substrate current
            alpha0: 0.0,
            beta0: 30.0,

            // Parasitic resistance (higher for PMOS)
            rdsw: 300.0,
            rd: 0.0,
            rs: 0.0,
            prwb: 0.0,
            prwg: 0.0,
            rsh: 0.0,

            // Process
            nch: 1.7e17,
            ngate: 1e20,
            nsub: 1e15,
            xt: 0.0,
            ndep: 1.0,

            // Quantum mechanical effects
            qme1: 0.0,
            qme2: 0.0,
            qme3: 0.0,
            polymod: 1,
            ngate_poly: 1e20,

            // Stress effects (disabled by default)
            saref: 1e-6,
            sbref: 1e-6,
            sa: 0.0,
            sb: 0.0,
            sd: 0.0,
            ku0: 0.0,
            kvth0: 0.0,
            stk2: 0.0,
            stheta: 0.0,

            // Gate tunneling current (disabled by default)
            agidl: 0.0,
            bgidl: 0.8,
            cgidl: 0.3,
            egidl: 0.0,

            // Capacitances
            cgso: 0.0,
            cgdo: 0.0,
            cgbo: 0.0,
            cj: 5e-4,
            cjsw: 5e-10,
            cjswg: 0.0,
            mj: 0.5,
            mjsw: 0.33,
            mjswg: 0.33,
            pb: 1.0,
            pbsw: 1.0,
            pbswg: 1.0,
            capmod: 0,

            // Temperature
            tnom: 300.15,
            kt1: -0.11,
            kt1l: 0.0,
            kt2: 0.022,
            ute: -1.5,
            ua1: 4.31e-9,
            ub1: -7.61e-18,
            uc1: -5.6e-11,
            at: 3.3e2,
            prt: 0.0,

            // Instance (defaults)
            w: 1e-6,
            l: 100e-9,
            nf: 1.0,
            as_: 0.0,
            ad: 0.0,
            ps: 0.0,
            pd: 0.0,
            nrd: 0.0,
            nrs: 0.0,
            mult: 1.0,
            mos_type: MosfetType::Pmos,
        }
    }

    /// Calculate thermal voltage at nominal temperature.
    #[inline]
    pub fn vt(&self) -> f64 {
        Self::KB * self.tnom / Self::Q
    }

    /// Calculate thermal voltage at a given temperature (K).
    #[inline]
    pub fn vt_at(&self, temp: f64) -> f64 {
        Self::KB * temp / Self::Q
    }

    /// Calculate oxide capacitance per unit area (F/m^2) using electrical oxide thickness.
    #[inline]
    pub fn coxe(&self) -> f64 {
        Self::EPS_OX / self.toxe
    }

    /// Calculate oxide capacitance per unit area using physical oxide thickness.
    #[inline]
    pub fn coxp(&self) -> f64 {
        Self::EPS_OX / self.toxp
    }

    /// Calculate effective channel length (m).
    #[inline]
    pub fn leff(&self) -> f64 {
        self.l - 2.0 * self.lint
    }

    /// Calculate effective channel width (m).
    #[inline]
    pub fn weff(&self) -> f64 {
        (self.w - 2.0 * self.wint) * self.nf
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nmos_defaults() {
        let params = Bsim4Params::nmos_default();
        assert_eq!(params.vth0, 0.4);
        assert_eq!(params.u0, 670.0);
        assert_eq!(params.toxe, 4.1e-9);
        assert_eq!(params.mos_type, MosfetType::Nmos);
    }

    #[test]
    fn test_pmos_defaults() {
        let params = Bsim4Params::pmos_default();
        assert_eq!(params.vth0, -0.4);
        assert_eq!(params.u0, 250.0);
        assert_eq!(params.mos_type, MosfetType::Pmos);
    }

    #[test]
    fn test_thermal_voltage() {
        let params = Bsim4Params::nmos_default();
        let vt = params.vt();
        assert!((vt - 0.0259).abs() < 0.001);
    }

    #[test]
    fn test_oxide_capacitance() {
        let params = Bsim4Params::nmos_default();
        let coxe = params.coxe();
        // Cox = eps_ox / toxe = 3.45e-11 / 4.1e-9 ≈ 8.42e-3 F/m^2
        assert!((coxe - 8.42e-3).abs() < 1e-3);
    }

    #[test]
    fn test_effective_dimensions() {
        let mut params = Bsim4Params::nmos_default();
        params.w = 1e-6;
        params.l = 100e-9;
        params.lint = 5e-9;
        params.wint = 10e-9;
        params.nf = 2.0;

        let leff = params.leff();
        let weff = params.weff();

        assert!((leff - 90e-9).abs() < 1e-12);
        assert!((weff - 1.96e-6).abs() < 1e-12);
    }

    #[test]
    fn test_quantum_params_default_off() {
        let params = Bsim4Params::nmos_default();
        assert_eq!(params.qme1, 0.0);
        assert_eq!(params.qme2, 0.0);
        assert_eq!(params.qme3, 0.0);
    }

    #[test]
    fn test_stress_params_default_off() {
        let params = Bsim4Params::nmos_default();
        assert_eq!(params.sa, 0.0);
        assert_eq!(params.sb, 0.0);
        assert_eq!(params.ku0, 0.0);
        assert_eq!(params.kvth0, 0.0);
    }

    #[test]
    fn test_igidl_params_default_off() {
        let params = Bsim4Params::nmos_default();
        assert_eq!(params.agidl, 0.0);
    }
}
