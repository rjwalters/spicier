//! BSIM4 derived (pre-calculated) parameters.
//!
//! These are parameters computed once from the model/instance parameters
//! and reused during device evaluation. Includes BSIM4-specific derived
//! values for quantum mechanical effects and stress modeling.

use super::params::Bsim4Params;

/// Pre-calculated BSIM4 parameters derived from model parameters.
#[derive(Debug, Clone)]
pub struct Bsim4Derived {
    /// Effective channel length (m)
    pub leff: f64,
    /// Effective channel width (m)
    pub weff: f64,
    /// Electrical oxide capacitance per unit area (F/m^2)
    pub coxe: f64,
    /// Physical oxide capacitance per unit area (F/m^2)
    pub coxp: f64,
    /// Thermal voltage (V) at operating temperature
    pub vt: f64,
    /// sqrt(2 * q * eps_si * Nch)
    pub sqrtk1: f64,
    /// Surface potential at strong inversion (~2*phi_s)
    pub phi: f64,
    /// Sqrt of surface potential
    pub sqrt_phi: f64,
    /// Channel doping term for body effect
    pub k1ox: f64,
    /// Low-field mobility in m^2/V-s (converted from cm^2/V-s)
    pub u0_si: f64,
    /// Pre-computed characteristic length for SCE
    pub lt: f64,
    /// Source/drain resistance (ohms)
    pub rds: f64,
    /// Abulk coefficient for bulk charge effect
    pub abulk0: f64,
    /// Built-in potential Vbi (V) for SCE calculation
    pub vbi: f64,

    // ========================================
    // Quantum Mechanical Derived Values
    // ========================================
    /// QM correction to threshold voltage (V)
    pub dvth_qm: f64,
    /// Poly depletion voltage shift (V)
    pub dvth_poly: f64,

    // ========================================
    // Stress Effect Derived Values
    // ========================================
    /// Stress-adjusted mobility factor (multiplier)
    pub stress_mu_factor: f64,
    /// Stress-adjusted Vth shift (V)
    pub stress_vth_shift: f64,

    // ========================================
    // Capacitance-related derived values
    // ========================================
    /// Total gate-source overlap capacitance (F)
    pub cgs_ov: f64,
    /// Total gate-drain overlap capacitance (F)
    pub cgd_ov: f64,
    /// Total gate-bulk overlap capacitance (F)
    pub cgb_ov: f64,
    /// Source diffusion area (m^2) - auto-calculated if zero
    pub as_eff: f64,
    /// Drain diffusion area (m^2) - auto-calculated if zero
    pub ad_eff: f64,
    /// Source diffusion perimeter (m) - auto-calculated if zero
    pub ps_eff: f64,
    /// Drain diffusion perimeter (m) - auto-calculated if zero
    pub pd_eff: f64,

    // ========================================
    // Temperature-scaled parameters
    // ========================================
    /// Operating temperature (K)
    pub temp: f64,
    /// Temperature ratio T/Tnom
    pub temp_ratio: f64,
    /// Temperature-scaled threshold voltage offset (V)
    pub vth0_temp: f64,
    /// Temperature-scaled saturation velocity (m/s)
    pub vsat_temp: f64,
    /// Temperature-scaled UA mobility degradation (m/V)
    pub ua_temp: f64,
    /// Temperature-scaled UB mobility degradation (m/V)^2
    pub ub_temp: f64,
    /// Temperature-scaled UC mobility degradation (m/V^2)
    pub uc_temp: f64,
}

impl Bsim4Derived {
    /// Compute derived parameters from model parameters at nominal temperature.
    pub fn from_params(p: &Bsim4Params) -> Self {
        Self::from_params_at_temp(p, p.tnom)
    }

    /// Compute derived parameters at a specific operating temperature.
    pub fn from_params_at_temp(p: &Bsim4Params, temp: f64) -> Self {
        let leff = p.leff();
        let weff = p.weff();
        let coxe = p.coxe();
        let coxp = p.coxp();

        // Thermal voltage at operating temperature
        let vt = p.vt_at(temp);

        // Temperature ratio
        let temp_ratio = temp / p.tnom;
        let delta_t = temp - p.tnom;

        // Surface potential: phi = 2 * vt * ln(Nch / ni)
        let vt_nom = p.vt();
        let phi_tnom = 2.0 * vt_nom * (p.nch / Bsim4Params::NI).ln();
        let phi = phi_tnom * temp_ratio;
        let sqrt_phi = phi.sqrt();

        // Body effect coefficient
        let sqrtk1 = (2.0 * Bsim4Params::Q * Bsim4Params::EPS_SI * p.nch * 1e6).sqrt();
        let k1ox = sqrtk1 / coxe;

        // Temperature-scaled mobility
        let u0_scaled = p.u0 * temp_ratio.powf(p.ute);
        let u0_si = u0_scaled * 1e-4;

        // Characteristic length for SCE
        let xdep = (2.0 * Bsim4Params::EPS_SI * phi / (Bsim4Params::Q * p.nch * 1e6)).sqrt();
        let lt = (Bsim4Params::EPS_SI * xdep * p.toxe / Bsim4Params::EPS_OX).sqrt();

        // Temperature-scaled source/drain resistance
        let rds_tnom = p.rdsw / (weff * 1e6);
        let rds = rds_tnom * (1.0 + p.prt * delta_t);

        // Bulk charge effect coefficient
        let abulk0 = 1.0 + k1ox / (2.0 * sqrt_phi);

        // Built-in potential
        let vbi = vt * ((p.nch * 1e6) / Bsim4Params::NI).ln() + 0.56;

        // ========================================
        // Quantum Mechanical Effects (BSIM4-specific)
        // ========================================

        // QM correction to threshold voltage
        // The QM effect raises the effective bandgap due to quantization of
        // the inversion layer, shifting Vth upward.
        // dVth_QM ≈ qme1 * (eps_si * Vt / (q * Nch * 1e6))^(1/3)
        let dvth_qm = if p.qme1 != 0.0 {
            let qm_arg = Bsim4Params::EPS_SI * vt / (Bsim4Params::Q * p.nch * 1e6);
            let qm_factor = qm_arg.cbrt();
            p.qme1 * qm_factor + p.qme2 * leff + p.qme3 * (leff * leff)
        } else {
            0.0
        };

        // Poly depletion effect
        // When gate is polysilicon, depletion in the gate reduces effective Cox
        // dVth_poly ≈ Vt * ln(Ngate/ni) * eps_si / (eps_ox * Ngate * tox)
        // Simplified: adds a small positive shift to Vth
        let dvth_poly = if p.polymod == 1 && p.ngate_poly > 0.0 && p.ngate_poly < 1e25 {
            let phis_gate = vt * (p.ngate_poly / Bsim4Params::NI).ln();
            // Poly depletion capacitance in series with oxide
            let xdpoly = (2.0 * Bsim4Params::EPS_SI * phis_gate
                / (Bsim4Params::Q * p.ngate_poly * 1e6))
            .sqrt();
            // Additional voltage drop across poly depletion region
            // Simplified model: dV ≈ Q_inv * xdpoly / eps_si
            // For small effect, approximate as fraction of phi
            let cpoly = Bsim4Params::EPS_SI / xdpoly.max(1e-12);
            // Series capacitance reduces effective Cox
            // Ceff = Cox * Cpoly / (Cox + Cpoly)
            // dVth_poly ≈ Vgs * Cox / (Cox + Cpoly) correction
            // Simplified to small correction
            let _ceff_ratio = coxe / (coxe + cpoly);
            // Typical poly depletion shift: 10-50mV
            (coxe / cpoly).min(0.1) * phi * 0.5
        } else {
            0.0
        };

        // ========================================
        // Stress Effects (BSIM4-specific)
        // ========================================
        let (stress_mu_factor, stress_vth_shift) = if p.sa > 0.0 && p.sb > 0.0 {
            // Layout-dependent stress from STI (shallow trench isolation)
            // The stress depends on the distance from gate to OD edge
            // Mobility enhancement/degradation: du0/u0 = ku0 * (1/SA + 1/SB - 1/SAref - 1/SBref)
            let inv_sa = 1.0 / p.sa;
            let inv_sb = 1.0 / p.sb;
            let inv_saref = 1.0 / p.saref;
            let inv_sbref = 1.0 / p.sbref;

            let stress_term = (inv_sa + inv_sb) - (inv_saref + inv_sbref);
            let mu_factor = 1.0 + p.ku0 * stress_term * (1.0 + p.stk2 * delta_t);
            let vth_shift = p.kvth0 * stress_term * (1.0 + p.stheta * delta_t);

            (mu_factor.max(0.5), vth_shift)
        } else {
            (1.0, 0.0)
        };

        // Overlap capacitances
        let cgs_ov = p.cgso * weff;
        let cgd_ov = p.cgdo * weff;
        let cgb_ov = p.cgbo * leff;

        // Diffusion areas and perimeters
        let diff_length = 0.5e-6;
        let as_eff = if p.as_ > 0.0 {
            p.as_
        } else {
            weff * diff_length
        };
        let ad_eff = if p.ad > 0.0 { p.ad } else { weff * diff_length };
        let ps_eff = if p.ps > 0.0 {
            p.ps
        } else {
            2.0 * (weff + diff_length)
        };
        let pd_eff = if p.pd > 0.0 {
            p.pd
        } else {
            2.0 * (weff + diff_length)
        };

        // Temperature-scaled parameters
        let vth0_temp = p.kt1 * (temp_ratio - 1.0) + p.kt1l / leff * (temp_ratio - 1.0);
        let vsat_temp = (p.vsat - p.at * delta_t).max(1e4);
        let ua_temp = p.ua + p.ua1 * delta_t;
        let ub_temp = p.ub + p.ub1 * delta_t;
        let uc_temp = p.uc + p.uc1 * delta_t;

        Self {
            leff,
            weff,
            coxe,
            coxp,
            vt,
            sqrtk1,
            phi,
            sqrt_phi,
            k1ox,
            u0_si,
            lt,
            rds,
            abulk0,
            vbi,
            dvth_qm,
            dvth_poly,
            stress_mu_factor,
            stress_vth_shift,
            cgs_ov,
            cgd_ov,
            cgb_ov,
            as_eff,
            ad_eff,
            ps_eff,
            pd_eff,
            temp,
            temp_ratio,
            vth0_temp,
            vsat_temp,
            ua_temp,
            ub_temp,
            uc_temp,
        }
    }

    /// Calculate junction capacitance with voltage dependence.
    pub fn junction_cap(cj0: f64, v: f64, pb: f64, mj: f64) -> f64 {
        const FC: f64 = 0.5;

        if cj0 <= 0.0 {
            return 0.0;
        }

        let v = v.min(pb * 0.95);

        if v < FC * pb {
            let ratio = 1.0 - v / pb;
            cj0 / ratio.powf(mj)
        } else {
            let f1 = (1.0 - FC).powf(1.0 + mj);
            let f2 = 1.0 + mj;
            let f3 = 1.0 - FC * (1.0 + mj);
            cj0 / f1 * (f3 + mj * v / pb / f2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_derived_nmos() {
        let params = Bsim4Params::nmos_default();
        let derived = Bsim4Derived::from_params(&params);

        assert!((derived.leff - 100e-9).abs() < 1e-12);
        assert!((derived.weff - 1e-6).abs() < 1e-12);
        assert!((derived.vt - 0.0259).abs() < 0.001);
        assert!(derived.coxe > 7e-3 && derived.coxe < 10e-3);
        assert!(derived.phi > 0.7 && derived.phi < 1.1);
        assert!(derived.lt > 5e-9 && derived.lt < 100e-9);
    }

    #[test]
    fn test_qm_correction_default_zero() {
        let params = Bsim4Params::nmos_default();
        let derived = Bsim4Derived::from_params(&params);
        assert_eq!(derived.dvth_qm, 0.0);
    }

    #[test]
    fn test_qm_correction_nonzero() {
        let mut params = Bsim4Params::nmos_default();
        params.qme1 = 0.5;
        let derived = Bsim4Derived::from_params(&params);
        // QM correction should be positive (raises Vth)
        assert!(
            derived.dvth_qm > 0.0,
            "QM correction should be positive: {}",
            derived.dvth_qm
        );
        // Typical magnitude: a few mV to ~50mV
        assert!(
            derived.dvth_qm < 1.0,
            "QM correction should be < 1V: {}",
            derived.dvth_qm
        );
    }

    #[test]
    fn test_stress_default_unity() {
        let params = Bsim4Params::nmos_default();
        let derived = Bsim4Derived::from_params(&params);
        assert_eq!(derived.stress_mu_factor, 1.0);
        assert_eq!(derived.stress_vth_shift, 0.0);
    }

    #[test]
    fn test_stress_effect() {
        let mut params = Bsim4Params::nmos_default();
        params.sa = 0.5e-6;
        params.sb = 0.5e-6;
        params.saref = 1e-6;
        params.sbref = 1e-6;
        params.ku0 = 1e-6;
        params.kvth0 = 1e-6;
        let derived = Bsim4Derived::from_params(&params);

        // SA/SB < SAref/SBref means more stress -> mobility change
        assert!(
            derived.stress_mu_factor != 1.0,
            "Stress should modify mobility factor"
        );
    }

    #[test]
    fn test_temperature_scaling() {
        let params = Bsim4Params::nmos_default();

        let derived_nom = Bsim4Derived::from_params(&params);
        let derived_hot = Bsim4Derived::from_params_at_temp(&params, 400.0);

        // Mobility should decrease with temperature
        assert!(derived_hot.u0_si < derived_nom.u0_si);

        // Vsat should decrease with temperature
        assert!(derived_hot.vsat_temp < derived_nom.vsat_temp);

        // Vth0_temp should be negative at high temp (KT1 < 0)
        assert!(derived_hot.vth0_temp < 0.0);
    }
}
