//! BSIM4 MOSFET model (Level 54).
//!
//! This module implements the Berkeley Short-channel IGFET Model version 4,
//! extending BSIM3 with quantum mechanical effects, stress modeling,
//! and gate tunneling current for advanced process nodes.
//!
//! # Features
//!
//! This "BSIM4-lite" implementation includes:
//! - All BSIM3 physics (SCE, DIBL, mobility, velocity saturation, CLM)
//! - Quantum mechanical corrections (poly depletion, inversion quantization)
//! - Layout-dependent stress effects (SA/SB/SD via SAREF/SBREF)
//! - Gate-induced drain leakage (IGIDL/IGISL)
//! - Full MOSFET instance parameters (AD, AS, PD, PS, NRD, NRS, MULT)
//!
//! # Usage
//!
//! ```text
//! .MODEL NMOD NMOS LEVEL=54 VTH0=0.4 U0=400 TOXE=4.1e-9
//! M1 d g s b NMOD W=1u L=150n SA=0.5u SB=0.5u
//! ```
//!
//! # References
//!
//! - BSIM4 Manual: https://bsim.berkeley.edu/models/bsim4/

pub mod derived;
pub mod evaluate;
pub mod params;

pub use derived::Bsim4Derived;
pub use evaluate::{
    Bsim4CapResult, Bsim4EvalResult, Bsim4Region, evaluate as bsim4_evaluate,
    evaluate_capacitances as bsim4_evaluate_caps,
};
pub use params::Bsim4Params;

use super::level1::MosfetType;
use crate::stamp::Stamp;

use nalgebra::DVector;
use spicier_core::mna::MnaSystem;
use spicier_core::netlist::{AcDeviceInfo, TransientDeviceInfo};
use spicier_core::{Element, NodeId, Stamper};

/// A BSIM4 MOSFET device.
#[derive(Debug, Clone)]
pub struct Bsim4Mosfet {
    /// Device name (e.g., "M1").
    pub name: String,
    /// Drain node.
    pub node_drain: NodeId,
    /// Gate node.
    pub node_gate: NodeId,
    /// Source node.
    pub node_source: NodeId,
    /// Bulk (body) node.
    pub node_bulk: NodeId,
    /// Model parameters.
    pub params: Bsim4Params,
    /// Pre-calculated derived parameters.
    derived: Bsim4Derived,
}

impl Bsim4Mosfet {
    /// Create a new BSIM4 NMOS transistor with default parameters.
    pub fn nmos(
        name: impl Into<String>,
        drain: NodeId,
        gate: NodeId,
        source: NodeId,
        bulk: NodeId,
    ) -> Self {
        let params = Bsim4Params::nmos_default();
        let derived = Bsim4Derived::from_params(&params);
        Self {
            name: name.into(),
            node_drain: drain,
            node_gate: gate,
            node_source: source,
            node_bulk: bulk,
            params,
            derived,
        }
    }

    /// Create a new BSIM4 PMOS transistor with default parameters.
    pub fn pmos(
        name: impl Into<String>,
        drain: NodeId,
        gate: NodeId,
        source: NodeId,
        bulk: NodeId,
    ) -> Self {
        let params = Bsim4Params::pmos_default();
        let derived = Bsim4Derived::from_params(&params);
        Self {
            name: name.into(),
            node_drain: drain,
            node_gate: gate,
            node_source: source,
            node_bulk: bulk,
            params,
            derived,
        }
    }

    /// Create a BSIM4 MOSFET with custom parameters.
    pub fn with_params(
        name: impl Into<String>,
        drain: NodeId,
        gate: NodeId,
        source: NodeId,
        bulk: NodeId,
        params: Bsim4Params,
    ) -> Self {
        let derived = Bsim4Derived::from_params(&params);
        Self {
            name: name.into(),
            node_drain: drain,
            node_gate: gate,
            node_source: source,
            node_bulk: bulk,
            params,
            derived,
        }
    }

    /// Update derived parameters after modifying instance parameters.
    pub fn update_derived(&mut self) {
        self.derived = Bsim4Derived::from_params(&self.params);
    }

    /// Set the operating temperature and update derived parameters.
    pub fn set_temperature(&mut self, temp: f64) {
        self.derived = Bsim4Derived::from_params_at_temp(&self.params, temp);
    }

    /// Get the current operating temperature (K).
    pub fn temperature(&self) -> f64 {
        self.derived.temp
    }

    /// Get the MOSFET type.
    pub fn mos_type(&self) -> MosfetType {
        self.params.mos_type
    }

    /// Evaluate drain current and conductances at given terminal voltages.
    pub fn evaluate(&self, vgs: f64, vds: f64, vbs: f64) -> Bsim4EvalResult {
        bsim4_evaluate(&self.params, &self.derived, vgs, vds, vbs)
    }

    /// Stamp the linearized BSIM4 model into the MNA system.
    pub fn stamp_linearized_at(&self, mna: &mut MnaSystem, vgs: f64, vds: f64, vbs: f64) {
        let result = self.evaluate(vgs, vds, vbs);

        let d = node_to_index(self.node_drain);
        let g = node_to_index(self.node_gate);
        let s = node_to_index(self.node_source);
        let b = node_to_index(self.node_bulk);

        let ids = result.ids;
        let gds = result.gds;
        let gm = result.gm;
        let gmbs = result.gmbs;

        // Stamp gds (drain-source conductance)
        mna.stamp_conductance(d, s, gds);

        // Stamp gm (transconductance) as VCCS: I = gm * Vgs
        if let Some(di) = d {
            if let Some(gi) = g {
                mna.add_element(di, gi, gm);
            }
            if let Some(si) = s {
                mna.add_element(di, si, -gm);
            }
        }
        if let Some(si) = s {
            if let Some(gi) = g {
                mna.add_element(si, gi, -gm);
            }
            mna.add_element(si, si, gm);
        }

        // Stamp gmbs (body transconductance) as VCCS: I = gmbs * Vbs
        if let Some(di) = d {
            if let Some(bi) = b {
                mna.add_element(di, bi, gmbs);
            }
            if let Some(si) = s {
                mna.add_element(di, si, -gmbs);
            }
        }
        if let Some(si) = s {
            if let Some(bi) = b {
                mna.add_element(si, bi, -gmbs);
            }
            mna.add_element(si, si, gmbs);
        }

        // Equivalent current source: Ieq = Ids - gds*Vds - gm*Vgs - gmbs*Vbs
        let ieq = ids - gds * vds - gm * vgs - gmbs * vbs;
        mna.stamp_current_source(d, s, ieq);
    }
}

fn node_to_index(node: NodeId) -> Option<usize> {
    if node.is_ground() {
        None
    } else {
        Some((node.as_u32() - 1) as usize)
    }
}

impl Stamp for Bsim4Mosfet {
    fn stamp(&self, mna: &mut MnaSystem) {
        // Initial stamp: Gmin shunt between drain and source
        let d = node_to_index(self.node_drain);
        let s = node_to_index(self.node_source);
        mna.stamp_conductance(d, s, 1e-12);
    }
}

impl Element for Bsim4Mosfet {
    fn name(&self) -> &str {
        &self.name
    }

    fn nodes(&self) -> Vec<NodeId> {
        vec![
            self.node_drain,
            self.node_gate,
            self.node_source,
            self.node_bulk,
        ]
    }
}

impl Stamper for Bsim4Mosfet {
    fn stamp(&self, mna: &mut MnaSystem) {
        Stamp::stamp(self, mna);
    }

    fn device_name(&self) -> &str {
        &self.name
    }

    fn is_nonlinear(&self) -> bool {
        true
    }

    fn stamp_nonlinear(&self, mna: &mut MnaSystem, solution: &DVector<f64>) {
        let vg = node_to_index(self.node_gate)
            .map(|i| solution[i])
            .unwrap_or(0.0);
        let vd = node_to_index(self.node_drain)
            .map(|i| solution[i])
            .unwrap_or(0.0);
        let vs = node_to_index(self.node_source)
            .map(|i| solution[i])
            .unwrap_or(0.0);
        let vb = node_to_index(self.node_bulk)
            .map(|i| solution[i])
            .unwrap_or(0.0);

        let vgs = vg - vs;
        let vds = vd - vs;
        let vbs = vb - vs;

        self.stamp_linearized_at(mna, vgs, vds, vbs);
    }

    fn ac_info_at(&self, solution: &DVector<f64>) -> AcDeviceInfo {
        let vg = node_to_index(self.node_gate)
            .map(|i| solution[i])
            .unwrap_or(0.0);
        let vd = node_to_index(self.node_drain)
            .map(|i| solution[i])
            .unwrap_or(0.0);
        let vs = node_to_index(self.node_source)
            .map(|i| solution[i])
            .unwrap_or(0.0);
        let vb = node_to_index(self.node_bulk)
            .map(|i| solution[i])
            .unwrap_or(0.0);

        let vgs = vg - vs;
        let vds = vd - vs;
        let vbs = vb - vs;

        let result = self.evaluate(vgs, vds, vbs);

        let caps = bsim4_evaluate_caps(
            &self.params,
            &self.derived,
            vgs,
            vds,
            vbs,
            result.region,
            result.vth,
            result.vdsat,
        );

        // Reuse the Bsim3Mosfet AC info variant since the small-signal
        // model is identical (gds + gm*Vgs + gmbs*Vbs + capacitances)
        AcDeviceInfo::Bsim3Mosfet {
            drain: node_to_index(self.node_drain),
            gate: node_to_index(self.node_gate),
            source: node_to_index(self.node_source),
            bulk: node_to_index(self.node_bulk),
            gds: result.gds,
            gm: result.gm,
            gmbs: result.gmbs,
            cgs: caps.cgs,
            cgd: caps.cgd,
            cgb: caps.cgb,
            cbs: caps.cbs,
            cbd: caps.cbd,
        }
    }

    fn transient_info(&self) -> TransientDeviceInfo {
        TransientDeviceInfo::None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nmos_creation() {
        let m = Bsim4Mosfet::nmos(
            "M1",
            NodeId::new(1),
            NodeId::new(2),
            NodeId::GROUND,
            NodeId::GROUND,
        );

        assert_eq!(m.name, "M1");
        assert_eq!(m.mos_type(), MosfetType::Nmos);
        assert_eq!(m.node_drain, NodeId::new(1));
        assert_eq!(m.node_gate, NodeId::new(2));
        assert_eq!(m.node_source, NodeId::GROUND);
        assert_eq!(m.node_bulk, NodeId::GROUND);
    }

    #[test]
    fn test_pmos_creation() {
        let m = Bsim4Mosfet::pmos(
            "M2",
            NodeId::new(1),
            NodeId::new(2),
            NodeId::new(3),
            NodeId::new(3),
        );

        assert_eq!(m.mos_type(), MosfetType::Pmos);
    }

    #[test]
    fn test_evaluate_saturation() {
        let m = Bsim4Mosfet::nmos(
            "M1",
            NodeId::new(1),
            NodeId::new(2),
            NodeId::GROUND,
            NodeId::GROUND,
        );

        let result = m.evaluate(1.0, 1.0, 0.0);

        assert_eq!(result.region, Bsim4Region::Saturation);
        assert!(result.ids > 0.0);
        assert!(result.gm > 0.0);
        assert!(result.gds > 0.0);
    }

    #[test]
    fn test_with_custom_params() {
        let mut params = Bsim4Params::nmos_default();
        params.vth0 = 0.5;
        params.w = 2e-6;
        params.l = 150e-9;
        params.qme1 = 0.3;

        let m = Bsim4Mosfet::with_params(
            "M1",
            NodeId::new(1),
            NodeId::new(2),
            NodeId::GROUND,
            NodeId::GROUND,
            params,
        );

        assert_eq!(m.params.vth0, 0.5);
        assert_eq!(m.params.w, 2e-6);
        assert_eq!(m.params.l, 150e-9);
        assert_eq!(m.params.qme1, 0.3);
    }

    #[test]
    fn test_stamp_linearized() {
        let m = Bsim4Mosfet::nmos(
            "M1",
            NodeId::new(1),
            NodeId::new(2),
            NodeId::GROUND,
            NodeId::GROUND,
        );

        let mut mna = MnaSystem::new(2, 0);
        m.stamp_linearized_at(&mut mna, 1.0, 1.0, 0.0);

        let matrix = mna.to_dense_matrix();
        assert!(matrix[(0, 0)].abs() > 1e-15);
        assert!(matrix[(0, 1)].abs() > 1e-15);
    }

    #[test]
    fn test_ac_info() {
        let m = Bsim4Mosfet::nmos(
            "M1",
            NodeId::new(1),
            NodeId::new(2),
            NodeId::GROUND,
            NodeId::GROUND,
        );

        let solution = DVector::from_vec(vec![1.0, 1.0]);
        let ac_info = m.ac_info_at(&solution);

        match ac_info {
            AcDeviceInfo::Bsim3Mosfet {
                drain,
                gate,
                gds,
                gm,
                gmbs,
                cgs,
                cgd,
                cgb,
                cbs,
                cbd,
                ..
            } => {
                assert_eq!(drain, Some(0));
                assert_eq!(gate, Some(1));
                assert!(gm > 0.0);
                assert!(gds > 0.0);
                assert!(gmbs >= 0.0);
                assert!(cgs >= 0.0);
                assert!(cgd >= 0.0);
                assert!(cgb >= 0.0);
                assert!(cbs >= 0.0);
                assert!(cbd >= 0.0);
            }
            _ => panic!("Expected AcDeviceInfo::Bsim3Mosfet"),
        }
    }

    #[test]
    fn test_temperature_set() {
        let mut m = Bsim4Mosfet::nmos(
            "M1",
            NodeId::new(1),
            NodeId::new(2),
            NodeId::GROUND,
            NodeId::GROUND,
        );

        assert!((m.temperature() - 300.15).abs() < 0.01);

        m.set_temperature(400.0);
        assert!((m.temperature() - 400.0).abs() < 0.01);
    }
}
