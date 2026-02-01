//! Controlled source device models: VCVS (E), VCCS (G), CCCS (F), CCVS (H).

use spicier_core::mna::MnaSystem;
use spicier_core::netlist::AcDeviceInfo;
use spicier_core::{Element, NodeId, Stamper};

use crate::stamp::Stamp;

/// Convert a NodeId to an MNA matrix index (None for ground).
fn node_to_index(node: NodeId) -> Option<usize> {
    if node.is_ground() {
        None
    } else {
        Some((node.as_u32() - 1) as usize)
    }
}

// ────────────────────── VCVS (E element) ──────────────────────

/// Voltage-controlled voltage source.
///
/// V(out+, out-) = gain * V(ctrl+, ctrl-)
///
/// Requires one branch current variable (like a voltage source).
#[derive(Debug, Clone)]
pub struct Vcvs {
    pub name: String,
    pub out_pos: NodeId,
    pub out_neg: NodeId,
    pub ctrl_pos: NodeId,
    pub ctrl_neg: NodeId,
    pub gain: f64,
    pub current_index: usize,
}

impl Vcvs {
    pub fn new(
        name: impl Into<String>,
        out_pos: NodeId,
        out_neg: NodeId,
        ctrl_pos: NodeId,
        ctrl_neg: NodeId,
        gain: f64,
        current_index: usize,
    ) -> Self {
        Self {
            name: name.into(),
            out_pos,
            out_neg,
            ctrl_pos,
            ctrl_neg,
            gain,
            current_index,
        }
    }
}

impl Stamp for Vcvs {
    fn stamp(&self, mna: &mut MnaSystem) {
        let op = node_to_index(self.out_pos);
        let on = node_to_index(self.out_neg);
        let cp = node_to_index(self.ctrl_pos);
        let cn = node_to_index(self.ctrl_neg);
        let br = mna.num_nodes + self.current_index;

        // Branch current couples to output nodes (like a voltage source):
        // KCL at out+: ... + I_branch = 0
        // KCL at out-: ... - I_branch = 0
        if let Some(i) = op {
            mna.matrix_mut()[(i, br)] += 1.0;
        }
        if let Some(i) = on {
            mna.matrix_mut()[(i, br)] -= 1.0;
        }

        // Branch equation: V(out+) - V(out-) - gain * (V(ctrl+) - V(ctrl-)) = 0
        if let Some(i) = op {
            mna.matrix_mut()[(br, i)] += 1.0;
        }
        if let Some(i) = on {
            mna.matrix_mut()[(br, i)] -= 1.0;
        }
        if let Some(i) = cp {
            mna.matrix_mut()[(br, i)] -= self.gain;
        }
        if let Some(i) = cn {
            mna.matrix_mut()[(br, i)] += self.gain;
        }
    }
}

impl Element for Vcvs {
    fn name(&self) -> &str {
        &self.name
    }

    fn nodes(&self) -> Vec<NodeId> {
        vec![self.out_pos, self.out_neg, self.ctrl_pos, self.ctrl_neg]
    }

    fn num_current_vars(&self) -> usize {
        1
    }
}

impl Stamper for Vcvs {
    fn stamp(&self, mna: &mut MnaSystem) {
        Stamp::stamp(self, mna);
    }

    fn num_current_vars(&self) -> usize {
        1
    }

    fn device_name(&self) -> &str {
        &self.name
    }

    fn branch_index(&self) -> Option<usize> {
        Some(self.current_index)
    }

    fn ac_info(&self) -> AcDeviceInfo {
        AcDeviceInfo::Vcvs {
            out_pos: node_to_index(self.out_pos),
            out_neg: node_to_index(self.out_neg),
            ctrl_pos: node_to_index(self.ctrl_pos),
            ctrl_neg: node_to_index(self.ctrl_neg),
            branch_idx: self.current_index,
            gain: self.gain,
        }
    }
}

// ────────────────────── VCCS (G element) ──────────────────────

/// Voltage-controlled current source.
///
/// I(out+→out-) = gm * V(ctrl+, ctrl-)
///
/// No branch current variable needed.
#[derive(Debug, Clone)]
pub struct Vccs {
    pub name: String,
    pub out_pos: NodeId,
    pub out_neg: NodeId,
    pub ctrl_pos: NodeId,
    pub ctrl_neg: NodeId,
    pub gm: f64,
}

impl Vccs {
    pub fn new(
        name: impl Into<String>,
        out_pos: NodeId,
        out_neg: NodeId,
        ctrl_pos: NodeId,
        ctrl_neg: NodeId,
        gm: f64,
    ) -> Self {
        Self {
            name: name.into(),
            out_pos,
            out_neg,
            ctrl_pos,
            ctrl_neg,
            gm,
        }
    }
}

impl Stamp for Vccs {
    fn stamp(&self, mna: &mut MnaSystem) {
        let op = node_to_index(self.out_pos);
        let on = node_to_index(self.out_neg);
        let cp = node_to_index(self.ctrl_pos);
        let cn = node_to_index(self.ctrl_neg);

        // SPICE convention: I = gm * V(ctrl+, ctrl-) enters out_pos, leaves out_neg.
        // In MNA (matrix * x = rhs), current entering a node goes to the RHS,
        // so on the LHS (matrix side) the sign is negative.
        // KCL at out+: -gm * V(ctrl+) + gm * V(ctrl-) + ... = rhs
        // KCL at out-: +gm * V(ctrl+) - gm * V(ctrl-) + ... = rhs
        if let Some(i) = op {
            if let Some(j) = cp {
                mna.matrix_mut()[(i, j)] -= self.gm;
            }
            if let Some(j) = cn {
                mna.matrix_mut()[(i, j)] += self.gm;
            }
        }
        if let Some(i) = on {
            if let Some(j) = cp {
                mna.matrix_mut()[(i, j)] += self.gm;
            }
            if let Some(j) = cn {
                mna.matrix_mut()[(i, j)] -= self.gm;
            }
        }
    }
}

impl Element for Vccs {
    fn name(&self) -> &str {
        &self.name
    }

    fn nodes(&self) -> Vec<NodeId> {
        vec![self.out_pos, self.out_neg, self.ctrl_pos, self.ctrl_neg]
    }
}

impl Stamper for Vccs {
    fn stamp(&self, mna: &mut MnaSystem) {
        Stamp::stamp(self, mna);
    }

    fn device_name(&self) -> &str {
        &self.name
    }

    fn ac_info(&self) -> AcDeviceInfo {
        AcDeviceInfo::Vccs {
            out_pos: node_to_index(self.out_pos),
            out_neg: node_to_index(self.out_neg),
            ctrl_pos: node_to_index(self.ctrl_pos),
            ctrl_neg: node_to_index(self.ctrl_neg),
            gm: self.gm,
        }
    }
}

// ────────────────────── CCCS (F element) ──────────────────────

/// Current-controlled current source.
///
/// I(out+→out-) = gain * I(Vsource)
///
/// References the branch current variable of a voltage source.
#[derive(Debug, Clone)]
pub struct Cccs {
    pub name: String,
    pub out_pos: NodeId,
    pub out_neg: NodeId,
    pub vsource_branch_idx: usize,
    pub gain: f64,
}

impl Cccs {
    pub fn new(
        name: impl Into<String>,
        out_pos: NodeId,
        out_neg: NodeId,
        vsource_branch_idx: usize,
        gain: f64,
    ) -> Self {
        Self {
            name: name.into(),
            out_pos,
            out_neg,
            vsource_branch_idx,
            gain,
        }
    }
}

impl Stamp for Cccs {
    fn stamp(&self, mna: &mut MnaSystem) {
        let op = node_to_index(self.out_pos);
        let on = node_to_index(self.out_neg);
        let br = mna.num_nodes + self.vsource_branch_idx;

        // I(out) = gain * I(Vsource)
        // KCL at out+: ... + gain * I_vsource
        // KCL at out-: ... - gain * I_vsource
        if let Some(i) = op {
            mna.matrix_mut()[(i, br)] += self.gain;
        }
        if let Some(i) = on {
            mna.matrix_mut()[(i, br)] -= self.gain;
        }
    }
}

impl Element for Cccs {
    fn name(&self) -> &str {
        &self.name
    }

    fn nodes(&self) -> Vec<NodeId> {
        vec![self.out_pos, self.out_neg]
    }
}

impl Stamper for Cccs {
    fn stamp(&self, mna: &mut MnaSystem) {
        Stamp::stamp(self, mna);
    }

    fn device_name(&self) -> &str {
        &self.name
    }

    fn ac_info(&self) -> AcDeviceInfo {
        AcDeviceInfo::Cccs {
            out_pos: node_to_index(self.out_pos),
            out_neg: node_to_index(self.out_neg),
            vsource_branch_idx: self.vsource_branch_idx,
            gain: self.gain,
        }
    }
}

// ────────────────────── CCVS (H element) ──────────────────────

/// Current-controlled voltage source.
///
/// V(out+, out-) = gain * I(Vsource)
///
/// Requires one branch current variable (like a voltage source).
#[derive(Debug, Clone)]
pub struct Ccvs {
    pub name: String,
    pub out_pos: NodeId,
    pub out_neg: NodeId,
    pub vsource_branch_idx: usize,
    pub gain: f64,
    pub current_index: usize,
}

impl Ccvs {
    pub fn new(
        name: impl Into<String>,
        out_pos: NodeId,
        out_neg: NodeId,
        vsource_branch_idx: usize,
        gain: f64,
        current_index: usize,
    ) -> Self {
        Self {
            name: name.into(),
            out_pos,
            out_neg,
            vsource_branch_idx,
            gain,
            current_index,
        }
    }
}

impl Stamp for Ccvs {
    fn stamp(&self, mna: &mut MnaSystem) {
        let op = node_to_index(self.out_pos);
        let on = node_to_index(self.out_neg);
        let br = mna.num_nodes + self.current_index;
        let ctrl_br = mna.num_nodes + self.vsource_branch_idx;

        // Branch current couples to output nodes:
        if let Some(i) = op {
            mna.matrix_mut()[(i, br)] += 1.0;
        }
        if let Some(i) = on {
            mna.matrix_mut()[(i, br)] -= 1.0;
        }

        // Branch equation: V(out+) - V(out-) - gain * I(Vsource) = 0
        if let Some(i) = op {
            mna.matrix_mut()[(br, i)] += 1.0;
        }
        if let Some(i) = on {
            mna.matrix_mut()[(br, i)] -= 1.0;
        }
        mna.matrix_mut()[(br, ctrl_br)] -= self.gain;
    }
}

impl Element for Ccvs {
    fn name(&self) -> &str {
        &self.name
    }

    fn nodes(&self) -> Vec<NodeId> {
        vec![self.out_pos, self.out_neg]
    }

    fn num_current_vars(&self) -> usize {
        1
    }
}

impl Stamper for Ccvs {
    fn stamp(&self, mna: &mut MnaSystem) {
        Stamp::stamp(self, mna);
    }

    fn num_current_vars(&self) -> usize {
        1
    }

    fn device_name(&self) -> &str {
        &self.name
    }

    fn branch_index(&self) -> Option<usize> {
        Some(self.current_index)
    }

    fn ac_info(&self) -> AcDeviceInfo {
        AcDeviceInfo::Ccvs {
            out_pos: node_to_index(self.out_pos),
            out_neg: node_to_index(self.out_neg),
            vsource_branch_idx: self.vsource_branch_idx,
            branch_idx: self.current_index,
            gain: self.gain,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vcvs_stamp() {
        // E1: out=(1,0), ctrl=(2,0), gain=2.0, branch idx=0
        let mut mna = MnaSystem::new(2, 1);
        let e = Vcvs::new("E1", NodeId::new(1), NodeId::GROUND, NodeId::new(2), NodeId::GROUND, 2.0, 0);
        Stamp::stamp(&e, &mut mna);

        // Branch current coupling: matrix[0, 2] = +1 (out_pos)
        assert_eq!(mna.matrix()[(0, 2)], 1.0);
        // Branch equation: matrix[2, 0] = +1 (out_pos), matrix[2, 1] = -2 (ctrl_pos * -gain)
        assert_eq!(mna.matrix()[(2, 0)], 1.0);
        assert_eq!(mna.matrix()[(2, 1)], -2.0);
    }

    #[test]
    fn test_vccs_stamp() {
        // G1: out=(1,0), ctrl=(2,0), gm=0.001
        // Current gm * V(ctrl) enters out_pos (node 1)
        let mut mna = MnaSystem::new(2, 0);
        let g = Vccs::new("G1", NodeId::new(1), NodeId::GROUND, NodeId::new(2), NodeId::GROUND, 0.001);
        Stamp::stamp(&g, &mut mna);

        // matrix[0, 1] -= gm (current enters out_pos → negative on LHS)
        assert!((mna.matrix()[(0, 1)] - (-0.001)).abs() < 1e-15);
    }

    #[test]
    fn test_cccs_stamp() {
        // F1: out=(2,0), references Vsource branch 0, gain=3.0
        // V1 at branch 0, 2 nodes, 1 vsource
        let mut mna = MnaSystem::new(2, 1);
        let f = Cccs::new("F1", NodeId::new(2), NodeId::GROUND, 0, 3.0);
        Stamp::stamp(&f, &mut mna);

        // matrix[1, 2] += gain (out_pos row, vsource branch col)
        assert_eq!(mna.matrix()[(1, 2)], 3.0);
    }

    #[test]
    fn test_ccvs_stamp() {
        // H1: out=(2,0), references Vsource branch 0, gain=100, own branch idx=1
        let mut mna = MnaSystem::new(2, 2);
        let h = Ccvs::new("H1", NodeId::new(2), NodeId::GROUND, 0, 100.0, 1);
        Stamp::stamp(&h, &mut mna);

        // Branch current coupling: matrix[1, 3] = +1 (out_pos, own branch)
        assert_eq!(mna.matrix()[(1, 3)], 1.0);
        // Branch equation: matrix[3, 1] = +1, matrix[3, 2] = -100 (ctrl branch)
        assert_eq!(mna.matrix()[(3, 1)], 1.0);
        assert_eq!(mna.matrix()[(3, 2)], -100.0);
    }
}
