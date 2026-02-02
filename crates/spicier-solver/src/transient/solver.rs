//! Transient simulation solver functions.

use nalgebra::DVector;
use spicier_core::mna::MnaSystem;

use crate::dispatch::DispatchConfig;
use crate::error::Result;
use crate::gmres::GmresConfig;
use crate::linear::{CachedSparseLu, SPARSE_THRESHOLD, solve_dense};
use crate::operator::RealOperator;
use crate::preconditioner::{JacobiPreconditioner, RealPreconditioner};
use crate::sparse_operator::SparseRealOperator;

use super::companion::{CapacitorState, InductorState};
use super::result::{AdaptiveTransientResult, TimePoint, TransientResult};
use super::types::{AdaptiveTransientParams, IntegrationMethod, TRBDF2_GAMMA, TransientParams};

/// Callback for stamping the circuit at each transient timestep.
pub trait TransientStamper {
    /// Stamp all non-reactive (resistive + source) elements at the given time.
    ///
    /// For time-varying sources (PULSE, SIN, PWL), the source value should be
    /// evaluated at the specified time. For DC sources, time is ignored.
    fn stamp_at_time(&self, mna: &mut MnaSystem, time: f64);

    /// Get the number of nodes.
    fn num_nodes(&self) -> usize;

    /// Get the number of voltage source current variables.
    fn num_vsources(&self) -> usize;
}

/// Run a transient simulation.
///
/// # Arguments
/// * `stamper` - Stamps resistive elements and sources
/// * `caps` - Capacitor companion model states
/// * `inds` - Inductor companion model states
/// * `params` - Transient parameters
/// * `dc_solution` - Initial DC operating point
pub fn solve_transient(
    stamper: &dyn TransientStamper,
    caps: &mut [CapacitorState],
    inds: &mut [InductorState],
    params: &TransientParams,
    dc_solution: &DVector<f64>,
) -> Result<TransientResult> {
    let num_nodes = stamper.num_nodes();
    let num_vsources = stamper.num_vsources();
    let h = params.tstep;

    // Initialize reactive element states from DC solution
    for cap in caps.iter_mut() {
        let vp = cap.node_pos.map(|i| dc_solution[i]).unwrap_or(0.0);
        let vn = cap.node_neg.map(|i| dc_solution[i]).unwrap_or(0.0);
        cap.v_prev = vp - vn;
        cap.i_prev = 0.0; // No current through caps at DC
    }

    // Extract initial inductor currents from DC solution branch currents.
    // In DC, inductors are modeled as short circuits (0V voltage sources) with
    // branch current variables. The branch current index points to the position
    // in dc_solution after all node voltages.
    for ind in inds.iter_mut() {
        let vp = ind.node_pos.map(|i| dc_solution[i]).unwrap_or(0.0);
        let vn = ind.node_neg.map(|i| dc_solution[i]).unwrap_or(0.0);
        ind.v_prev = vp - vn;
        // Extract initial current from DC solution if the branch index is valid
        let branch_idx = num_nodes + ind.branch_index;
        if branch_idx < dc_solution.len() {
            ind.i_prev = dc_solution[branch_idx];
        } else {
            ind.i_prev = 0.0; // Fallback if index out of range
        }
    }

    // Create properly-sized solution for transient analysis.
    // The transient MNA excludes inductor branch currents (they use companion models).
    let mna_size = num_nodes + num_vsources;
    let mut solution = DVector::zeros(mna_size);
    // Copy node voltages
    for i in 0..num_nodes.min(dc_solution.len()) {
        solution[i] = dc_solution[i];
    }
    // Copy voltage source currents (skip inductor branch currents in DC solution)
    // This assumes voltage source currents come before inductor currents in DC solution.
    for i in 0..num_vsources {
        let dc_idx = num_nodes + i;
        if dc_idx < dc_solution.len() {
            solution[num_nodes + i] = dc_solution[dc_idx];
        }
    }

    let mut result = TransientResult {
        points: Vec::new(),
        num_nodes,
    };

    // Store initial point
    result.points.push(TimePoint {
        time: 0.0,
        solution: solution.clone(),
    });

    let num_steps = (params.tstop / h).ceil() as usize;
    let mna_size = num_nodes + num_vsources;

    // Cached sparse solver (created on first timestep if needed)
    let mut cached_solver: Option<CachedSparseLu> = None;

    for step in 1..=num_steps {
        let t = (step as f64) * h;

        // Build MNA system for this timestep
        let mut mna = MnaSystem::new(num_nodes, num_vsources);

        // Stamp static elements (resistors, sources)
        stamper.stamp_at_time(&mut mna, t);

        // Stamp companion models for reactive elements and solve
        match params.method {
            IntegrationMethod::BackwardEuler => {
                for cap in caps.iter() {
                    cap.stamp_be(&mut mna, h);
                }
                for ind in inds.iter() {
                    ind.stamp_be(&mut mna, h);
                }

                // Solve
                solution = if mna_size >= SPARSE_THRESHOLD {
                    let solver = match &cached_solver {
                        Some(s) => s,
                        None => {
                            cached_solver = Some(CachedSparseLu::new(mna_size, &mna.triplets)?);
                            cached_solver.as_ref().unwrap()
                        }
                    };
                    solver.solve(&mna.triplets, mna.rhs())?
                } else {
                    solve_dense(&mna.to_dense_matrix(), mna.rhs())?
                };

                // Update state
                for cap in caps.iter_mut() {
                    let v = cap.voltage_from_solution(&solution);
                    cap.update(v, h, params.method);
                }
                for ind in inds.iter_mut() {
                    let v = ind.voltage_from_solution(&solution);
                    ind.update(v, h, params.method);
                }
            }
            IntegrationMethod::Trapezoidal => {
                for cap in caps.iter() {
                    cap.stamp_trap(&mut mna, h);
                }
                for ind in inds.iter() {
                    ind.stamp_trap(&mut mna, h);
                }

                // Solve
                solution = if mna_size >= SPARSE_THRESHOLD {
                    let solver = match &cached_solver {
                        Some(s) => s,
                        None => {
                            cached_solver = Some(CachedSparseLu::new(mna_size, &mna.triplets)?);
                            cached_solver.as_ref().unwrap()
                        }
                    };
                    solver.solve(&mna.triplets, mna.rhs())?
                } else {
                    solve_dense(&mna.to_dense_matrix(), mna.rhs())?
                };

                // Update state
                for cap in caps.iter_mut() {
                    let v = cap.voltage_from_solution(&solution);
                    cap.update(v, h, params.method);
                }
                for ind in inds.iter_mut() {
                    let v = ind.voltage_from_solution(&solution);
                    ind.update(v, h, params.method);
                }
            }
            IntegrationMethod::TrBdf2 => {
                // TR-BDF2: Two-stage method
                // Stage 1: Trapezoidal step for γ*h
                let h_gamma = TRBDF2_GAMMA * h;
                for cap in caps.iter() {
                    cap.stamp_trap(&mut mna, h_gamma);
                }
                for ind in inds.iter() {
                    ind.stamp_trap(&mut mna, h_gamma);
                }

                // Solve stage 1
                let solution_gamma = if mna_size >= SPARSE_THRESHOLD {
                    let solver = match &cached_solver {
                        Some(s) => s,
                        None => {
                            cached_solver = Some(CachedSparseLu::new(mna_size, &mna.triplets)?);
                            cached_solver.as_ref().unwrap()
                        }
                    };
                    solver.solve(&mna.triplets, mna.rhs())?
                } else {
                    solve_dense(&mna.to_dense_matrix(), mna.rhs())?
                };

                // Update state to intermediate point
                for cap in caps.iter_mut() {
                    let v = cap.voltage_from_solution(&solution_gamma);
                    cap.update_trbdf2_intermediate(v, h);
                }
                for ind in inds.iter_mut() {
                    let v = ind.voltage_from_solution(&solution_gamma);
                    ind.update_trbdf2_intermediate(v, h);
                    ind.v_prev = v; // Update v_prev for BDF2 stage
                }

                // Stage 2: BDF2 step for (1-γ)*h
                let mut mna2 = MnaSystem::new(num_nodes, num_vsources);
                stamper.stamp_at_time(&mut mna2, t);
                for cap in caps.iter() {
                    cap.stamp_trbdf2_bdf2(&mut mna2, h);
                }
                for ind in inds.iter() {
                    ind.stamp_trbdf2_bdf2(&mut mna2, h);
                }

                // Solve stage 2
                solution = if mna_size >= SPARSE_THRESHOLD {
                    cached_solver
                        .as_ref()
                        .unwrap()
                        .solve(&mna2.triplets, mna2.rhs())?
                } else {
                    solve_dense(&mna2.to_dense_matrix(), mna2.rhs())?
                };

                // Final state update
                for cap in caps.iter_mut() {
                    let v = cap.voltage_from_solution(&solution);
                    cap.update(v, h, params.method);
                }
                for ind in inds.iter_mut() {
                    let v = ind.voltage_from_solution(&solution);
                    ind.update(v, h, params.method);
                }
            }
        }

        result.points.push(TimePoint {
            time: t,
            solution: solution.clone(),
        });
    }

    Ok(result)
}

/// Run transient simulation with configurable dispatch.
///
/// This variant allows specifying the compute backend and solver strategy.
/// For large systems, can use GMRES instead of direct LU.
///
/// # Arguments
/// * `stamper` - Stamps resistive elements and sources
/// * `caps` - Capacitor companion model states
/// * `inds` - Inductor companion model states
/// * `params` - Transient parameters
/// * `dc_solution` - Initial DC operating point
/// * `config` - Dispatch configuration
pub fn solve_transient_dispatched(
    stamper: &dyn TransientStamper,
    caps: &mut [CapacitorState],
    inds: &mut [InductorState],
    params: &TransientParams,
    dc_solution: &DVector<f64>,
    config: &DispatchConfig,
) -> Result<TransientResult> {
    let num_nodes = stamper.num_nodes();
    let num_vsources = stamper.num_vsources();
    let h = params.tstep;
    let mna_size = num_nodes + num_vsources;

    // Decide solver strategy based on size
    let use_gmres = config.use_gmres(mna_size);

    // Initialize reactive element states from DC solution
    for cap in caps.iter_mut() {
        let vp = cap.node_pos.map(|i| dc_solution[i]).unwrap_or(0.0);
        let vn = cap.node_neg.map(|i| dc_solution[i]).unwrap_or(0.0);
        cap.v_prev = vp - vn;
        cap.i_prev = 0.0;
    }

    // Extract initial inductor currents from DC solution
    for ind in inds.iter_mut() {
        let vp = ind.node_pos.map(|i| dc_solution[i]).unwrap_or(0.0);
        let vn = ind.node_neg.map(|i| dc_solution[i]).unwrap_or(0.0);
        ind.v_prev = vp - vn;
        let branch_idx = num_nodes + ind.branch_index;
        if branch_idx < dc_solution.len() {
            ind.i_prev = dc_solution[branch_idx];
        } else {
            ind.i_prev = 0.0;
        }
    }

    // Create properly-sized solution for transient analysis
    let mut solution = DVector::zeros(mna_size);
    for i in 0..num_nodes.min(dc_solution.len()) {
        solution[i] = dc_solution[i];
    }
    for i in 0..num_vsources {
        let dc_idx = num_nodes + i;
        if dc_idx < dc_solution.len() {
            solution[num_nodes + i] = dc_solution[dc_idx];
        }
    }

    let mut result = TransientResult {
        points: Vec::new(),
        num_nodes,
    };

    result.points.push(TimePoint {
        time: 0.0,
        solution: solution.clone(),
    });

    let num_steps = (params.tstop / h).ceil() as usize;

    // Cached sparse solver for direct LU
    let mut cached_solver: Option<CachedSparseLu> = None;

    for step in 1..=num_steps {
        let t = (step as f64) * h;

        let mut mna = MnaSystem::new(num_nodes, num_vsources);
        stamper.stamp_at_time(&mut mna, t);

        // Helper closure for solving
        let solve_mna =
            |mna: &MnaSystem, cached: &mut Option<CachedSparseLu>| -> Result<DVector<f64>> {
                if use_gmres {
                    solve_transient_gmres(mna, &config.gmres_config)
                } else if mna_size >= SPARSE_THRESHOLD {
                    let solver = match cached.as_ref() {
                        Some(s) => s,
                        None => {
                            *cached = Some(CachedSparseLu::new(mna_size, &mna.triplets)?);
                            cached.as_ref().unwrap()
                        }
                    };
                    solver.solve(&mna.triplets, mna.rhs())
                } else {
                    solve_dense(&mna.to_dense_matrix(), mna.rhs())
                }
            };

        match params.method {
            IntegrationMethod::BackwardEuler => {
                for cap in caps.iter() {
                    cap.stamp_be(&mut mna, h);
                }
                for ind in inds.iter() {
                    ind.stamp_be(&mut mna, h);
                }
                solution = solve_mna(&mna, &mut cached_solver)?;
                for cap in caps.iter_mut() {
                    let v = cap.voltage_from_solution(&solution);
                    cap.update(v, h, params.method);
                }
                for ind in inds.iter_mut() {
                    let v = ind.voltage_from_solution(&solution);
                    ind.update(v, h, params.method);
                }
            }
            IntegrationMethod::Trapezoidal => {
                for cap in caps.iter() {
                    cap.stamp_trap(&mut mna, h);
                }
                for ind in inds.iter() {
                    ind.stamp_trap(&mut mna, h);
                }
                solution = solve_mna(&mna, &mut cached_solver)?;
                for cap in caps.iter_mut() {
                    let v = cap.voltage_from_solution(&solution);
                    cap.update(v, h, params.method);
                }
                for ind in inds.iter_mut() {
                    let v = ind.voltage_from_solution(&solution);
                    ind.update(v, h, params.method);
                }
            }
            IntegrationMethod::TrBdf2 => {
                // Stage 1: Trapezoidal for γ*h
                let h_gamma = TRBDF2_GAMMA * h;
                for cap in caps.iter() {
                    cap.stamp_trap(&mut mna, h_gamma);
                }
                for ind in inds.iter() {
                    ind.stamp_trap(&mut mna, h_gamma);
                }
                let solution_gamma = solve_mna(&mna, &mut cached_solver)?;

                // Update to intermediate state
                for cap in caps.iter_mut() {
                    let v = cap.voltage_from_solution(&solution_gamma);
                    cap.update_trbdf2_intermediate(v, h);
                }
                for ind in inds.iter_mut() {
                    let v = ind.voltage_from_solution(&solution_gamma);
                    ind.update_trbdf2_intermediate(v, h);
                    ind.v_prev = v;
                }

                // Stage 2: BDF2 for (1-γ)*h
                let mut mna2 = MnaSystem::new(num_nodes, num_vsources);
                stamper.stamp_at_time(&mut mna2, t);
                for cap in caps.iter() {
                    cap.stamp_trbdf2_bdf2(&mut mna2, h);
                }
                for ind in inds.iter() {
                    ind.stamp_trbdf2_bdf2(&mut mna2, h);
                }
                solution = solve_mna(&mna2, &mut cached_solver)?;

                // Final state update
                for cap in caps.iter_mut() {
                    let v = cap.voltage_from_solution(&solution);
                    cap.update(v, h, params.method);
                }
                for ind in inds.iter_mut() {
                    let v = ind.voltage_from_solution(&solution);
                    ind.update(v, h, params.method);
                }
            }
        }

        result.points.push(TimePoint {
            time: t,
            solution: solution.clone(),
        });
    }

    Ok(result)
}

/// Solve a transient timestep using GMRES.
fn solve_transient_gmres(mna: &MnaSystem, config: &GmresConfig) -> Result<DVector<f64>> {
    let size = mna.size();

    let operator = SparseRealOperator::from_triplets(size, &mna.triplets).ok_or_else(|| {
        crate::error::Error::SolverError("Failed to build sparse operator".into())
    })?;

    let preconditioner = JacobiPreconditioner::from_triplets(size, &mna.triplets);
    let rhs: Vec<f64> = mna.rhs().iter().copied().collect();

    let gmres_result = crate::gmres::solve_gmres_real_preconditioned(
        &operator as &dyn RealOperator,
        &preconditioner as &dyn RealPreconditioner,
        &rhs,
        config,
    );

    if !gmres_result.converged {
        log::warn!(
            "Transient GMRES did not converge after {} iterations (residual: {:.2e})",
            gmres_result.iterations,
            gmres_result.residual
        );
    }

    Ok(DVector::from_vec(gmres_result.x))
}

/// Run adaptive transient simulation with automatic timestep control.
///
/// Uses Local Truncation Error (LTE) estimation to automatically adjust
/// the timestep. Larger steps are taken when the solution is smooth,
/// smaller steps when it changes rapidly.
///
/// # Arguments
/// * `stamper` - Stamps resistive elements and sources
/// * `caps` - Capacitor companion model states
/// * `inds` - Inductor companion model states
/// * `params` - Adaptive transient parameters
/// * `dc_solution` - Initial DC operating point
pub fn solve_transient_adaptive(
    stamper: &dyn TransientStamper,
    caps: &mut [CapacitorState],
    inds: &mut [InductorState],
    params: &AdaptiveTransientParams,
    dc_solution: &DVector<f64>,
) -> Result<AdaptiveTransientResult> {
    let num_nodes = stamper.num_nodes();
    let num_vsources = stamper.num_vsources();
    let mna_size = num_nodes + num_vsources;

    let mut t = 0.0;
    let mut h = params.h_init;

    // Initialize reactive element states from DC solution
    for cap in caps.iter_mut() {
        let vp = cap.node_pos.map(|i| dc_solution[i]).unwrap_or(0.0);
        let vn = cap.node_neg.map(|i| dc_solution[i]).unwrap_or(0.0);
        cap.v_prev = vp - vn;
        cap.i_prev = 0.0;
    }

    // Extract initial inductor currents from DC solution
    for ind in inds.iter_mut() {
        let vp = ind.node_pos.map(|i| dc_solution[i]).unwrap_or(0.0);
        let vn = ind.node_neg.map(|i| dc_solution[i]).unwrap_or(0.0);
        ind.v_prev = vp - vn;
        let branch_idx = num_nodes + ind.branch_index;
        if branch_idx < dc_solution.len() {
            ind.i_prev = dc_solution[branch_idx];
        } else {
            ind.i_prev = 0.0;
        }
    }

    // Create properly-sized solution for transient analysis
    let mut solution = DVector::zeros(mna_size);
    for i in 0..num_nodes.min(dc_solution.len()) {
        solution[i] = dc_solution[i];
    }
    for i in 0..num_vsources {
        let dc_idx = num_nodes + i;
        if dc_idx < dc_solution.len() {
            solution[num_nodes + i] = dc_solution[dc_idx];
        }
    }

    let mut result = AdaptiveTransientResult {
        points: Vec::new(),
        num_nodes,
        total_steps: 0,
        rejected_steps: 0,
        min_step_used: f64::INFINITY,
        max_step_used: 0.0,
    };

    // Store initial point
    result.points.push(TimePoint {
        time: 0.0,
        solution: solution.clone(),
    });

    // Cached sparse solver
    let mut cached_solver: Option<CachedSparseLu> = None;

    // Save states for potential rollback
    let mut saved_cap_states: Vec<(f64, f64)> = caps.iter().map(|c| (c.v_prev, c.i_prev)).collect();
    let mut saved_ind_states: Vec<(f64, f64)> = inds.iter().map(|i| (i.i_prev, i.v_prev)).collect();

    while t < params.tstop {
        // Clamp timestep
        h = h.clamp(params.h_min, params.h_max);

        // Don't overshoot tstop
        if t + h > params.tstop {
            h = params.tstop - t;
        }

        // Build MNA system for this timestep
        let mut mna = MnaSystem::new(num_nodes, num_vsources);
        stamper.stamp_at_time(&mut mna, t);

        // Stamp companion models (using Trapezoidal for better accuracy)
        for cap in caps.iter() {
            cap.stamp_trap(&mut mna, h);
        }
        for ind in inds.iter() {
            ind.stamp_trap(&mut mna, h);
        }

        // Solve
        let new_solution = if mna_size >= SPARSE_THRESHOLD {
            let solver = match &cached_solver {
                Some(s) => s,
                None => {
                    cached_solver = Some(CachedSparseLu::new(mna_size, &mna.triplets)?);
                    cached_solver.as_ref().unwrap()
                }
            };
            solver.solve(&mna.triplets, mna.rhs())?
        } else {
            solve_dense(&mna.to_dense_matrix(), mna.rhs())?
        };

        // Estimate LTE for all reactive elements
        let mut max_lte = 0.0_f64;
        let mut max_ref = 0.0_f64; // Reference value for relative error

        for cap in caps.iter() {
            let v_new = cap.voltage_from_solution(&new_solution);
            let lte = cap.estimate_lte(v_new, h);
            max_lte = max_lte.max(lte);
            max_ref = max_ref.max(v_new.abs());
        }

        for ind in inds.iter() {
            let v_new = ind.voltage_from_solution(&new_solution);
            let lte = ind.estimate_lte(v_new, h);
            max_lte = max_lte.max(lte);
            max_ref = max_ref.max(ind.i_prev.abs());
        }

        // Compute tolerance: max(abstol, reltol * max_ref)
        let tol = params.abstol.max(params.reltol * max_ref);

        result.total_steps += 1;

        if max_lte > tol && h > params.h_min {
            // Reject step: LTE too large
            result.rejected_steps += 1;

            // Restore previous states
            for (cap, (v, i)) in caps.iter_mut().zip(saved_cap_states.iter()) {
                cap.v_prev = *v;
                cap.i_prev = *i;
            }
            for (ind, (i, v)) in inds.iter_mut().zip(saved_ind_states.iter()) {
                ind.i_prev = *i;
                ind.v_prev = *v;
            }

            // Reduce timestep (safety factor of 0.8)
            let factor = (tol / max_lte).sqrt().min(0.5);
            h *= factor.max(0.1); // Don't reduce by more than 10x
        } else {
            // Accept step
            t += h;
            solution = new_solution;

            // Update reactive element states
            for cap in caps.iter_mut() {
                let v_new = cap.voltage_from_solution(&solution);
                cap.update(v_new, h, IntegrationMethod::Trapezoidal);
            }
            for ind in inds.iter_mut() {
                let v_new = ind.voltage_from_solution(&solution);
                ind.update(v_new, h, IntegrationMethod::Trapezoidal);
            }

            // Save states for potential rollback
            saved_cap_states = caps.iter().map(|c| (c.v_prev, c.i_prev)).collect();
            saved_ind_states = inds.iter().map(|i| (i.i_prev, i.v_prev)).collect();

            // Track step size statistics
            result.min_step_used = result.min_step_used.min(h);
            result.max_step_used = result.max_step_used.max(h);

            // Store result
            result.points.push(TimePoint {
                time: t,
                solution: solution.clone(),
            });

            // Increase timestep for next step if LTE is small
            if max_lte < tol * 0.5 && h < params.h_max {
                let factor = (tol / max_lte.max(1e-20)).sqrt().min(2.0);
                h *= factor.min(1.5); // Don't increase by more than 1.5x
            }
        }
    }

    Ok(result)
}
