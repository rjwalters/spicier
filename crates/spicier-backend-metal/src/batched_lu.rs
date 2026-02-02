//! GPU-accelerated batched LU factorization and solve using wgpu/Metal.
//!
//! Each matrix in the batch is processed by a separate workgroup, enabling
//! massive parallelism for Monte Carlo, corner analysis, and parameter sweeps.

use crate::context::WgpuContext;
use crate::error::{Result, WgpuError};
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// Maximum matrix dimension supported (limited by workgroup shared memory).
pub const MAX_MATRIX_SIZE: usize = 128;

/// Minimum batch size for GPU to be worthwhile.
/// Note: Current implementation has high overhead, so this is set high.
pub const MIN_BATCH_SIZE: usize = 2000;

/// Minimum matrix size for GPU to be worthwhile.
/// Note: Current implementation needs large matrices to amortize overhead.
pub const MIN_MATRIX_SIZE: usize = 100;

/// Uniform buffer layout for shader parameters.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct Uniforms {
    n: u32,
    batch_size: u32,
}

/// Result of a batched LU solve operation.
#[derive(Debug, Clone)]
pub struct BatchedSolveResult {
    /// Solutions for each system (flattened: batch_size * n elements).
    pub solutions: Vec<f64>,
    /// Indices of matrices that were singular.
    pub singular_indices: Vec<usize>,
    /// Matrix dimension.
    pub n: usize,
    /// Number of systems solved.
    pub batch_size: usize,
}

impl BatchedSolveResult {
    /// Get the solution for a specific system.
    pub fn solution(&self, index: usize) -> Option<&[f64]> {
        if index >= self.batch_size {
            return None;
        }
        let start = index * self.n;
        let end = start + self.n;
        Some(&self.solutions[start..end])
    }

    /// Check if a specific system was singular.
    pub fn is_singular(&self, index: usize) -> bool {
        self.singular_indices.contains(&index)
    }

    /// Number of successfully solved systems.
    pub fn num_solved(&self) -> usize {
        self.batch_size - self.singular_indices.len()
    }
}

/// Configuration for GPU batched operations.
#[derive(Debug, Clone)]
pub struct GpuBatchConfig {
    /// Minimum batch size to use GPU.
    pub min_batch_size: usize,
    /// Minimum matrix dimension to use GPU.
    pub min_matrix_size: usize,
    /// Maximum matrix size (limited by shader).
    pub max_matrix_size: usize,
}

impl Default for GpuBatchConfig {
    fn default() -> Self {
        Self {
            min_batch_size: MIN_BATCH_SIZE,
            min_matrix_size: MIN_MATRIX_SIZE,
            max_matrix_size: MAX_MATRIX_SIZE,
        }
    }
}

impl GpuBatchConfig {
    /// Check if GPU should be used for the given problem size.
    pub fn should_use_gpu(&self, matrix_size: usize, batch_size: usize) -> bool {
        matrix_size >= self.min_matrix_size
            && matrix_size <= self.max_matrix_size
            && batch_size >= self.min_batch_size
    }
}

/// GPU-accelerated batched LU solver using wgpu/Metal compute shaders.
pub struct MetalBatchedLuSolver {
    ctx: Arc<WgpuContext>,
    config: GpuBatchConfig,
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl MetalBatchedLuSolver {
    const SHADER_SOURCE: &'static str = include_str!("batched_lu.wgsl");

    /// Create a new batched LU solver.
    pub fn new(ctx: Arc<WgpuContext>) -> Result<Self> {
        Self::with_config(ctx, GpuBatchConfig::default())
    }

    /// Create with custom configuration.
    pub fn with_config(ctx: Arc<WgpuContext>, config: GpuBatchConfig) -> Result<Self> {
        let device = &ctx.device;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Batched LU Shader"),
            source: wgpu::ShaderSource::Wgsl(Self::SHADER_SOURCE.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Batched LU Bind Group Layout"),
            entries: &[
                // Uniforms
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Matrices (read-write, modified in place during factorization)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // RHS/Solution vectors (read-write)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // Info array (singularity flags)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Batched LU Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Batched LU Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        log::info!(
            "Created Metal batched LU solver (GPU: {})",
            ctx.adapter_name()
        );

        Ok(Self {
            ctx,
            config,
            pipeline,
            bind_group_layout,
        })
    }

    /// Get the configuration.
    pub fn config(&self) -> &GpuBatchConfig {
        &self.config
    }

    /// Check if GPU should be used for the given problem size.
    pub fn should_use_gpu(&self, matrix_size: usize, batch_size: usize) -> bool {
        self.config.should_use_gpu(matrix_size, batch_size)
    }

    /// Solve a batch of linear systems Ax = b.
    ///
    /// # Arguments
    /// * `matrices` - Flattened matrices in column-major order (batch_size * n * n)
    /// * `rhs` - Flattened RHS vectors (batch_size * n)
    /// * `n` - Matrix/vector dimension
    /// * `batch_size` - Number of systems to solve
    ///
    /// # Returns
    /// Solutions and information about any singular systems.
    pub fn solve_batch(
        &self,
        matrices: &[f64],
        rhs: &[f64],
        n: usize,
        batch_size: usize,
    ) -> Result<BatchedSolveResult> {
        let expected_matrix_len = batch_size * n * n;
        let expected_rhs_len = batch_size * n;

        if matrices.len() != expected_matrix_len {
            return Err(WgpuError::InvalidDimension(format!(
                "Expected {} matrix elements, got {}",
                expected_matrix_len,
                matrices.len()
            )));
        }

        if rhs.len() != expected_rhs_len {
            return Err(WgpuError::InvalidDimension(format!(
                "Expected {} RHS elements, got {}",
                expected_rhs_len,
                rhs.len()
            )));
        }

        if n > self.config.max_matrix_size {
            return Err(WgpuError::InvalidDimension(format!(
                "Matrix size {} exceeds maximum {}",
                n, self.config.max_matrix_size
            )));
        }

        if batch_size == 0 {
            return Ok(BatchedSolveResult {
                solutions: vec![],
                singular_indices: vec![],
                n,
                batch_size: 0,
            });
        }

        let device = &self.ctx.device;
        let queue = &self.ctx.queue;

        // Convert f64 -> f32 for GPU (most GPUs don't support f64)
        // The shader expects row-major, but input is column-major, so we need to transpose
        let mut matrices_f32 = Vec::with_capacity(expected_matrix_len);
        for batch_idx in 0..batch_size {
            let mat_offset = batch_idx * n * n;
            // Transpose from column-major to row-major
            for row in 0..n {
                for col in 0..n {
                    matrices_f32.push(matrices[mat_offset + col * n + row] as f32);
                }
            }
        }

        let rhs_f32: Vec<f32> = rhs.iter().map(|&v| v as f32).collect();

        // Create uniform buffer
        let uniforms = Uniforms {
            n: n as u32,
            batch_size: batch_size as u32,
        };
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Batched LU Uniforms"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        // Create matrix buffer (read-write, modified during factorization)
        let matrix_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Batched LU Matrices"),
            contents: bytemuck::cast_slice(&matrices_f32),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // Create RHS buffer (will also hold solutions)
        let rhs_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Batched LU RHS"),
            contents: bytemuck::cast_slice(&rhs_f32),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        // Create info buffer
        let info_zeros: Vec<i32> = vec![0; batch_size];
        let info_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Batched LU Info"),
            contents: bytemuck::cast_slice(&info_zeros),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        // Create staging buffers for reading results
        let solution_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Solution Staging"),
            size: (expected_rhs_len * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let info_staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Info Staging"),
            size: (batch_size * std::mem::size_of::<i32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Batched LU Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: matrix_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: rhs_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: info_buffer.as_entire_binding(),
                },
            ],
        });

        // Encode and submit compute work
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Batched LU Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Batched LU Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&self.pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            // One workgroup per matrix in the batch
            compute_pass.dispatch_workgroups(batch_size as u32, 1, 1);
        }

        // Copy results to staging buffers
        encoder.copy_buffer_to_buffer(
            &rhs_buffer,
            0,
            &solution_staging,
            0,
            (expected_rhs_len * std::mem::size_of::<f32>()) as u64,
        );
        encoder.copy_buffer_to_buffer(
            &info_buffer,
            0,
            &info_staging,
            0,
            (batch_size * std::mem::size_of::<i32>()) as u64,
        );

        queue.submit(std::iter::once(encoder.finish()));

        // Read solutions
        let solutions = {
            let buffer_slice = solution_staging.slice(..);
            let (sender, receiver) = std::sync::mpsc::channel();
            buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
                sender.send(result).unwrap();
            });
            device.poll(wgpu::Maintain::Wait);
            receiver
                .recv()
                .map_err(|e| WgpuError::Buffer(format!("Failed to receive map result: {}", e)))?
                .map_err(|e| WgpuError::Buffer(format!("Buffer mapping failed: {:?}", e)))?;

            let data = buffer_slice.get_mapped_range();
            let solutions_f32: &[f32] = bytemuck::cast_slice(&data);
            let solutions: Vec<f64> = solutions_f32.iter().map(|&v| v as f64).collect();
            drop(data);
            solution_staging.unmap();
            solutions
        };

        // Read info
        let singular_indices = {
            let buffer_slice = info_staging.slice(..);
            let (sender, receiver) = std::sync::mpsc::channel();
            buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
                sender.send(result).unwrap();
            });
            device.poll(wgpu::Maintain::Wait);
            receiver
                .recv()
                .map_err(|e| WgpuError::Buffer(format!("Failed to receive map result: {}", e)))?
                .map_err(|e| WgpuError::Buffer(format!("Buffer mapping failed: {:?}", e)))?;

            let data = buffer_slice.get_mapped_range();
            let info_array: &[i32] = bytemuck::cast_slice(&data);
            let singular: Vec<usize> = info_array
                .iter()
                .enumerate()
                .filter_map(|(i, &v)| if v > 0 { Some(i) } else { None })
                .collect();
            drop(data);
            info_staging.unmap();
            singular
        };

        if !singular_indices.is_empty() {
            log::warn!(
                "{} of {} matrices were singular",
                singular_indices.len(),
                batch_size
            );
        }

        Ok(BatchedSolveResult {
            solutions,
            singular_indices,
            n,
            batch_size,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn try_create_context() -> Option<Arc<WgpuContext>> {
        WgpuContext::new().ok().map(Arc::new)
    }

    #[test]
    fn test_batched_lu_identity() {
        let ctx = match try_create_context() {
            Some(c) => c,
            None => {
                eprintln!("Skipping test: no GPU available");
                return;
            }
        };

        let solver = MetalBatchedLuSolver::new(ctx).unwrap();
        let n = 2;
        let batch_size = 2;

        // Two 2×2 identity matrices in column-major order
        let matrices = vec![
            1.0, 0.0, 0.0, 1.0, // Identity 0
            1.0, 0.0, 0.0, 1.0, // Identity 1
        ];

        let rhs = vec![
            1.0, 2.0, // b0 = [1, 2]
            3.0, 4.0, // b1 = [3, 4]
        ];

        let result = solver.solve_batch(&matrices, &rhs, n, batch_size).unwrap();

        assert_eq!(result.batch_size, 2);
        assert!(result.singular_indices.is_empty());

        let sol0 = result.solution(0).unwrap();
        assert!(
            (sol0[0] - 1.0).abs() < 1e-4,
            "sol0[0] = {} (expected 1.0)",
            sol0[0]
        );
        assert!(
            (sol0[1] - 2.0).abs() < 1e-4,
            "sol0[1] = {} (expected 2.0)",
            sol0[1]
        );

        let sol1 = result.solution(1).unwrap();
        assert!(
            (sol1[0] - 3.0).abs() < 1e-4,
            "sol1[0] = {} (expected 3.0)",
            sol1[0]
        );
        assert!(
            (sol1[1] - 4.0).abs() < 1e-4,
            "sol1[1] = {} (expected 4.0)",
            sol1[1]
        );
    }

    #[test]
    fn test_batched_lu_simple() {
        let ctx = match try_create_context() {
            Some(c) => c,
            None => {
                eprintln!("Skipping test: no GPU available");
                return;
            }
        };

        let solver = MetalBatchedLuSolver::new(ctx).unwrap();
        let n = 2;
        let batch_size = 1;

        // Matrix: [[2, 1], [1, 3]] in column-major: [2, 1, 1, 3]
        // Solving Ax = b where b = [5, 5]
        // Solution should be x = [2, 1]
        let matrices = vec![2.0, 1.0, 1.0, 3.0];
        let rhs = vec![5.0, 5.0];

        let result = solver.solve_batch(&matrices, &rhs, n, batch_size).unwrap();

        assert!(result.singular_indices.is_empty());

        let sol = result.solution(0).unwrap();
        assert!(
            (sol[0] - 2.0).abs() < 1e-4,
            "x[0] = {} (expected 2.0)",
            sol[0]
        );
        assert!(
            (sol[1] - 1.0).abs() < 1e-4,
            "x[1] = {} (expected 1.0)",
            sol[1]
        );
    }

    #[test]
    fn test_batched_lu_singular() {
        let ctx = match try_create_context() {
            Some(c) => c,
            None => {
                eprintln!("Skipping test: no GPU available");
                return;
            }
        };

        let solver = MetalBatchedLuSolver::new(ctx).unwrap();
        let n = 2;
        let batch_size = 2;

        // Matrix 0: identity (non-singular)
        // Matrix 1: [[1, 2], [1, 2]] (singular - rows are identical) in column-major: [1, 1, 2, 2]
        let matrices = vec![
            1.0, 0.0, 0.0, 1.0, // Identity
            1.0, 1.0, 2.0, 2.0, // Singular
        ];
        let rhs = vec![1.0, 2.0, 1.0, 2.0];

        let result = solver.solve_batch(&matrices, &rhs, n, batch_size).unwrap();

        assert!(result.is_singular(1), "Matrix 1 should be detected as singular");
        assert!(!result.is_singular(0), "Matrix 0 should not be singular");
    }

    #[test]
    fn test_config_thresholds() {
        let config = GpuBatchConfig::default();

        // Default thresholds: min_batch=2000, min_matrix=100
        assert!(!config.should_use_gpu(50, 100));    // Matrix too small
        assert!(!config.should_use_gpu(100, 100));   // Batch too small
        assert!(config.should_use_gpu(100, 2000));   // Both OK
        assert!(!config.should_use_gpu(200, 2000));  // Matrix too large
    }
}
