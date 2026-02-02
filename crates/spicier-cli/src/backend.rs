//! Compute backend detection and selection.

use spicier_solver::ComputeBackend;

/// Detect and select the compute backend based on CLI argument.
pub fn detect_backend(name: &str) -> ComputeBackend {
    match name.to_lowercase().as_str() {
        "cpu" => ComputeBackend::Cpu,
        "cuda" => {
            #[cfg(feature = "cuda")]
            {
                if spicier_backend_cuda::context::CudaContext::is_available() {
                    ComputeBackend::Cuda { device_id: 0 }
                } else {
                    eprintln!("Warning: CUDA requested but not available, falling back to CPU");
                    ComputeBackend::Cpu
                }
            }
            #[cfg(not(feature = "cuda"))]
            {
                eprintln!("Warning: CUDA support not compiled in, falling back to CPU");
                ComputeBackend::Cpu
            }
        }
        "metal" => {
            #[cfg(feature = "metal")]
            {
                if spicier_backend_metal::context::WgpuContext::is_available() {
                    ComputeBackend::Metal {
                        adapter_name: String::new(),
                    }
                } else {
                    eprintln!("Warning: Metal/WebGPU requested but not available, falling back to CPU");
                    ComputeBackend::Cpu
                }
            }
            #[cfg(not(feature = "metal"))]
            {
                eprintln!("Warning: Metal support not compiled in, falling back to CPU");
                ComputeBackend::Cpu
            }
        }
        _ => {
            // Try Metal first (macOS), then CUDA, then CPU
            #[cfg(feature = "metal")]
            {
                if spicier_backend_metal::context::WgpuContext::is_available() {
                    return ComputeBackend::Metal {
                        adapter_name: String::new(),
                    };
                }
            }
            #[cfg(feature = "cuda")]
            {
                if spicier_backend_cuda::context::CudaContext::is_available() {
                    return ComputeBackend::Cuda { device_id: 0 };
                }
            }
            ComputeBackend::Cpu
        }
    }
}
