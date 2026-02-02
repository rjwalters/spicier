// Batched LU factorization and solve compute shader.
//
// Each workgroup processes one matrix in the batch.
// Uses Doolittle's LU decomposition with partial pivoting.
// Operates directly on global memory (no workgroup shared memory for simplicity).
//
// Layout:
// - matrices: batch_size matrices with row_stride padding for coalesced access
// - rhs: batch_size vectors, each of length n (solutions written here)
// - info: batch_size integers (0 = success, >0 = singular at that row)

struct Uniforms {
    n: u32,
    batch_size: u32,
    row_stride: u32,    // Padded row stride (aligned to warp size)
    matrix_stride: u32, // Padded matrix size (row_stride * n)
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read_write> matrices: array<f32>;
@group(0) @binding(2) var<storage, read_write> rhs: array<f32>;
@group(0) @binding(3) var<storage, read_write> info: array<i32>;

// Get matrix element A[row, col] from global memory using stride-based layout
fn get_a(batch_idx: u32, row: u32, col: u32) -> f32 {
    let mat_offset = batch_idx * uniforms.matrix_stride;
    return matrices[mat_offset + row * uniforms.row_stride + col];
}

// Set matrix element A[row, col] in global memory using stride-based layout
fn set_a(batch_idx: u32, row: u32, col: u32, val: f32) {
    let mat_offset = batch_idx * uniforms.matrix_stride;
    matrices[mat_offset + row * uniforms.row_stride + col] = val;
}

// Get RHS/solution element
fn get_b(batch_idx: u32, i: u32) -> f32 {
    return rhs[batch_idx * uniforms.n + i];
}

// Set RHS/solution element
fn set_b(batch_idx: u32, i: u32, val: f32) {
    rhs[batch_idx * uniforms.n + i] = val;
}

@compute @workgroup_size(1)
fn main(@builtin(workgroup_id) workgroup_id: vec3<u32>) {
    let batch_idx = workgroup_id.x;
    let n = uniforms.n;

    if (batch_idx >= uniforms.batch_size) {
        return;
    }

    var singular_row: i32 = 0;

    // LU factorization with partial pivoting
    for (var k = 0u; k < n; k = k + 1u) {
        // Find pivot (largest absolute value in column k, rows k to n-1)
        var max_val = abs(get_a(batch_idx, k, k));
        var max_row = k;

        for (var i = k + 1u; i < n; i = i + 1u) {
            let val = abs(get_a(batch_idx, i, k));
            if (val > max_val) {
                max_val = val;
                max_row = i;
            }
        }

        // Check for singularity
        if (max_val < 1e-10) {
            singular_row = i32(k + 1u);
        }

        // Swap rows k and max_row in matrix and RHS
        if (max_row != k) {
            for (var j = 0u; j < n; j = j + 1u) {
                let tmp = get_a(batch_idx, k, j);
                set_a(batch_idx, k, j, get_a(batch_idx, max_row, j));
                set_a(batch_idx, max_row, j, tmp);
            }
            let tmp_b = get_b(batch_idx, k);
            set_b(batch_idx, k, get_b(batch_idx, max_row));
            set_b(batch_idx, max_row, tmp_b);
        }

        // Gaussian elimination
        let diag = get_a(batch_idx, k, k);
        if (abs(diag) > 1e-10) {
            for (var i = k + 1u; i < n; i = i + 1u) {
                let factor = get_a(batch_idx, i, k) / diag;
                set_a(batch_idx, i, k, factor);  // Store L factor

                for (var j = k + 1u; j < n; j = j + 1u) {
                    let aij = get_a(batch_idx, i, j);
                    let akj = get_a(batch_idx, k, j);
                    set_a(batch_idx, i, j, aij - factor * akj);
                }

                // Apply to RHS as well
                let bi = get_b(batch_idx, i);
                let bk = get_b(batch_idx, k);
                set_b(batch_idx, i, bi - factor * bk);
            }
        }
    }

    // Backward substitution: Ux = b (U is in upper triangle, L factors in lower)
    // At this point, b already has forward substitution applied
    for (var i_plus_one = n; i_plus_one > 0u; i_plus_one = i_plus_one - 1u) {
        let i = i_plus_one - 1u;
        var sum = get_b(batch_idx, i);

        for (var j = i + 1u; j < n; j = j + 1u) {
            sum = sum - get_a(batch_idx, i, j) * get_b(batch_idx, j);
        }

        let diag = get_a(batch_idx, i, i);
        if (abs(diag) > 1e-10) {
            set_b(batch_idx, i, sum / diag);
        } else {
            set_b(batch_idx, i, 0.0);
        }
    }

    info[batch_idx] = singular_row;
}
