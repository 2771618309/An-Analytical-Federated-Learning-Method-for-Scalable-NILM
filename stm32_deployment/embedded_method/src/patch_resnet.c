#include "patch_resnet.h"
#include <stdio.h>
#include "arm_math.h"
#include "edge_memory.h"

/*
 * Embedded PatchResNet inference and analytical linear solver utilities.
 *
 * This file keeps the deployment-side model logic used by the method:
 * patch embedding, residual 1D convolution blocks, feature extraction, and
 * local analytical-head matrix computation. Model parameters are supplied by
 * model_params.h/model_params_template.c and are intentionally separated from
 * the implementation.
 */

// ==================== Basic Operations ====================

// In-place ReLU activation.
void relu_inplace(float *data, int length) {
    for (int i = 0; i < length; i++) {
        if (data[i] < 0.0f) {
            data[i] = 0.0f;
        }
    }
}

// Flatten a [channels, length] tensor into a contiguous vector.
void flatten(float *input, float *output, int channels, int length) {
    // Input format: [channels, length].
    // Output format: [channels * length]. The function copies contiguous data.
    memcpy(output, input, channels * length * sizeof(float));
}


// Circular padding used by token embedding convolution.
void circular_pad1d(float *input, float *output, int channels, int length, int padding) {
    // Input: [channels, length].
    // Output: [channels, length + 2 * padding].
    int padded_len = length + 2 * padding;
    
    for (int c = 0; c < channels; c++) {
        for (int i = 0; i < padded_len; i++) {
            int src_idx;
            if (i < padding) {
                // Left circular padding.
                src_idx = length - padding + i;
            } else if (i >= length + padding) {
                // Right circular padding.
                src_idx = i - length - padding;
            } else {
                // Original middle region.
                src_idx = i - padding;
            }
            output[c * padded_len + i] = input[c * length + src_idx];
        }
    }
}


/**
 * @brief Normalize one sample in place to the range [-1, 1].
 * 
 * @param data Input/output array with length elements.
 * @param length Number of values in the sample.
 * @param min_val Optional output minimum value. Pass NULL if unused.
 * @param max_val Optional output maximum value. Pass NULL if unused.
 * @return 0 on normal normalization, -1 when the input range is degenerate.
 */
int normalize_sample_inplace(float *data, int length, float *min_val, float *max_val)
{
    if (!data || length <= 0) {
        return -1;
    }
    
    // 1. Find the sample minimum and maximum.
    float min = data[0];
    float max = data[0];
    
    for (int i = 1; i < length; i++) {
        if (data[i] < min) min = data[i];
        if (data[i] > max) max = data[i];
    }
    
    // 2. Check whether all values are effectively identical.
    float range = max - min;
    if (range < 1e-8f) {
        // Use zeros when the sample has no dynamic range.
        for (int i = 0; i < length; i++) {
            data[i] = 0.0f;
        }
        if (min_val) *min_val = min;
        if (max_val) *max_val = max;
        return -1;  // Indicates that the sample had no dynamic range.
    }
    
    // 3. Normalize to [0, 1], then scale to [-1, 1].
    for (int i = 0; i < length; i++) {
        data[i] = (data[i] - min) / range;  // Normalize to [0, 1].
        data[i] = data[i] * 2.0f - 1.0f;    // Scale to [-1, 1].
    }
    
    // 4. Return min/max when requested.
    if (min_val) *min_val = min;
    if (max_val) *max_val = max;
    
    return 0;
}

// Compute X * X^T + rg * I.
// X is row-major [batch_size, features]; output is [batch_size, batch_size].
void matmul_xxt_with_reg(float *X, int batch_size, int features, float rg, float *output) {
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < batch_size; j++) {
            float sum = 0.0f;
            for (int k = 0; k < features; k++) {
                sum += X[i * features + k] * X[j * features + k];
            }
            // Add the regularization term on the diagonal.
            if (rg > 0.0f && i == j) {
                sum += rg;
            }
            output[i * batch_size + j] = sum;
        }
    }
}

// Invert a symmetric positive definite matrix.
// A is row-major [n, n].
// A_inv is row-major [n, n].
// Return 0 on success, -1 if decomposition fails.
int invert_spd_matrix(float *A, int n, float *A_inv) {
    // 1. Copy A into a temporary lower-triangular Cholesky buffer.
    float *L = (float *)edge_malloc(n * n * sizeof(float));
    if (!L) return -1;

    for (int i = 0; i < n * n; ++i) {
        L[i] = A[i];
    }

    // 2. Cholesky decomposition: A = L * L^T.
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            float sum = L[i * n + j];
            for (int k = 0; k < j; ++k) {
                sum -= L[i * n + k] * L[j * n + k];
            }

            if (i == j) {
                if (sum <= 0.0f) {
                    // Matrix is not positive definite.
                    edge_free(L);
                    return -1;
                }
                L[i * n + j] = sqrtf(sum);
            } else {
                L[i * n + j] = sum / L[j * n + j];
            }
        }

        // Clear the upper triangular part.
        for (int j = i + 1; j < n; ++j) {
            L[i * n + j] = 0.0f;
        }
    }

    // 3. Invert L in place and store L_inv in the same lower-triangular buffer.
    // Standard lower-triangular inverse formula.
    for (int i = 0; i < n; ++i) {
        // Diagonal element.
        L[i * n + i] = 1.0f / L[i * n + i];

        // Off-diagonal lower-triangular elements.
        for (int j = 0; j < i; ++j) {
            float sum = 0.0f;
            for (int k = j; k < i; ++k) {
                sum -= L[i * n + k] * L[k * n + j];
            }
            L[i * n + j] = sum / L[i * n + i];
        }
    }

    // 4. A_inv = L_inv^T * L_inv
    // L now stores L_inv in lower-triangular form.
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            float sum = 0.0f;
            for (int k = i; k < n; ++k) {
                sum += L[k * n + i] * L[k * n + j];
            }
            A_inv[i * n + j] = sum;
            A_inv[j * n + i] = sum; // Symmetric copy.
        }
    }

    edge_free(L);
    return 0;
}


// ==================== Double-Precision Matrix Inversion ====================
// Gauss-Jordan matrix inversion with partial pivoting.
// Use Gauss-Jordan elimination with partial pivoting.
// A is an input matrix [n, n] and will be modified.
// A_inv is the output inverse matrix [n, n].
// Return 0 on success, -1 when the matrix is singular.
int invert_matrix_gauss_jordan_double(float *A, int n, float *A_inv) {
    // Use double precision for intermediate computations.
    double *aug = (double *)edge_malloc(n * 2 * n * sizeof(double));
	// Allocation check.
    if (aug == NULL) {
        fprintf(stderr, "ERROR: malloc failed for aug matrix (%d bytes)\n",
                   n * 2 * n * sizeof(double));
        return -1;
    }
    if (!aug) return -1;
    
    // Build the augmented matrix [A | I].
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            aug[i * (2 * n) + j] = (double)A[i * n + j];
            aug[i * (2 * n) + (n + j)] = (i == j) ? 1.0 : 0.0;
        }
    }
    
    // Gauss-Jordan elimination with partial pivot selection.
    for (int col = 0; col < n; col++) {
        // Find the largest pivot.
        int max_row = col;
        double max_val = fabs(aug[col * (2 * n) + col]);
        
        for (int row = col + 1; row < n; row++) {
            double val = fabs(aug[row * (2 * n) + col]);
            if (val > max_val) {
                max_val = val;
                max_row = row;
            }
        }
        
        // Check singularity.
        if (max_val < 1e-14) {
            fprintf(stderr, "Warning: Matrix is singular at col %d (pivot=%e)\n", col, max_val);
            edge_free(aug);
            return -1;
        }
        
        // Swap rows.
        if (max_row != col) {
            for (int k = 0; k < 2 * n; k++) {
                double tmp = aug[col * (2 * n) + k];
                aug[col * (2 * n) + k] = aug[max_row * (2 * n) + k];
                aug[max_row * (2 * n) + k] = tmp;
            }
        }
        
        // Normalize the pivot row.
        double pivot = aug[col * (2 * n) + col];
        for (int k = 0; k < 2 * n; k++) {
            aug[col * (2 * n) + k] /= pivot;
        }
        
        // Eliminate all other rows.
        for (int row = 0; row < n; row++) {
            if (row != col) {
                double factor = aug[row * (2 * n) + col];
                for (int k = 0; k < 2 * n; k++) {
                    aug[row * (2 * n) + k] -= factor * aug[col * (2 * n) + k];
                }
            }
        }
    }
    
    // Extract the inverse matrix and cast back to float.
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A_inv[i * n + j] = (float)aug[i * (2 * n) + (n + j)];
        }
    }
    
    edge_free(aug);
    return 0;
}

// ==================== Single-Precision Matrix Inversion ====================
// Use Gauss-Jordan elimination with partial pivoting in float precision.
// A is an input matrix [n, n] and will be modified.
// A_inv is the output inverse matrix [n, n].
// Return 0 on success, -1 when the matrix is singular.
int invert_matrix_gauss_jordan_float(float *A, int n, float *A_inv) {
    // Use float precision for computations.
    float *aug = (float *)edge_malloc(n * 2 * n * sizeof(float));
    
    // Allocation check.
    if (aug == NULL) {
        fprintf(stderr, "ERROR: malloc failed for aug matrix (%d bytes)\n",
                   n * 2 * n * sizeof(float));
        return -1;
    }
    
    // Build the augmented matrix [A | I].
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            aug[i * (2 * n) + j] = A[i * n + j];
            aug[i * (2 * n) + (n + j)] = (i == j) ? 1.0f : 0.0f;
        }
    }
    
    // Float pivot threshold is looser than the double-precision threshold.
    const float PIVOT_THRESHOLD = 1e-7f;  // Float precision has about 7 significant digits.
    
    // Gauss-Jordan elimination with partial pivot selection.
    for (int col = 0; col < n; col++) {
        // Find the largest pivot.
        int max_row = col;
        float max_val = fabsf(aug[col * (2 * n) + col]);
        
        for (int row = col + 1; row < n; row++) {
            float val = fabsf(aug[row * (2 * n) + col]);
            if (val > max_val) {
                max_val = val;
                max_row = row;
            }
        }
        
        // Check singularity.
        if (max_val < PIVOT_THRESHOLD) {
            fprintf(stderr, "Warning: Matrix is singular at col %d (pivot=%.2e)\n",
                      col, max_val);
            edge_free(aug);
            return -1;
        }
        
        // Swap rows.
        if (max_row != col) {
            for (int k = 0; k < 2 * n; k++) {
                float tmp = aug[col * (2 * n) + k];
                aug[col * (2 * n) + k] = aug[max_row * (2 * n) + k];
                aug[max_row * (2 * n) + k] = tmp;
            }
        }
        
        // Precompute the inverse pivot to avoid repeated division.
        float pivot = aug[col * (2 * n) + col];
        float inv_pivot = 1.0f / pivot;
        
        // Normalize from col onward; earlier entries are already zero.
        for (int k = col; k < 2 * n; k++) {
            aug[col * (2 * n) + k] *= inv_pivot;
        }
        
        // Eliminate all other rows.
        for (int row = 0; row < n; row++) {
            if (row != col) {
                float factor = aug[row * (2 * n) + col];
                // Process only the nonzero region.
                for (int k = col; k < 2 * n; k++) {
                    aug[row * (2 * n) + k] -= factor * aug[col * (2 * n) + k];
                }
            }
        }
    }
    
    // Extract the inverse matrix.
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A_inv[i * n + j] = aug[i * (2 * n) + (n + j)];
        }
    }
    
    edge_free(aug);
    return 0;
}



// Compute X^T * Y.
// X: [batch_size, features]
// Y is either [batch_size, out_dim] or a label vector [batch_size].
// is_label_vector=1 means Y stores class indices and is handled as one-hot labels.
// Output out has shape [features, out_dim].
// ========================================
// Optimized X^T * Y implementation with label-vector support.
// ========================================
void matmul_xt_y(float *X, float *Y, int batch_size, int features, int out_dim, int is_label_vector, float *out) 
{
    memset(out, 0, features * out_dim * sizeof(float));

    if (is_label_vector) {
        // Use one-hot sparsity for label vectors.
        for (int b = 0; b < batch_size; b++) {
            int label = (int)(Y[b] + 0.5f);
            if (label >= 0 && label < out_dim) {
                for (int f = 0; f < features; f++) {
                    out[f * out_dim + label] += X[b * features + f];
                }
            }
        }
    } else {
        // Standard matrix multiplication.
        for (int f = 0; f < features; f++) {
            for (int o = 0; o < out_dim; o++) {
                float sum = 0.0f;
                for (int b = 0; b < batch_size; b++) {
                    sum += X[b * features + f] * Y[b * out_dim + o];
                }
                out[f * out_dim + o] = sum;
            }
        }
    }
}


// Compute G = X X^T + rg I, G_inv, and W = X^T @ G_inv @ Y.
// ========================================
// Optimized regularized Gram solve.
// ========================================
// Compute G = X X^T + rg I, G_inv, and W = X^T @ G_inv @ Y.
// Parameters:
// X: input matrix [batch_size, features].
// Y: label matrix [batch_size, out_dim] or label vector [batch_size].
// batch_size: number of samples.
// features: number of feature columns.
// out_dim: number of output classes.
// is_label_vector: 1 for label vector, 0 for one-hot matrix.
// rg: regularization coefficient.
// gram: Gram matrix [batch_size, batch_size].
// gram_inv: inverse Gram matrix [batch_size, batch_size].
// alpha: intermediate matrix [batch_size, out_dim].
// W: output weight matrix [features, out_dim].
int gram_with_reg_and_inv(
    float *X, float *Y, 
    int batch_size, int features, int out_dim, 
    int is_label_vector, float rg, 
    float *gram, float *gram_inv, float* alpha, float *W)
{
    // 1. Validate X.
    for (int i = 0; i < batch_size * features; i++) {
        if (isnan(X[i]) || isinf(X[i])) {
            fprintf(stderr, "ERROR: X[%d] = %f (NaN/Inf)\n", i, X[i]);
            memset(W, 0, features * out_dim * sizeof(float));
            return -1;
        }
    }

    // 2. Compute Gram matrix G = X @ X^T + rg * I.
    matmul_xxt_with_reg(X, batch_size, features, rg, gram);
    
    for (int i = 0; i < batch_size * batch_size; i++) {
        if (isnan(gram[i]) || isinf(gram[i])) {
            fprintf(stderr, "ERROR: gram[%d] = %f (NaN/Inf)\n", i, gram[i]);
            memset(W, 0, features * out_dim * sizeof(float));
            return -1;
        }
    }

    // 3. Invert G to obtain G_inv.
//    int ret = invert_matrix_gauss_jordan_double(gram, batch_size, gram_inv);
		int ret = invert_matrix_gauss_jordan_float(gram, batch_size, gram_inv);
    if (ret != 0) {
        fprintf(stderr, "ERROR: Matrix inversion failed\n");
        memset(W, 0, features * out_dim * sizeof(float));
        return -1;
    }
    
    for (int i = 0; i < batch_size * batch_size; i++) {
        if (isnan(gram_inv[i]) || isinf(gram_inv[i])) {
            fprintf(stderr, "ERROR: gram_inv failed\n");
            memset(W, 0, features * out_dim * sizeof(float));
            return -1;
        }
    }

    // 4. Compute alpha = G_inv @ Y.
    memset(alpha, 0, batch_size * out_dim * sizeof(float));
    
    if (is_label_vector) {
        for (int j = 0; j < batch_size; j++) {
            int label = (int)(Y[j] + 0.5f);
            if (label >= 0 && label < out_dim) {
                for (int i = 0; i < batch_size; i++) {
                    alpha[i * out_dim + label] += gram_inv[i * batch_size + j];
                }
            }
        }
    } else {
        for (int i = 0; i < batch_size; i++) {
            for (int o = 0; o < out_dim; o++) {
                float sum = 0.0f;
                for (int j = 0; j < batch_size; j++) {
                    sum += gram_inv[i * batch_size + j] * Y[j * out_dim + o];
                }
                alpha[i * out_dim + o] = sum;
            }
        }
    }

    // 5. Compute W = X^T @ alpha.
    matmul_xt_y(X, alpha, batch_size, features, out_dim, 0, W);
    
//    // Optional NaN/Inf cleanup.
//    for (int i = 0; i < features * out_dim; i++) {
//        if (isnan(W[i]) || isinf(W[i])) {
//            W[i] = 0.0f;
//        }
//    }

    return 0;
}


/**
 * @brief Compute W = X^T @ (X @ X^T + rg*I)^(-1) @ Y.
 * 
 * @param X Input feature matrix [batch_size, features], row-major.
 * @param Y Label data:
 *                   - if is_label_vector=1: class-index vector [batch_size].
 *                   - if is_label_vector=0: one-hot matrix [batch_size, out_dim].
 * @param batch_size Number of samples.
 * @param features Number of feature columns.
 * @param out_dim Number of output classes.
 * @param is_label_vector Label format flag: 1=class indices, 0=one-hot matrix.
 * @param rg Regularization coefficient.
 * @param gram Output Gram matrix [batch_size, batch_size].
 * @param gram_inv Output inverse Gram matrix [batch_size, batch_size].
 * @param alpha Output intermediate alpha = G_inv @ Y [batch_size, out_dim].
 * @param W Output weight matrix [features, out_dim].
 */
void gram_with_reg_and_inv_cmsis(float *X, float *Y, int batch_size, int features, 
                           int out_dim, int is_label_vector, float rg, 
                           float *gram, float *gram_inv, float* alpha, float *W)
{
    // =====================================================
    // Validate input pointers.
    // =====================================================
    if (!X || !Y || !gram || !gram_inv || !alpha || !W) {
        if (W) memset(W, 0, features * out_dim * sizeof(float));
        return;
    }

    // =====================================================
    // Step 1: compute Gram matrix G = X @ X^T + rg * I.
    // 
    // The Gram matrix measures sample-to-sample similarity.
    // Add rg*I to improve numerical stability.
    // 
    // CMSIS-DSP functions used:
    // - arm_mat_trans_f32(): transpose X to compute X^T.
    // - arm_mat_mult_f32(): matrix multiplication for X @ X^T.
    // =====================================================
    
    // Allocate X^T storage [features, batch_size].
    float *Xt = (float *)edge_malloc(features * batch_size * sizeof(float));
    if (!Xt) {
        memset(W, 0, features * out_dim * sizeof(float));
        return;
    }
    
    // Initialize CMSIS-DSP matrix descriptors.
    arm_matrix_instance_f32 X_mat;   // Original matrix X [batch_size, features].
    arm_matrix_instance_f32 Xt_mat;  // Transposed matrix X^T [features, batch_size].
    arm_matrix_instance_f32 G_mat;   // Gram matrix G [batch_size, batch_size].
    
    arm_mat_init_f32(&X_mat, batch_size, features, X);
    arm_mat_init_f32(&Xt_mat, features, batch_size, Xt);
    arm_mat_init_f32(&G_mat, batch_size, batch_size, gram);
    
    // CMSIS-DSP: compute X^T.
    arm_mat_trans_f32(&X_mat, &Xt_mat);
    
    // CMSIS-DSP: compute G = X @ X^T.
    arm_mat_mult_f32(&X_mat, &Xt_mat, &G_mat);
    
    // Add regularization: G = G + rg * I.
    for (int i = 0; i < batch_size; i++) {
        gram[i * batch_size + i] += rg;
    }
    
    // Release temporary X^T storage.
    edge_free(Xt);

    // =====================================================
    // Step 2: compute inverse Gram matrix G_inv = G^(-1).
    // 
    // Use float64 inversion to improve numerical precision.
    // 
    // CMSIS-DSP functions used:
    // - arm_mat_inverse_f64(): double-precision matrix inversion.
    // 
    // Flow: float32 -> float64 -> inverse -> float32.
    // =====================================================
    
    // Allocate double-precision temporary buffers.
    float64_t *gram_f64 = (float64_t *)edge_malloc(batch_size * batch_size * sizeof(float64_t));
    float64_t *gram_inv_f64 = (float64_t *)edge_malloc(batch_size * batch_size * sizeof(float64_t));
    
    if (!gram_f64 || !gram_inv_f64) {
        if (gram_f64) edge_free(gram_f64);
        if (gram_inv_f64) edge_free(gram_inv_f64);
        memset(W, 0, features * out_dim * sizeof(float));
        return;
    }
    
    // Promote precision from float32 to float64.
    for (int i = 0; i < batch_size * batch_size; i++) {
        gram_f64[i] = (float64_t)gram[i];
    }
    
    
    // Initialize f64 matrix descriptors directly.
		arm_matrix_instance_f64 G_mat_f64 = {batch_size, batch_size, gram_f64};
		arm_matrix_instance_f64 Ginv_mat_f64 = {batch_size, batch_size, gram_inv_f64};
    
    // CMSIS-DSP: double-precision matrix inversion.
    arm_status status = arm_mat_inverse_f64(&G_mat_f64, &Ginv_mat_f64);
    
    // Check whether inversion succeeded.
    if (status != ARM_MATH_SUCCESS) {
        // If inversion fails, return a zero weight matrix.
        edge_free(gram_f64);
        edge_free(gram_inv_f64);
        memset(W, 0, features * out_dim * sizeof(float));
        return;
    }
    
    // Convert precision back from float64 to float32.
    for (int i = 0; i < batch_size * batch_size; i++) {
        gram_inv[i] = (float)gram_inv_f64[i];
    }
    
    // Release double-precision temporary buffers.
    edge_free(gram_f64);
    edge_free(gram_inv_f64);

    // =====================================================
    // Step 3: compute alpha = G_inv @ Y.
    // 
    // If Y is a label vector, use one-hot sparsity without building Y explicitly.
    // If Y is one-hot, use CMSIS-DSP matrix multiplication.
    // 
    // CMSIS-DSP functions used:
    // - arm_mat_mult_f32(): matrix multiplication when Y is a matrix.
    // =====================================================
    
    // Initialize alpha to zeros.
    memset(alpha, 0, batch_size * out_dim * sizeof(float));
    
    if (is_label_vector) {
        // -------------------------------------------------
        // Sparse path: Y stores class indices.
        // 
        // One-hot labels have exactly one positive entry per row.
        // alpha[i, c] = sum_j G_inv[i, j] * Y_onehot[j, c].
        //             = sum_{j: label[j] == c} G_inv[i, j].
        // 
        // This avoids building the full one-hot matrix.
        // -------------------------------------------------
        for (int j = 0; j < batch_size; j++) {
            int label = (int)(Y[j] + 0.5f);  // Read the label of sample j.
            if (label >= 0 && label < out_dim) {
                // Accumulate column j of G_inv into the label column of alpha.
                for (int i = 0; i < batch_size; i++) {
                    alpha[i * out_dim + label] += gram_inv[i * batch_size + j];
                }
            }
        }
    } else {
        // -------------------------------------------------
        // Standard matrix multiplication when Y is already one-hot.
        // alpha = G_inv @ Y
        // -------------------------------------------------
        arm_matrix_instance_f32 Ginv_mat;   // G^(-1) [batch_size, batch_size]
        arm_matrix_instance_f32 Y_mat;      // Y [batch_size, out_dim]
        arm_matrix_instance_f32 Alpha_mat;  // alpha [batch_size, out_dim]
        
        arm_mat_init_f32(&Ginv_mat, batch_size, batch_size, gram_inv);
        arm_mat_init_f32(&Y_mat, batch_size, out_dim, Y);
        arm_mat_init_f32(&Alpha_mat, batch_size, out_dim, alpha);
        
        // CMSIS-DSP: compute alpha = G_inv @ Y.
        arm_mat_mult_f32(&Ginv_mat, &Y_mat, &Alpha_mat);
    }

    // =====================================================
    // Step 4: compute W = X^T @ alpha.
    // 
    // W has shape [features, out_dim].
    // This is the final local analytical linear-head weight.
    // 
    // CMSIS-DSP functions used:
    // - arm_mat_trans_f32(): transpose X to compute X^T.
    // - arm_mat_mult_f32(): matrix multiplication for X^T @ alpha.
    // =====================================================
    
    // Allocate X^T storage.
    float *Xt2 = (float *)edge_malloc(features * batch_size * sizeof(float));
    if (!Xt2) {
        memset(W, 0, features * out_dim * sizeof(float));
        return;
    }
    
    // Reinitialize matrix descriptors because Xt was released earlier.
    arm_matrix_instance_f32 X_mat2;     // X [batch_size, features]
    arm_matrix_instance_f32 Xt2_mat;    // X^T [features, batch_size]
    arm_matrix_instance_f32 Alpha_mat;  // alpha [batch_size, out_dim]
    arm_matrix_instance_f32 W_mat;      // W [features, out_dim]
    
    arm_mat_init_f32(&X_mat2, batch_size, features, X);
    arm_mat_init_f32(&Xt2_mat, features, batch_size, Xt2);
    
    // CMSIS-DSP: compute X^T.
    arm_mat_trans_f32(&X_mat2, &Xt2_mat);
    
    arm_mat_init_f32(&Alpha_mat, batch_size, out_dim, alpha);
    arm_mat_init_f32(&W_mat, features, out_dim, W);
    
    // CMSIS-DSP: compute W = X^T @ alpha.
    arm_mat_mult_f32(&Xt2_mat, &Alpha_mat, &W_mat);
    
    // Release temporary X^T storage.
    edge_free(Xt2);
    
    // =====================================================
    // Computation complete; W stores the final weight matrix.
    // W is row-major [features, out_dim].
    // =====================================================
}

// ==================== Positional Embedding ====================

// Initialize positional embedding from saved parameters.
void init_positional_embedding(PositionalEmbedding *pe, int d_model, int max_len, float *saved_pe) {
    pe->d_model = d_model;
    pe->max_len = max_len;
    
    // Use the saved positional encoding directly.
    pe->pe = saved_pe;
}

// Positional embedding forward pass.
void positional_embedding_forward(PositionalEmbedding *pe, float *output, int seq_len) {
    // Output the first seq_len positions: [seq_len, d_model].
    if (seq_len > pe->max_len) {
        fprintf(stderr, "Warning: seq_len %d exceeds max_len %d\n", seq_len, pe->max_len);
        seq_len = pe->max_len;
    }
    
    memcpy(output, pe->pe, seq_len * pe->d_model * sizeof(float));
}





// ==================== Token Embedding ====================

// Initialize the token embedding layer.
void init_token_embedding(TokenEmbedding *te, int c_in, int d_model) {
    te->c_in = c_in;
    te->d_model = d_model;
    
    // Configure the convolution: kernel_size=3, circular padding, no bias.
    te->token_conv.in_channels = c_in;
    te->token_conv.out_channels = d_model;
    te->token_conv.kernel_size = 3;
    te->token_conv.stride = 1;
    te->token_conv.padding = 1; // circular padding
    te->token_conv.use_circular_pad = 1; // Use circular padding.
    te->token_conv.bias = NULL;
    
    // Weights are assigned externally: te->token_conv.weights = ...;
}

// Token embedding forward pass.
void token_embedding_forward(TokenEmbedding *te, float *input, float *output, 
                            int batch_size, int seq_len) {
    // Input: [batch_size, seq_len, c_in].
    // Convert to: [batch_size, c_in, seq_len].
    // Output: [batch_size, d_model, seq_len], then convert back to [batch_size, seq_len, d_model].
    
    float *transposed_input = (float*)edge_malloc(batch_size * te->c_in * seq_len * sizeof(float));
    float *conv_output = (float*)edge_malloc(batch_size * te->d_model * seq_len * sizeof(float));
    
    if (!transposed_input || !conv_output) {
        fprintf(stderr, "Memory allocation failed in token_embedding_forward\n");
        if (transposed_input) edge_free(transposed_input);
        if (conv_output) edge_free(conv_output);
        return;
    }
    
    // Transpose: [B, N, C] -> [B, C, N].
    for (int b = 0; b < batch_size; b++) {
        for (int n = 0; n < seq_len; n++) {
            for (int c = 0; c < te->c_in; c++) {
                transposed_input[b * te->c_in * seq_len + c * seq_len + n] = 
                    input[b * seq_len * te->c_in + n * te->c_in + c];
            }
        }
    }
    
    // Apply convolution to each batch item.
    for (int b = 0; b < batch_size; b++) {
        conv1d_forward(&te->token_conv, 
                      &transposed_input[b * te->c_in * seq_len], 
                      seq_len,
                      &conv_output[b * te->d_model * seq_len]);
    }
    
    // Transpose back: [B, D, N] -> [B, N, D].
    for (int b = 0; b < batch_size; b++) {
        for (int n = 0; n < seq_len; n++) {
            for (int d = 0; d < te->d_model; d++) {
                output[b * seq_len * te->d_model + n * te->d_model + d] = 
                    conv_output[b * te->d_model * seq_len + d * seq_len + n];
            }
        }
    }
    
    edge_free(transposed_input);
    edge_free(conv_output);
}

// ==================== Patch Embedding ====================

// Initialize patch embedding from saved parameters.
void init_patch_embedding(PatchEmbedding *pe, int d_model, int patch_len, int stride, 
                         float dropout, float *saved_pos_emb, float *token_conv_weights) {
    pe->d_model = d_model;
    pe->patch_len = patch_len;
    pe->stride = stride;
    pe->dropout_rate = dropout;
    
    // Initialize token embedding.
    init_token_embedding(&pe->value_embedding, patch_len, d_model);
    pe->value_embedding.token_conv.weights = token_conv_weights;
    
    // Initialize positional embedding from saved parameters.
    init_positional_embedding(&pe->pos_embedding, d_model, 5000, saved_pos_emb);
}

// Patch embedding forward pass for single-sample inference.
void patch_embedding_forward(PatchEmbedding *pe, float *input, float *output, 
                             int batch_size, int input_channels, int input_length, int *n_vars) {
    // Input: [input_channels, input_length].
    // Output: [num_patches, d_model]; each patch becomes one token/channel.
    // batch_size is kept for API compatibility; this implementation assumes batch_size=1 and one input channel.

    *n_vars = input_channels;

    int num_patches = input_length / pe->patch_len;

    // Step 1: split input into patches [num_patches, patch_len].
    float *patches = (float*)edge_malloc(num_patches * pe->patch_len * sizeof(float));
    if (!patches) {
        fprintf(stderr, "Memory allocation failed for patches\n");
        return;
    }

    // Process only the first input channel because INPUT_CHANNELS = 1.
    for (int p = 0; p < num_patches; p++) {
        int start_idx = p * pe->patch_len;
        for (int i = 0; i < pe->patch_len; i++) {
            patches[p * pe->patch_len + i] = input[start_idx + i];
        }
    }

    // Step 2: Token Embedding -> [num_patches, d_model]
    float *value_emb = (float*)edge_malloc(num_patches * pe->d_model * sizeof(float));
    if (!value_emb) {
        fprintf(stderr, "Memory allocation failed for value_emb\n");
        edge_free(patches);
        return;
    }
    token_embedding_forward(&pe->value_embedding, patches, value_emb, 1, num_patches);

    // Step 3: add shared positional embedding.
    float *pos_emb = (float*)edge_malloc(num_patches * pe->d_model * sizeof(float));
    if (!pos_emb) {
        fprintf(stderr, "Memory allocation failed for pos_emb\n");
        edge_free(patches);
        edge_free(value_emb);
        return;
    }
    positional_embedding_forward(&pe->pos_embedding, pos_emb, num_patches);

    // Step 4: Add value_emb + pos_emb -> output: [num_patches, d_model]
    for (int p = 0; p < num_patches; p++) {
        for (int d = 0; d < pe->d_model; d++) {
            output[p * pe->d_model + d] = value_emb[p * pe->d_model + d] + pos_emb[p * pe->d_model + d];
        }
    }

    edge_free(patches);
    edge_free(value_emb);
    edge_free(pos_emb);
}

/**
 * @brief Release resources owned by a PatchEmbedding descriptor.
 * @param pe Patch embedding descriptor.
 *
 * The open-source descriptor stores externally owned parameter pointers, so no
 * parameter memory is freed here. Clearing the pointers prevents accidental
 * reuse after the embedding object is retired.
 */
void free_patch_embedding(PatchEmbedding *pe) {
    if (pe == NULL) {
        return;
    }

    pe->value_embedding.token_conv.weights = NULL;
    pe->value_embedding.token_conv.bias = NULL;
    pe->pos_embedding.pe = NULL;
}

// ==================== Convolution Layer ====================

// 1D convolution forward pass with stride, padding, optional bias, and optional circular padding.
void conv1d_forward(Conv1DLayer *conv, float *input, int input_length, float *output) {
    int padded_length = input_length + 2 * conv->padding;
    int output_length = (padded_length - conv->kernel_size) / conv->stride + 1;

    // Create padded input.
    float *padded_input = (float*)calloc(padded_length * conv->in_channels, sizeof(float));
    if (padded_input == NULL) {
        fprintf(stderr, "Memory allocation failed in conv1d_forward\n");
        return;
    }

    // Apply the selected padding mode.
    if (conv->use_circular_pad && conv->padding > 0) {
        // Circular padding.
        circular_pad1d(input, padded_input, conv->in_channels, input_length, conv->padding);
    } else {
        // Zero padding.
        for (int c_in = 0; c_in < conv->in_channels; c_in++) {
            memcpy(&padded_input[c_in * padded_length + conv->padding], 
                   &input[c_in * input_length], 
                   input_length * sizeof(float));
        }
    }

    // Iterate over output channels.
    for (int c_out = 0; c_out < conv->out_channels; c_out++) {
        // Iterate over output sequence positions.
        for (int i = 0; i < output_length; i++) {
            output[c_out * output_length + i] = 0.0f; // Initialize to zero.
            
            // Iterate over input channels.
            for (int c_in = 0; c_in < conv->in_channels; c_in++) {
                // Perform convolution accumulation.
                for (int k = 0; k < conv->kernel_size; k++) {
                    int input_index = i * conv->stride + k;
                    output[c_out * output_length + i] += 
                        conv->weights[c_out * conv->in_channels * conv->kernel_size + 
                                     c_in * conv->kernel_size + k] 
                        * padded_input[c_in * padded_length + input_index];
                }
            }
            
            // Add bias when present.
            if (conv->bias != NULL) {
                output[c_out * output_length + i] += conv->bias[c_out];
            }
        }
    }

    edge_free(padded_input);
}

// ==================== BatchNorm Layer ====================

// BatchNorm1d forward pass.
// Input format: [channels, seq_len].
void batchnorm1d_forward(BatchNorm1DLayer *bn, float *input, float *output, int seq_len) {
    const float eps = 1e-5f; // Prevent division by zero.
    
    for (int c = 0; c < bn->num_features; c++) {
        float mean = bn->running_mean[c];
        float var = bn->running_var[c];
        float gamma = bn->weight[c];
        float beta = bn->bias[c];
        
        // Normalize and apply affine BatchNorm parameters.
        for (int i = 0; i < seq_len; i++) {
            int idx = c * seq_len + i;
            float normalized = (input[idx] - mean) / sqrtf(var + eps);
            output[idx] = gamma * normalized + beta;
        }
    }
}

// ==================== Linear Layer ====================

// Linear layer forward pass with optional bias.
void linear_forward(LinearLayer *layer, float *input, float *output) {
    for (int i = 0; i < layer->out_features; i++) {
        output[i] = 0.0f; // Initialize to zero.
        for (int j = 0; j < layer->in_features; j++) {
            output[i] += layer->weights[i * layer->in_features + j] * input[j];
        }
        
        // Add bias when present.
        if (layer->bias != NULL) {
            output[i] += layer->bias[i];
        }
    }
}

// ==================== Residual Block ====================

// Residual block forward pass.
// ResBlock = Conv1d -> BN -> ReLU -> Conv1d -> BN
void resblock_forward(ResBlock *block, float *input, float *output, int seq_len) {
    // Allocate temporary buffers.
    float *temp1 = (float*)edge_malloc(block->conv1.out_channels * seq_len * sizeof(float));
    float *temp2 = (float*)edge_malloc(block->conv1.out_channels * seq_len * sizeof(float));
    
    if (temp1 == NULL || temp2 == NULL) {
        fprintf(stderr, "Memory allocation failed in resblock_forward\n");
        if (temp1) edge_free(temp1);
        if (temp2) edge_free(temp2);
        return;
    }
    
    // Conv1 -> BN1 -> ReLU
    conv1d_forward(&block->conv1, input, seq_len, temp1);
    batchnorm1d_forward(&block->bn1, temp1, temp2, seq_len);
    relu_inplace(temp2, block->conv1.out_channels * seq_len);
    
    // Conv2 -> BN2
    conv1d_forward(&block->conv2, temp2, seq_len, temp1);
    batchnorm1d_forward(&block->bn2, temp1, output, seq_len);
    
    // Residual connection: output += input.
    for (int i = 0; i < block->conv1.out_channels * seq_len; i++) {
        output[i] += input[i];
    }
    
    edge_free(temp1);
    edge_free(temp2);
}

// ==================== Full Model Forward Pass ====================

void patch_resnet_forward(PatchResNetModel *model, 
                         float *input, 
                         float *output, 
                         float *features) {
    // Input: [INPUT_CHANNELS, INPUT_SEQ_LEN].
    // output stores class logits; features stores the concatenated projection vector.
    
    int n_vars = 0;
    int input_size = INPUT_CHANNELS * INPUT_SEQ_LEN;  // Flattened input size.
    int feature_dim = 10 * D_MODEL;                    // Flattened feature dimension.
    int concat_dim = input_size + feature_dim;         // Total concatenated dimension.
    
    // Stage 1: Patch embedding.
    float *patch_out = (float*)edge_malloc(NUM_PATCHES * D_MODEL * sizeof(float));
    if (!patch_out) {
        fprintf(stderr, "Failed to allocate patch_out\n");
        return;
    }
    
    patch_embedding_forward(&model->patch_emb, input, patch_out, 
                           1, INPUT_CHANNELS, INPUT_SEQ_LEN, &n_vars);
    
    // Stage 2: BatchNorm3.
    float *bn3_out = (float*)edge_malloc(NUM_PATCHES * D_MODEL * sizeof(float));
    if (!bn3_out) {
        edge_free(patch_out);
        fprintf(stderr, "Failed to allocate bn3_out\n");
        return;
    }
    
    batchnorm1d_forward(&model->bn3, patch_out, bn3_out, D_MODEL);
    edge_free(patch_out);
    
    // Stage 3: Conv1.
    float *conv1_out = (float*)edge_malloc(OUT_CHANNELS * D_MODEL * sizeof(float));
    if (!conv1_out) {
        edge_free(bn3_out);
        fprintf(stderr, "Failed to allocate conv1_out\n");
        return;
    }
    
    conv1d_forward(&model->conv1, bn3_out, D_MODEL, conv1_out);
    edge_free(bn3_out);
    
    // Stage 4: BN1 -> ReLU.
    float *bn1_out = (float*)edge_malloc(OUT_CHANNELS * D_MODEL * sizeof(float));
    if (!bn1_out) {
        edge_free(conv1_out);
        fprintf(stderr, "Failed to allocate bn1_out\n");
        return;
    }
    
    batchnorm1d_forward(&model->bn1, conv1_out, bn1_out, D_MODEL);
    edge_free(conv1_out);
    relu_inplace(bn1_out, OUT_CHANNELS * D_MODEL);
    
    // Stage 5: ResBlock1.
    float *res1_out = (float*)edge_malloc(OUT_CHANNELS * D_MODEL * sizeof(float));
    if (!res1_out) {
        edge_free(bn1_out);
        fprintf(stderr, "Failed to allocate res1_out\n");
        return;
    }
    
    resblock_forward(&model->resblock1, bn1_out, res1_out, D_MODEL);
    edge_free(bn1_out);
    relu_inplace(res1_out, OUT_CHANNELS * D_MODEL);
    
    // Stage 6: ResBlock2.
    float *res2_out = (float*)edge_malloc(OUT_CHANNELS * D_MODEL * sizeof(float));
    if (!res2_out) {
        edge_free(res1_out);
        fprintf(stderr, "Failed to allocate res2_out\n");
        return;
    }
    
    resblock_forward(&model->resblock2, res1_out, res2_out, D_MODEL);
    edge_free(res1_out);
    relu_inplace(res2_out, OUT_CHANNELS * D_MODEL);
    
    // Stage 7: ResBlock3.
    float *res3_out = (float*)edge_malloc(OUT_CHANNELS * D_MODEL * sizeof(float));
    if (!res3_out) {
        edge_free(res2_out);
        fprintf(stderr, "Failed to allocate res3_out\n");
        return;
    }
    
    resblock_forward(&model->resblock3, res2_out, res3_out, D_MODEL);
    edge_free(res2_out);
    relu_inplace(res3_out, OUT_CHANNELS * D_MODEL);
    
    // Stage 8: Output convolution.
    float *conv_out_result = (float*)edge_malloc(10 * D_MODEL * sizeof(float));
    if (!conv_out_result) {
        edge_free(res3_out);
        fprintf(stderr, "Failed to allocate conv_out_result\n");
        return;
    }
    
    conv1d_forward(&model->conv_out, res3_out, D_MODEL, conv_out_result);
    edge_free(res3_out);
    
    // Stage 9: BN2.
    float *bn2_out = (float*)edge_malloc(10 * D_MODEL * sizeof(float));
    if (!bn2_out) {
        edge_free(conv_out_result);
        fprintf(stderr, "Failed to allocate bn2_out\n");
        return;
    }
    
    batchnorm1d_forward(&model->bn2, conv_out_result, bn2_out, D_MODEL);
    edge_free(conv_out_result);
    
    // Stage 10: Flatten.
    float *flatten_out = (float*)edge_malloc(feature_dim * sizeof(float));
    if (!flatten_out) {
        edge_free(bn2_out);
        fprintf(stderr, "Failed to allocate flatten_out\n");
        return;
    }
    
    flatten(bn2_out, flatten_out, 10, D_MODEL);
    edge_free(bn2_out);
    
    // Stage 11: Feature concatenation.
    // Concatenate: concat_features = [input | flatten_out].
    // Matches PyTorch: torch.cat([train_x, reps], dim=1).
    float *concat_features = (float*)edge_malloc(concat_dim * sizeof(float));
    if (!concat_features) {
        edge_free(flatten_out);
        fprintf(stderr, "Failed to allocate concat_features\n");
        return;
    }
    
    // Copy raw normalized input.
    memcpy(concat_features, input, input_size * sizeof(float));
    
    // Append flattened learned features.
    memcpy(concat_features + input_size, flatten_out, feature_dim * sizeof(float));
    
    edge_free(flatten_out);
   
    memcpy(features, concat_features, concat_dim * sizeof(float));
    
    
    // Stage 12: FC1 linear layer.
    float *temp_output = (float*)edge_malloc(NUM_CLASSES * sizeof(float));
    if (!temp_output) {
        edge_free(concat_features);
        fprintf(stderr, "Failed to allocate temp_output\n");
        return;
    }
    
    linear_forward(&model->fc1, concat_features, temp_output);
    memcpy(output, temp_output, NUM_CLASSES * sizeof(float));
    
    edge_free(concat_features);
    edge_free(temp_output);
}





// ==================== Model Initialization ====================

// Initialize model metadata. Weights and biases are assigned externally.
void init_patch_resnet_model(PatchResNetModel *model) {
    // Patch embedding parameters.
    // Parameters are provided by model_params.h:
    // - patch_embedding_position_embedding_pe (positional encoding).
    // - patch_embedding_value_embedding_tokenConv_weight (token convolution weights).
    // Assign these pointers after init_patch_resnet_model(), or adapt this function to accept them.
    // Example: init_patch_embedding(&model->patch_emb, D_MODEL, PATCH_LEN, STRIDE, 0.0f, 
    //                            patch_embedding_position_embedding_pe, 
    //                            patch_embedding_value_embedding_tokenConv_weight);
    
    // Patch embedding data pointers are assigned externally.
    model->patch_emb.d_model = D_MODEL;
    model->patch_emb.patch_len = PATCH_LEN;
    model->patch_emb.stride = STRIDE;
    model->patch_emb.dropout_rate = 0.0f;
    model->patch_emb.value_embedding.c_in = PATCH_LEN;
    model->patch_emb.value_embedding.d_model = D_MODEL;
    model->patch_emb.value_embedding.token_conv.in_channels = PATCH_LEN;
    model->patch_emb.value_embedding.token_conv.out_channels = D_MODEL;
    model->patch_emb.value_embedding.token_conv.kernel_size = 3;
    model->patch_emb.value_embedding.token_conv.stride = 1;
    model->patch_emb.value_embedding.token_conv.padding = 1;
    model->patch_emb.value_embedding.token_conv.use_circular_pad = 1;
    model->patch_emb.value_embedding.token_conv.bias = NULL; // No bias.

    model->patch_emb.pos_embedding.d_model = D_MODEL;
    model->patch_emb.pos_embedding.max_len = MAX_LEN;
    
    // Conv1 parameters.
    model->conv1.in_channels = NUM_PATCHES;
    model->conv1.out_channels = OUT_CHANNELS;
    model->conv1.kernel_size = 3;
    model->conv1.stride = 1;
    model->conv1.padding = 1;
    model->conv1.bias = NULL; // No bias.
    model->conv1.use_circular_pad = 0; // Use zero padding.
    
    // BatchNorm parameters.
    model->bn1.num_features = OUT_CHANNELS;
    model->bn2.num_features = 10;
    model->bn3.num_features = NUM_PATCHES;
    
    // ResBlock1
    model->resblock1.conv1.in_channels = OUT_CHANNELS;
    model->resblock1.conv1.out_channels = OUT_CHANNELS;
    model->resblock1.conv1.kernel_size = 3;
    model->resblock1.conv1.stride = 1;
    model->resblock1.conv1.padding = 1;
    model->resblock1.conv1.bias = NULL;
    model->resblock1.conv1.use_circular_pad = 0;
    
    model->resblock1.bn1.num_features = OUT_CHANNELS;
    
    model->resblock1.conv2.in_channels = OUT_CHANNELS;
    model->resblock1.conv2.out_channels = OUT_CHANNELS;
    model->resblock1.conv2.kernel_size = 3;
    model->resblock1.conv2.stride = 1;
    model->resblock1.conv2.padding = 1;
    model->resblock1.conv2.bias = NULL;
    model->resblock1.conv2.use_circular_pad = 0;
    
    model->resblock1.bn2.num_features = OUT_CHANNELS;
    
    // ResBlock2 uses the same layout as ResBlock1.
    model->resblock2.conv1.in_channels = OUT_CHANNELS;
    model->resblock2.conv1.out_channels = OUT_CHANNELS;
    model->resblock2.conv1.kernel_size = 3;
    model->resblock2.conv1.stride = 1;
    model->resblock2.conv1.padding = 1;
    model->resblock2.conv1.bias = NULL;
    model->resblock2.conv1.use_circular_pad = 0;
    
    model->resblock2.bn1.num_features = OUT_CHANNELS;
    
    model->resblock2.conv2.in_channels = OUT_CHANNELS;
    model->resblock2.conv2.out_channels = OUT_CHANNELS;
    model->resblock2.conv2.kernel_size = 3;
    model->resblock2.conv2.stride = 1;
    model->resblock2.conv2.padding = 1;
    model->resblock2.conv2.bias = NULL;
    model->resblock2.conv2.use_circular_pad = 0;
    
    model->resblock2.bn2.num_features = OUT_CHANNELS;
    
    // ResBlock3 uses the same layout as ResBlock1 and ResBlock2.
    model->resblock3.conv1.in_channels = OUT_CHANNELS;
    model->resblock3.conv1.out_channels = OUT_CHANNELS;
    model->resblock3.conv1.kernel_size = 3;
    model->resblock3.conv1.stride = 1;
    model->resblock3.conv1.padding = 1;
    model->resblock3.conv1.bias = NULL;
    model->resblock3.conv1.use_circular_pad = 0;
    
    model->resblock3.bn1.num_features = OUT_CHANNELS;
    
    model->resblock3.conv2.in_channels = OUT_CHANNELS;
    model->resblock3.conv2.out_channels = OUT_CHANNELS;
    model->resblock3.conv2.kernel_size = 3;
    model->resblock3.conv2.stride = 1;
    model->resblock3.conv2.padding = 1;
    model->resblock3.conv2.bias = NULL;
    model->resblock3.conv2.use_circular_pad = 0;
    
    model->resblock3.bn2.num_features = OUT_CHANNELS;
    
    // Output convolution parameters.
    model->conv_out.in_channels = OUT_CHANNELS;
    model->conv_out.out_channels = 10;
    model->conv_out.kernel_size = 3;
    model->conv_out.stride = 1;
    model->conv_out.padding = 1;
    model->conv_out.bias = NULL;
    model->conv_out.use_circular_pad = 0;
    
    // Fully connected layer parameters.
    model->fc1.in_features = 11 * D_MODEL; // 10 * 150 = 1500
    model->fc1.out_features = NUM_CLASSES;
    // Weight and bias pointers must be assigned externally.
    // Example: model->conv1.weights = conv1_weight_array;
}



