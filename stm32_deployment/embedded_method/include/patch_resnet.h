#ifndef PATCH_RESNET_H
#define PATCH_RESNET_H

#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "arm_math.h"
#include "edge_config.h"

/*
 * STM32 implementation of the patch-residual NILM feature extractor and the
 * local analytical update utilities.
 */

/** One-dimensional convolution layer descriptor. */
typedef struct {
    float *weights;
    float *bias;
    int in_channels;
    int out_channels;
    int kernel_size;
    int stride;
    int padding;
    int use_circular_pad;
} Conv1DLayer;

/** BatchNorm1d layer descriptor used in inference mode. */
typedef struct {
    float *weight;
    float *bias;
    float *running_mean;
    float *running_var;
    int num_features;
} BatchNorm1DLayer;

/** Fully connected layer descriptor. */
typedef struct {
    float *weights;
    float *bias;
    int in_features;
    int out_features;
} LinearLayer;

/** Saved sinusoidal positional embedding descriptor. */
typedef struct {
    float *pe;
    int d_model;
    int max_len;
} PositionalEmbedding;

/** Token embedding layer implemented as circular 1D convolution. */
typedef struct {
    Conv1DLayer token_conv;
    int c_in;
    int d_model;
} TokenEmbedding;

/** Patch embedding block for one-dimensional waveforms. */
typedef struct {
    int patch_len;
    int stride;
    int d_model;
    TokenEmbedding value_embedding;
    PositionalEmbedding pos_embedding;
    float dropout_rate;
} PatchEmbedding;

/** Two-convolution residual block descriptor. */
typedef struct {
    Conv1DLayer conv1;
    BatchNorm1DLayer bn1;
    Conv1DLayer conv2;
    BatchNorm1DLayer bn2;
} ResBlock;

/** Complete embedded patch-residual feature extractor. */
typedef struct {
    PatchEmbedding patch_emb;
    BatchNorm1DLayer bn3;
    Conv1DLayer conv1;
    BatchNorm1DLayer bn1;
    ResBlock resblock1;
    ResBlock resblock2;
    ResBlock resblock3;
    Conv1DLayer conv_out;
    BatchNorm1DLayer bn2;
    LinearLayer fc1;
} PatchResNetModel;

/**
 * @brief Apply ReLU activation in place.
 * @param data Input/output array.
 * @param length Number of elements.
 */
void relu_inplace(float *data, int length);

/**
 * @brief Copy a [channels, length] tensor into a flat vector.
 * @param input Input tensor in row-major channel-first layout.
 * @param output Output flat vector with channels * length elements.
 * @param channels Number of channels.
 * @param length Number of values per channel.
 */
void flatten(float *input, float *output, int channels, int length);

/**
 * @brief Apply circular 1D padding.
 * @param input Input tensor [channels, length].
 * @param output Output tensor [channels, length + 2 * padding].
 * @param channels Number of channels.
 * @param length Input sequence length.
 * @param padding Number of values padded on each side.
 */
void circular_pad1d(float *input, float *output, int channels, int length, int padding);

/**
 * @brief Normalize one waveform sample in place to [-1, 1].
 * @param data Input/output waveform array.
 * @param length Waveform length.
 * @param min_val Optional output minimum value.
 * @param max_val Optional output maximum value.
 * @return 0 on normal normalization, -1 when the sample has no dynamic range.
 */
int normalize_sample_inplace(float *data, int length, float *min_val, float *max_val);

/**
 * @brief Compute X * X^T + rg * I.
 * @param X Row-major input matrix [batch_size, features].
 * @param batch_size Number of samples.
 * @param features Number of feature columns.
 * @param rg Regularization coefficient.
 * @param output Output matrix [batch_size, batch_size].
 */
void matmul_xxt_with_reg(float *X, int batch_size, int features, float rg, float *output);

/**
 * @brief Compute X^T * Y with optional label-vector handling.
 * @param X Row-major input matrix [batch_size, features].
 * @param Y Label matrix [batch_size, out_dim] or label vector [batch_size].
 * @param batch_size Number of samples.
 * @param features Number of feature columns.
 * @param out_dim Number of output classes.
 * @param is_label_vector 1 when Y stores class indices, otherwise 0.
 * @param out Output matrix [features, out_dim].
 */
void matmul_xt_y(
    float *X,
    float *Y,
    int batch_size,
    int features,
    int out_dim,
    int is_label_vector,
    float *out
);

/**
 * @brief Invert a symmetric positive definite matrix using Cholesky inversion.
 * @param A Input matrix [n, n], row-major.
 * @param n Matrix dimension.
 * @param A_inv Output inverse matrix [n, n], row-major.
 * @return 0 on success, -1 on decomposition failure.
 */
int invert_spd_matrix(float *A, int n, float *A_inv);

/**
 * @brief Invert a matrix with double-precision Gauss-Jordan elimination.
 * @param A Input matrix [n, n], modified in place.
 * @param n Matrix dimension.
 * @param A_inv Output inverse matrix [n, n].
 * @return 0 on success, -1 when the matrix is singular.
 */
int invert_matrix_gauss_jordan_double(float *A, int n, float *A_inv);

/**
 * @brief Invert a matrix with float-precision Gauss-Jordan elimination.
 * @param A Input matrix [n, n], modified in place.
 * @param n Matrix dimension.
 * @param A_inv Output inverse matrix [n, n].
 * @return 0 on success, -1 when the matrix is singular.
 */
int invert_matrix_gauss_jordan_float(float *A, int n, float *A_inv);

/**
 * @brief Compute local analytical weights with a regularized Gram solve.
 * @param X Row-major projection feature matrix [batch_size, features].
 * @param Y Label matrix [batch_size, out_dim] or label vector [batch_size].
 * @param batch_size Number of samples.
 * @param features Number of feature columns.
 * @param out_dim Number of output classes.
 * @param is_label_vector 1 when Y stores class indices, otherwise 0.
 * @param rg Regularization coefficient.
 * @param gram Output Gram matrix [batch_size, batch_size].
 * @param gram_inv Output inverse Gram matrix [batch_size, batch_size].
 * @param alpha Output intermediate matrix [batch_size, out_dim].
 * @param W Output local analytical weight matrix [features, out_dim].
 * @return 0 on success, -1 on invalid values or inversion failure.
 */
int gram_with_reg_and_inv(
    float *X,
    float *Y,
    int batch_size,
    int features,
    int out_dim,
    int is_label_vector,
    float rg,
    float *gram,
    float *gram_inv,
    float *alpha,
    float *W
);

/**
 * @brief CMSIS-DSP variant of the regularized Gram solve.
 * @param X Row-major projection feature matrix [batch_size, features].
 * @param Y Label matrix [batch_size, out_dim] or label vector [batch_size].
 * @param batch_size Number of samples.
 * @param features Number of feature columns.
 * @param out_dim Number of output classes.
 * @param is_label_vector 1 when Y stores class indices, otherwise 0.
 * @param rg Regularization coefficient.
 * @param gram Output Gram matrix [batch_size, batch_size].
 * @param gram_inv Output inverse Gram matrix [batch_size, batch_size].
 * @param alpha Output intermediate matrix [batch_size, out_dim].
 * @param W Output local analytical weight matrix [features, out_dim].
 */
void gram_with_reg_and_inv_cmsis(
    float *X,
    float *Y,
    int batch_size,
    int features,
    int out_dim,
    int is_label_vector,
    float rg,
    float *gram,
    float *gram_inv,
    float *alpha,
    float *W
);

/**
 * @brief Initialize a positional embedding descriptor from saved parameters.
 * @param pe Positional embedding descriptor to initialize.
 * @param d_model Embedding dimension.
 * @param max_len Maximum sequence length.
 * @param saved_pe Pointer to saved positional encoding values.
 */
void init_positional_embedding(PositionalEmbedding *pe, int d_model, int max_len, float *saved_pe);

/**
 * @brief Copy positional encodings for the requested sequence length.
 * @param pe Initialized positional embedding descriptor.
 * @param output Output buffer [seq_len, d_model].
 * @param seq_len Number of positions to copy.
 */
void positional_embedding_forward(PositionalEmbedding *pe, float *output, int seq_len);

/**
 * @brief Initialize token embedding metadata.
 * @param te Token embedding descriptor to initialize.
 * @param c_in Input patch length.
 * @param d_model Output embedding dimension.
 */
void init_token_embedding(TokenEmbedding *te, int c_in, int d_model);

/**
 * @brief Run token embedding forward pass.
 * @param te Initialized token embedding descriptor.
 * @param input Input tensor [batch_size, seq_len, c_in].
 * @param output Output tensor [batch_size, seq_len, d_model].
 * @param batch_size Batch size.
 * @param seq_len Number of patch tokens.
 */
void token_embedding_forward(TokenEmbedding *te, float *input, float *output, int batch_size, int seq_len);

/**
 * @brief Initialize patch embedding metadata and parameter pointers.
 * @param pe Patch embedding descriptor to initialize.
 * @param d_model Patch embedding dimension.
 * @param patch_len Number of waveform points per patch.
 * @param stride Patch extraction stride.
 * @param dropout Stored dropout value; unused during inference.
 * @param saved_pos_emb Saved positional encoding pointer.
 * @param token_conv_weights Token convolution weight pointer.
 */
void init_patch_embedding(
    PatchEmbedding *pe,
    int d_model,
    int patch_len,
    int stride,
    float dropout,
    float *saved_pos_emb,
    float *token_conv_weights
);

/**
 * @brief Run patch embedding forward pass for a single waveform sample.
 * @param pe Initialized patch embedding descriptor.
 * @param input Input waveform [input_channels, input_length].
 * @param output Output patch tokens [num_patches, d_model].
 * @param batch_size Kept for API compatibility; expected to be 1.
 * @param input_channels Number of input channels.
 * @param input_length Input waveform length.
 * @param n_vars Optional output number of variables/channels.
 */
void patch_embedding_forward(
    PatchEmbedding *pe,
    float *input,
    float *output,
    int batch_size,
    int input_channels,
    int input_length,
    int *n_vars
);

/**
 * @brief Free patch embedding resources when owned by the descriptor.
 * @param pe Patch embedding descriptor.
 */
void free_patch_embedding(PatchEmbedding *pe);

/**
 * @brief Run Conv1D forward pass.
 * @param conv Initialized convolution descriptor.
 * @param input Input tensor [in_channels, input_length].
 * @param input_length Input sequence length.
 * @param output Output tensor [out_channels, output_length].
 */
void conv1d_forward(Conv1DLayer *conv, float *input, int input_length, float *output);

/**
 * @brief Run BatchNorm1d inference forward pass.
 * @param bn Initialized BatchNorm descriptor.
 * @param input Input tensor [channels, seq_len].
 * @param output Output tensor [channels, seq_len].
 * @param seq_len Sequence length.
 */
void batchnorm1d_forward(BatchNorm1DLayer *bn, float *input, float *output, int seq_len);

/**
 * @brief Run linear layer forward pass.
 * @param layer Initialized linear layer descriptor.
 * @param input Input vector [in_features].
 * @param output Output vector [out_features].
 */
void linear_forward(LinearLayer *layer, float *input, float *output);

/**
 * @brief Run one residual block forward pass.
 * @param block Initialized residual block descriptor.
 * @param input Input tensor [channels, seq_len].
 * @param output Output tensor [channels, seq_len].
 * @param seq_len Sequence length.
 */
void resblock_forward(ResBlock *block, float *input, float *output, int seq_len);

/**
 * @brief Run the full patch-residual model forward pass.
 * @param model Initialized model with parameter pointers assigned.
 * @param input Normalized waveform [INPUT_SEQ_LEN].
 * @param output Output logits [NUM_CLASSES].
 * @param features Output projection vector [PROJECTION_DIM].
 */
void patch_resnet_forward(PatchResNetModel *model, float *input, float *output, float *features);

/**
 * @brief Initialize model layer metadata.
 * @param model Model descriptor to initialize. Parameter pointers are assigned
 * externally by edge_training.c.
 */
void init_patch_resnet_model(PatchResNetModel *model);

#endif
