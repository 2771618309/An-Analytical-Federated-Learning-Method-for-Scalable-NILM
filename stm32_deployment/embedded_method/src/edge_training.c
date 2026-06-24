#include "edge_config.h"

#include <stdio.h>
#include <string.h>

#include "edge_memory.h"
#include "model_params.h"
#include "patch_resnet.h"
#include "serial_protocol.h"

volatile int g_edge_abort_flag = 0;

/**
 * @brief Attach exported model parameter arrays to the model descriptor.
 * @param model Model descriptor initialized by init_patch_resnet_model().
 *
 * The parameter arrays are declared in model_params.h. The public release uses
 * placeholder arrays; real deployments should replace them with exported
 * checkpoint parameters.
 */
static void assign_model_parameters(PatchResNetModel *model)
{
    model->patch_emb.value_embedding.token_conv.weights =
        (float *)patch_embedding_value_embedding_tokenConv_weight;
    model->patch_emb.pos_embedding.pe = (float *)patch_embedding_position_embedding_pe;

    model->conv1.weights = (float *)conv1_weight;

    model->bn1.weight = (float *)bn1_weight;
    model->bn1.bias = (float *)bn1_bias;
    model->bn1.running_mean = (float *)bn1_running_mean;
    model->bn1.running_var = (float *)bn1_running_var;

    model->bn3.weight = (float *)bn3_weight;
    model->bn3.bias = (float *)bn3_bias;
    model->bn3.running_mean = (float *)bn3_running_mean;
    model->bn3.running_var = (float *)bn3_running_var;

    model->resblock1.conv1.weights = (float *)resblock1_0_weight;
    model->resblock1.conv2.weights = (float *)resblock1_3_weight;
    model->resblock1.bn1.weight = (float *)resblock1_1_weight;
    model->resblock1.bn1.bias = (float *)resblock1_1_bias;
    model->resblock1.bn1.running_mean = (float *)resblock1_1_running_mean;
    model->resblock1.bn1.running_var = (float *)resblock1_1_running_var;
    model->resblock1.bn2.weight = (float *)resblock1_4_weight;
    model->resblock1.bn2.bias = (float *)resblock1_4_bias;
    model->resblock1.bn2.running_mean = (float *)resblock1_4_running_mean;
    model->resblock1.bn2.running_var = (float *)resblock1_4_running_var;

    model->resblock2.conv1.weights = (float *)resblock2_0_weight;
    model->resblock2.conv2.weights = (float *)resblock2_3_weight;
    model->resblock2.bn1.weight = (float *)resblock2_1_weight;
    model->resblock2.bn1.bias = (float *)resblock2_1_bias;
    model->resblock2.bn1.running_mean = (float *)resblock2_1_running_mean;
    model->resblock2.bn1.running_var = (float *)resblock2_1_running_var;
    model->resblock2.bn2.weight = (float *)resblock2_4_weight;
    model->resblock2.bn2.bias = (float *)resblock2_4_bias;
    model->resblock2.bn2.running_mean = (float *)resblock2_4_running_mean;
    model->resblock2.bn2.running_var = (float *)resblock2_4_running_var;

    model->resblock3.conv1.weights = (float *)resblock3_0_weight;
    model->resblock3.conv2.weights = (float *)resblock3_3_weight;
    model->resblock3.bn1.weight = (float *)resblock3_1_weight;
    model->resblock3.bn1.bias = (float *)resblock3_1_bias;
    model->resblock3.bn1.running_mean = (float *)resblock3_1_running_mean;
    model->resblock3.bn1.running_var = (float *)resblock3_1_running_var;
    model->resblock3.bn2.weight = (float *)resblock3_4_weight;
    model->resblock3.bn2.bias = (float *)resblock3_4_bias;
    model->resblock3.bn2.running_mean = (float *)resblock3_4_running_mean;
    model->resblock3.bn2.running_var = (float *)resblock3_4_running_var;

    model->conv_out.weights = (float *)cov_out_weight;

    model->bn2.weight = (float *)bn2_weight;
    model->bn2.bias = (float *)bn2_bias;
    model->bn2.running_mean = (float *)bn2_running_mean;
    model->bn2.running_var = (float *)bn2_running_var;

    model->fc1.weights = (float *)fc_weight;
    model->fc1.bias = NULL;
}

/**
 * @brief Run the STM32-side analytical local update flow.
 * @param client_id Logical client identifier. Kept for host-side bookkeeping.
 * @param data Input matrix containing waveform values and one label column.
 * @param samples Number of local samples in data.
 * @param signal_features Number of columns in data.
 * @param scale Quantization scale used to convert int16 values to float.
 * @return 0 on success, -1 on invalid input, allocation failure, or solve failure.
 *
 * Flow:
 * 1. Initialize the embedded patch-residual model.
 * 2. Normalize each waveform sample to [-1, 1].
 * 3. Extract projection-branch features [raw waveform | learned feature].
 * 4. Compute the local analytical classifier W.
 * 5. Upload W and block-wise upper-triangular X^T X matrices.
 */
int edge_start_training(
    int client_id,
    int16_t data[][MAX_FEATURES],
    int samples,
    int signal_features,
    float scale
)
{
    (void)client_id;

    if (data == NULL || samples <= 0 || samples > MAX_SAMPLES) {
        return -1;
    }
    if (signal_features < INPUT_SEQ_LEN + 1) {
        return -1;
    }

    g_edge_abort_flag = 0;
    edge_heap_reset();

    float value_scale = (scale > 0.0f) ? scale : CLIENT_SCALE;
    float min_val = 0.0f;
    float max_val = 0.0f;
    int status = -1;

    PatchResNetModel model;
    init_patch_resnet_model(&model);
    assign_model_parameters(&model);

    float *xtx_block_buffer =
        (float *)edge_malloc(XTX_BLOCK_SIZE * XTX_BLOCK_SIZE * sizeof(float));
    float *all_features =
        (float *)edge_malloc(samples * PROJECTION_DIM * sizeof(float));
    float *input = (float *)edge_malloc(INPUT_SEQ_LEN * sizeof(float));
    float *output = (float *)edge_malloc(NUM_CLASSES * sizeof(float));
    float *features = (float *)edge_malloc(PROJECTION_DIM * sizeof(float));
    float *W = (float *)edge_malloc(PROJECTION_DIM * NUM_CLASSES * sizeof(float));
    float *alpha = (float *)edge_malloc(samples * NUM_CLASSES * sizeof(float));
    float *all_labels = (float *)edge_malloc(samples * sizeof(float));
    float *gram = (float *)edge_malloc(samples * samples * sizeof(float));
    float *gram_inv = (float *)edge_malloc(samples * samples * sizeof(float));

    if (
        xtx_block_buffer == NULL ||
        all_features == NULL ||
        input == NULL ||
        output == NULL ||
        features == NULL ||
        W == NULL ||
        alpha == NULL ||
        all_labels == NULL ||
        gram == NULL ||
        gram_inv == NULL
    ) {
        goto cleanup;
    }

    for (int n = 0; n < samples; n++) {
        if (g_edge_abort_flag) {
            goto cleanup;
        }

        for (int i = 0; i < INPUT_SEQ_LEN; i++) {
            input[i] = (float)data[n][i] / value_scale;
        }
        all_labels[n] = (float)data[n][LABEL_INDEX] / value_scale;

        normalize_sample_inplace(input, INPUT_SEQ_LEN, &min_val, &max_val);
        patch_resnet_forward(&model, input, output, features);

        memcpy(
            all_features + n * PROJECTION_DIM,
            features,
            PROJECTION_DIM * sizeof(float)
        );
    }

    if (
        gram_with_reg_and_inv(
            all_features,
            all_labels,
            samples,
            PROJECTION_DIM,
            NUM_CLASSES,
            1,
            LOCAL_REGULARIZATION,
            gram,
            gram_inv,
            alpha,
            W
        ) != 0
    ) {
        goto cleanup;
    }

    edge_send_training_results(W, PROJECTION_DIM, NUM_CLASSES);
    edge_compute_and_upload_xtx_streaming(
        all_features,
        samples,
        PROJECTION_DIM,
        XTX_BLOCK_SIZE,
        xtx_block_buffer
    );

    status = 0;

cleanup:
    if (xtx_block_buffer) edge_free(xtx_block_buffer);
    if (all_features) edge_free(all_features);
    if (input) edge_free(input);
    if (output) edge_free(output);
    if (features) edge_free(features);
    if (W) edge_free(W);
    if (alpha) edge_free(alpha);
    if (all_labels) edge_free(all_labels);
    if (gram) edge_free(gram);
    if (gram_inv) edge_free(gram_inv);

    if (g_edge_abort_flag) {
        edge_heap_reset();
        g_edge_abort_flag = 0;
    }

    return status;
}
