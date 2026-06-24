#include "model_params.h"

/*
 * Placeholder model parameters.
 *
 * Replace these zero-initialized arrays with parameters exported from your own
 * trained PyTorch checkpoint. The complete deployment weights used in the paper
 * are not included in this public release.
 */

const float patch_embedding_value_embedding_tokenConv_weight[D_MODEL * PATCH_LEN * 3] = {0};
const float patch_embedding_position_embedding_pe[NUM_PATCHES * D_MODEL] = {0};

const float conv1_weight[OUT_CHANNELS * NUM_PATCHES * 3] = {0};

const float bn1_weight[OUT_CHANNELS] = {0};
const float bn1_bias[OUT_CHANNELS] = {0};
const float bn1_running_mean[OUT_CHANNELS] = {0};
const float bn1_running_var[OUT_CHANNELS] = {0};

const float bn3_weight[NUM_PATCHES] = {0};
const float bn3_bias[NUM_PATCHES] = {0};
const float bn3_running_mean[NUM_PATCHES] = {0};
const float bn3_running_var[NUM_PATCHES] = {0};

const float resblock1_0_weight[OUT_CHANNELS * OUT_CHANNELS * 3] = {0};
const float resblock1_1_weight[OUT_CHANNELS] = {0};
const float resblock1_1_bias[OUT_CHANNELS] = {0};
const float resblock1_1_running_mean[OUT_CHANNELS] = {0};
const float resblock1_1_running_var[OUT_CHANNELS] = {0};
const float resblock1_3_weight[OUT_CHANNELS * OUT_CHANNELS * 3] = {0};
const float resblock1_4_weight[OUT_CHANNELS] = {0};
const float resblock1_4_bias[OUT_CHANNELS] = {0};
const float resblock1_4_running_mean[OUT_CHANNELS] = {0};
const float resblock1_4_running_var[OUT_CHANNELS] = {0};

const float resblock2_0_weight[OUT_CHANNELS * OUT_CHANNELS * 3] = {0};
const float resblock2_1_weight[OUT_CHANNELS] = {0};
const float resblock2_1_bias[OUT_CHANNELS] = {0};
const float resblock2_1_running_mean[OUT_CHANNELS] = {0};
const float resblock2_1_running_var[OUT_CHANNELS] = {0};
const float resblock2_3_weight[OUT_CHANNELS * OUT_CHANNELS * 3] = {0};
const float resblock2_4_weight[OUT_CHANNELS] = {0};
const float resblock2_4_bias[OUT_CHANNELS] = {0};
const float resblock2_4_running_mean[OUT_CHANNELS] = {0};
const float resblock2_4_running_var[OUT_CHANNELS] = {0};

const float resblock3_0_weight[OUT_CHANNELS * OUT_CHANNELS * 3] = {0};
const float resblock3_1_weight[OUT_CHANNELS] = {0};
const float resblock3_1_bias[OUT_CHANNELS] = {0};
const float resblock3_1_running_mean[OUT_CHANNELS] = {0};
const float resblock3_1_running_var[OUT_CHANNELS] = {0};
const float resblock3_3_weight[OUT_CHANNELS * OUT_CHANNELS * 3] = {0};
const float resblock3_4_weight[OUT_CHANNELS] = {0};
const float resblock3_4_bias[OUT_CHANNELS] = {0};
const float resblock3_4_running_mean[OUT_CHANNELS] = {0};
const float resblock3_4_running_var[OUT_CHANNELS] = {0};

const float cov_out_weight[CONV_OUT_CHANNELS * OUT_CHANNELS * 3] = {0};

const float bn2_weight[CONV_OUT_CHANNELS] = {0};
const float bn2_bias[CONV_OUT_CHANNELS] = {0};
const float bn2_running_mean[CONV_OUT_CHANNELS] = {0};
const float bn2_running_var[CONV_OUT_CHANNELS] = {0};

const float fc_weight[NUM_CLASSES * PROJECTION_DIM] = {0};
