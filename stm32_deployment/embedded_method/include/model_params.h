#ifndef MODEL_PARAMS_H
#define MODEL_PARAMS_H

#include "edge_config.h"

/*
 * Model parameter declarations.
 *
 * This release does not include the trained deployment weights. The arrays are
 * declared here so the embedded method code is complete. Replace the template
 * definitions in params/model_params_template.c with parameters exported from
 * your own PyTorch checkpoint before deployment.
 */

extern const float patch_embedding_value_embedding_tokenConv_weight[D_MODEL * PATCH_LEN * 3];
extern const float patch_embedding_position_embedding_pe[NUM_PATCHES * D_MODEL];

extern const float conv1_weight[OUT_CHANNELS * NUM_PATCHES * 3];

extern const float bn1_weight[OUT_CHANNELS];
extern const float bn1_bias[OUT_CHANNELS];
extern const float bn1_running_mean[OUT_CHANNELS];
extern const float bn1_running_var[OUT_CHANNELS];

extern const float bn3_weight[NUM_PATCHES];
extern const float bn3_bias[NUM_PATCHES];
extern const float bn3_running_mean[NUM_PATCHES];
extern const float bn3_running_var[NUM_PATCHES];

extern const float resblock1_0_weight[OUT_CHANNELS * OUT_CHANNELS * 3];
extern const float resblock1_1_weight[OUT_CHANNELS];
extern const float resblock1_1_bias[OUT_CHANNELS];
extern const float resblock1_1_running_mean[OUT_CHANNELS];
extern const float resblock1_1_running_var[OUT_CHANNELS];
extern const float resblock1_3_weight[OUT_CHANNELS * OUT_CHANNELS * 3];
extern const float resblock1_4_weight[OUT_CHANNELS];
extern const float resblock1_4_bias[OUT_CHANNELS];
extern const float resblock1_4_running_mean[OUT_CHANNELS];
extern const float resblock1_4_running_var[OUT_CHANNELS];

extern const float resblock2_0_weight[OUT_CHANNELS * OUT_CHANNELS * 3];
extern const float resblock2_1_weight[OUT_CHANNELS];
extern const float resblock2_1_bias[OUT_CHANNELS];
extern const float resblock2_1_running_mean[OUT_CHANNELS];
extern const float resblock2_1_running_var[OUT_CHANNELS];
extern const float resblock2_3_weight[OUT_CHANNELS * OUT_CHANNELS * 3];
extern const float resblock2_4_weight[OUT_CHANNELS];
extern const float resblock2_4_bias[OUT_CHANNELS];
extern const float resblock2_4_running_mean[OUT_CHANNELS];
extern const float resblock2_4_running_var[OUT_CHANNELS];

extern const float resblock3_0_weight[OUT_CHANNELS * OUT_CHANNELS * 3];
extern const float resblock3_1_weight[OUT_CHANNELS];
extern const float resblock3_1_bias[OUT_CHANNELS];
extern const float resblock3_1_running_mean[OUT_CHANNELS];
extern const float resblock3_1_running_var[OUT_CHANNELS];
extern const float resblock3_3_weight[OUT_CHANNELS * OUT_CHANNELS * 3];
extern const float resblock3_4_weight[OUT_CHANNELS];
extern const float resblock3_4_bias[OUT_CHANNELS];
extern const float resblock3_4_running_mean[OUT_CHANNELS];
extern const float resblock3_4_running_var[OUT_CHANNELS];

extern const float cov_out_weight[CONV_OUT_CHANNELS * OUT_CHANNELS * 3];

extern const float bn2_weight[CONV_OUT_CHANNELS];
extern const float bn2_bias[CONV_OUT_CHANNELS];
extern const float bn2_running_mean[CONV_OUT_CHANNELS];
extern const float bn2_running_var[CONV_OUT_CHANNELS];

extern const float fc_weight[NUM_CLASSES * PROJECTION_DIM];

#endif
