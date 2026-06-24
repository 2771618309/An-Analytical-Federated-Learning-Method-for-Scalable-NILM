#ifndef EDGE_CONFIG_H
#define EDGE_CONFIG_H

#include <stdint.h>

/*
 * Public configuration for the STM32 analytical NILM method demo.
 *
 * These constants match the released embedded method implementation. Adjust
 * them when exporting a different feature extractor or analytical head.
 */

#ifndef INPUT_SEQ_LEN
#define INPUT_SEQ_LEN 150
#endif

#ifndef INPUT_CHANNELS
#define INPUT_CHANNELS 1
#endif

#ifndef OUT_CHANNELS
#define OUT_CHANNELS 16
#endif

#ifndef PATCH_LEN
#define PATCH_LEN 30
#endif

#ifndef STRIDE
#define STRIDE 30
#endif

#ifndef NUM_PATCHES
#define NUM_PATCHES 5
#endif

#ifndef D_MODEL
#define D_MODEL 150
#endif

#ifndef CONV_OUT_CHANNELS
#define CONV_OUT_CHANNELS 10
#endif

#ifndef FEATURE_DIM
#define FEATURE_DIM (CONV_OUT_CHANNELS * D_MODEL)
#endif

#ifndef PROJECTION_DIM
#define PROJECTION_DIM (FEATURE_DIM + INPUT_SEQ_LEN)
#endif

#ifndef NUM_CLASSES
#define NUM_CLASSES 63
#endif

#ifndef MAX_LEN
#define MAX_LEN NUM_PATCHES
#endif

#ifndef LABEL_INDEX
#define LABEL_INDEX INPUT_SEQ_LEN
#endif

#ifndef MAX_SAMPLES
#define MAX_SAMPLES 225
#endif

#ifndef MAX_FEATURES
#define MAX_FEATURES (INPUT_SEQ_LEN + 1)
#endif

#ifndef CLIENT_SCALE
#define CLIENT_SCALE 100.0f
#endif

#ifndef LOCAL_REGULARIZATION
#define LOCAL_REGULARIZATION 0.5f
#endif

#ifndef XTX_BLOCK_SIZE
#define XTX_BLOCK_SIZE 275
#endif

extern volatile int g_edge_abort_flag;

/**
 * @brief Run one edge-side analytical training/update round.
 * @param client_id Logical client identifier used by the host workflow.
 * @param data Input samples. Each row contains INPUT_SEQ_LEN waveform values
 * followed by one label at LABEL_INDEX.
 * @param samples Number of rows in data.
 * @param signal_features Number of columns provided by the host.
 * @param scale Quantization scale for int16 input values.
 * @return 0 on success, -1 on invalid input, allocation failure, or solve failure.
 */
int edge_start_training(
    int client_id,
    int16_t data[][MAX_FEATURES],
    int samples,
    int signal_features,
    float scale
);

#endif
