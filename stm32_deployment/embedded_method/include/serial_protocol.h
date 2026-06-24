#ifndef SERIAL_PROTOCOL_H
#define SERIAL_PROTOCOL_H

#include <stdint.h>

/*
 * Minimal serial protocol used by the embedded analytical method.
 *
 * The implementation uses SLIP framing for binary payloads. Platform-specific
 * UART functions are injected through edge_serial_set_io().
 */

typedef void (*edge_serial_write_fn)(const uint8_t *data, uint32_t length);
typedef int (*edge_serial_read_byte_fn)(uint8_t *byte, uint32_t timeout_ms);
typedef void (*edge_serial_delay_fn)(uint32_t ms);

typedef struct {
    edge_serial_write_fn write;
    edge_serial_read_byte_fn read_byte;
    edge_serial_delay_fn delay_ms;
} EdgeSerialIO;

/**
 * @brief Register platform-specific serial callbacks.
 * @param io Callback table. Pass NULL to clear callbacks.
 */
void edge_serial_set_io(const EdgeSerialIO *io);

/**
 * @brief Receive one SLIP-framed binary payload.
 * @param output Output byte buffer.
 * @param expected_bytes Number of decoded bytes expected.
 * @param timeout_ms Timeout passed to the platform read callback.
 * @return 0 on exact-size success, otherwise -1.
 */
int edge_receive_slip_payload(uint8_t *output, uint32_t expected_bytes, uint32_t timeout_ms);

/**
 * @brief Send an ASCII protocol line.
 * @param text Null-terminated string to send.
 */
void edge_send_text(const char *text);

/**
 * @brief Send a row-major float matrix as a text header plus SLIP payload.
 * @param tag Matrix tag written in the text header.
 * @param matrix Row-major float matrix.
 * @param rows Matrix row count.
 * @param cols Matrix column count.
 */
void edge_send_matrix_binary(const char *tag, const float *matrix, int rows, int cols);

/**
 * @brief Upload the local analytical classifier weight matrix.
 * @param W Weight matrix [w_rows, w_cols].
 * @param w_rows Number of rows.
 * @param w_cols Number of columns.
 */
void edge_send_training_results(const float *W, int w_rows, int w_cols);

/**
 * @brief Upload one block of the upper-triangular X^T X matrix.
 * @param block_data Row-major block data.
 * @param row_start Global row offset.
 * @param col_start Global column offset.
 * @param block_rows Block row count.
 * @param block_cols Block column count.
 */
void edge_upload_xtx_block_binary(
    const float *block_data,
    int row_start,
    int col_start,
    int block_rows,
    int block_cols
);

/**
 * @brief Compute X^T X by blocks and upload upper-triangular blocks.
 * @param X Row-major feature matrix [samples, features].
 * @param samples Number of samples.
 * @param features Number of feature columns.
 * @param block_size Maximum row/column size for one uploaded block.
 * @param block_buffer Temporary buffer with block_size * block_size floats.
 */
void edge_compute_and_upload_xtx_streaming(
    const float *X,
    int samples,
    int features,
    int block_size,
    float *block_buffer
);

#endif
