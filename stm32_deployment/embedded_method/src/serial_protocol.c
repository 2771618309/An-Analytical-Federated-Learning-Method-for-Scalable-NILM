#include "serial_protocol.h"

#include <stdio.h>
#include <string.h>

#include "edge_config.h"

#define SLIP_END 0xC0
#define SLIP_ESC 0xDB
#define SLIP_ESC_END 0xDC
#define SLIP_ESC_ESC 0xDD
#define SLIP_ESC_LF 0xDE
#define SLIP_ESC_CR 0xDF

static EdgeSerialIO g_serial_io = {0};

/**
 * @brief Register platform-specific serial I/O callbacks.
 * @param io Callback table. Pass NULL to clear all callbacks.
 */
void edge_serial_set_io(const EdgeSerialIO *io)
{
    if (io == NULL) {
        memset(&g_serial_io, 0, sizeof(g_serial_io));
        return;
    }
    g_serial_io = *io;
}

/**
 * @brief Delay through the registered platform callback when available.
 * @param ms Delay duration in milliseconds.
 */
static void edge_delay(uint32_t ms)
{
    if (g_serial_io.delay_ms != NULL) {
        g_serial_io.delay_ms(ms);
    }
}

/**
 * @brief Write one byte through the registered serial callback.
 * @param byte Byte to send.
 */
static void edge_write_byte(uint8_t byte)
{
    if (g_serial_io.write != NULL) {
        g_serial_io.write(&byte, 1);
    }
}

/**
 * @brief Send an ASCII protocol header or terminator.
 * @param text Null-terminated ASCII string to send.
 */
void edge_send_text(const char *text)
{
    if (g_serial_io.write == NULL || text == NULL) {
        return;
    }

    g_serial_io.write((const uint8_t *)text, (uint32_t)strlen(text));
    edge_delay(10);
}

/**
 * @brief Send one binary payload with SLIP escaping.
 * @param data Binary payload pointer.
 * @param length Payload length in bytes.
 */
static void edge_send_slip_payload(const uint8_t *data, uint32_t length)
{
    if (g_serial_io.write == NULL || data == NULL || length == 0) {
        return;
    }

    edge_write_byte(SLIP_END);

    for (uint32_t i = 0; i < length; i++) {
        if (g_edge_abort_flag) {
            return;
        }

        switch (data[i]) {
            case SLIP_END:
                edge_write_byte(SLIP_ESC);
                edge_write_byte(SLIP_ESC_END);
                break;
            case SLIP_ESC:
                edge_write_byte(SLIP_ESC);
                edge_write_byte(SLIP_ESC_ESC);
                break;
            case 0x0A:
                edge_write_byte(SLIP_ESC);
                edge_write_byte(SLIP_ESC_LF);
                break;
            case 0x0D:
                edge_write_byte(SLIP_ESC);
                edge_write_byte(SLIP_ESC_CR);
                break;
            default:
                edge_write_byte(data[i]);
                break;
        }

        if ((i + 1) % 2048 == 0) {
            edge_delay(2);
        }
    }

    edge_write_byte(SLIP_END);
    edge_delay(5);
}

/**
 * @brief Receive and decode one SLIP-framed binary payload.
 * @param output Output byte buffer.
 * @param expected_bytes Number of decoded bytes expected.
 * @param timeout_ms Timeout passed to the platform read callback.
 * @return 0 when exactly expected_bytes are decoded, otherwise -1.
 */
int edge_receive_slip_payload(uint8_t *output, uint32_t expected_bytes, uint32_t timeout_ms)
{
    if (g_serial_io.read_byte == NULL || output == NULL || expected_bytes == 0) {
        return -1;
    }

    uint8_t byte = 0;
    int frame_started = 0;
    int escape_next = 0;
    uint32_t received = 0;

    while (!frame_started) {
        if (g_serial_io.read_byte(&byte, timeout_ms) != 0) {
            return -1;
        }
        if (byte == SLIP_END) {
            frame_started = 1;
        }
    }

    while (received < expected_bytes) {
        if (g_serial_io.read_byte(&byte, timeout_ms) != 0) {
            return -1;
        }

        if (byte == SLIP_END) {
            break;
        }

        if (escape_next) {
            escape_next = 0;
            switch (byte) {
                case SLIP_ESC_END:
                    output[received++] = SLIP_END;
                    break;
                case SLIP_ESC_ESC:
                    output[received++] = SLIP_ESC;
                    break;
                case SLIP_ESC_LF:
                    output[received++] = 0x0A;
                    break;
                case SLIP_ESC_CR:
                    output[received++] = 0x0D;
                    break;
                default:
                    output[received++] = byte;
                    break;
            }
        } else if (byte == SLIP_ESC) {
            escape_next = 1;
        } else {
            output[received++] = byte;
        }
    }

    return (received == expected_bytes) ? 0 : -1;
}

/**
 * @brief Send a row-major float matrix with a text header and SLIP payload.
 * @param tag Header tag, for example "W_SHAPE" or "XTX_BLOCK".
 * @param matrix Row-major float matrix.
 * @param rows Matrix row count.
 * @param cols Matrix column count.
 */
void edge_send_matrix_binary(const char *tag, const float *matrix, int rows, int cols)
{
    if (tag == NULL || matrix == NULL || rows <= 0 || cols <= 0) {
        return;
    }

    char header[128];
    snprintf(header, sizeof(header), "%s %d %d BIN\n", tag, rows, cols);
    edge_send_text(header);
    edge_delay(20);

    uint32_t total_bytes = (uint32_t)(rows * cols * (int)sizeof(float));
    edge_send_slip_payload((const uint8_t *)matrix, total_bytes);

    edge_send_text("END\n");
    edge_delay(20);
}

/**
 * @brief Upload the local analytical classifier weight matrix.
 * @param W Local analytical weight matrix [w_rows, w_cols].
 * @param w_rows Number of rows in W.
 * @param w_cols Number of columns in W.
 */
void edge_send_training_results(const float *W, int w_rows, int w_cols)
{
    edge_send_matrix_binary("W_SHAPE", W, w_rows, w_cols);
}

/**
 * @brief Upload one upper-triangular block of X^T X.
 * @param block_data Row-major block buffer.
 * @param row_start Global row index of the block.
 * @param col_start Global column index of the block.
 * @param block_rows Number of rows in the block.
 * @param block_cols Number of columns in the block.
 */
void edge_upload_xtx_block_binary(
    const float *block_data,
    int row_start,
    int col_start,
    int block_rows,
    int block_cols
)
{
    if (block_data == NULL || block_rows <= 0 || block_cols <= 0) {
        return;
    }

    char header[128];
    snprintf(
        header,
        sizeof(header),
        "XTX_BLOCK %d %d %d %d BIN\n",
        row_start,
        col_start,
        block_rows,
        block_cols
    );
    edge_send_text(header);
    edge_delay(10);

    uint32_t block_bytes = (uint32_t)(block_rows * block_cols * (int)sizeof(float));
    edge_send_slip_payload((const uint8_t *)block_data, block_bytes);

    edge_send_text("BLOCK_END\n");
    edge_delay(10);
}

/**
 * @brief Compute X^T X block by block and upload upper-triangular blocks.
 * @param X Row-major feature matrix [samples, features].
 * @param samples Number of samples.
 * @param features Number of feature columns.
 * @param block_size Maximum block row/column size.
 * @param block_buffer Temporary buffer with block_size * block_size floats.
 */
void edge_compute_and_upload_xtx_streaming(
    const float *X,
    int samples,
    int features,
    int block_size,
    float *block_buffer
)
{
    if (X == NULL || block_buffer == NULL || samples <= 0 || features <= 0 || block_size <= 0) {
        return;
    }

    int total_blocks = 0;
    for (int i = 0; i < features; i += block_size) {
        for (int j = i; j < features; j += block_size) {
            total_blocks++;
        }
    }

    char header[128];
    snprintf(header, sizeof(header), "XTX_START %d %d BIN\n", features, total_blocks);
    edge_send_text(header);
    edge_delay(50);

    for (int i = 0; i < features; i += block_size) {
        if (g_edge_abort_flag) {
            return;
        }

        int i_end = (i + block_size < features) ? (i + block_size) : features;
        int i_size = i_end - i;

        for (int j = i; j < features; j += block_size) {
            int j_end = (j + block_size < features) ? (j + block_size) : features;
            int j_size = j_end - j;

            memset(block_buffer, 0, (size_t)(block_size * block_size) * sizeof(float));

            for (int ii = 0; ii < i_size; ii++) {
                int col_i = i + ii;
                for (int jj = 0; jj < j_size; jj++) {
                    int col_j = j + jj;
                    float sum = 0.0f;

                    for (int s = 0; s < samples; s++) {
                        sum += X[s * features + col_i] * X[s * features + col_j];
                    }

                    block_buffer[ii * j_size + jj] = sum;
                }
            }

            edge_upload_xtx_block_binary(block_buffer, i, j, i_size, j_size);
        }
    }

    edge_send_text("XTX_END\n\n");
    edge_delay(200);
}
