# STM32 Embedded Method Implementation

This folder contains the public STM32-side implementation of the analytical
NILM method. It is not a complete Keil, STM32CubeIDE, or RT-Thread project.
Only the method-related C code is included.

## Scope

Included:

- Patch-residual feature extractor forward implementation.
- Projection branch that concatenates the raw waveform and extracted features.
- Local analytical update on the edge device.
- Regularized Gram solve for the local analytical classifier.
- Block-wise upper-triangular `X^T X` computation and upload.
- SLIP-based binary matrix transmission helpers.
- Placeholder parameter declarations and template definitions.

Not included:

- Complete Keil or STM32Cube project.
- Company product code or device business logic.
- Real deployment model weights.
- Private data, logs, checkpoints, or hardware-specific credentials.
- Board support files, startup files, linker scripts, or vendor HAL packages.

## Folder Structure

```text
embedded_method/
|-- include/
|   |-- edge_config.h        # Public dimensions and method configuration
|   |-- edge_memory.h        # Allocator abstraction
|   |-- model_params.h       # Model parameter declarations
|   |-- patch_resnet.h       # Feature extractor and analytical solver API
|   `-- serial_protocol.h    # SLIP/matrix upload API
|-- src/
|   |-- edge_training.c      # Edge-side method flow
|   |-- patch_resnet.c       # Patch ResNet forward and local solver utilities
|   `-- serial_protocol.c    # Method-related serial protocol helpers
|-- params/
|   `-- model_params_template.c
`-- docs/
    `-- protocol.md
```

## Relation to the Paper

| Paper / deployment component | Code location |
| --- | --- |
| Embedded shared feature extractor | `src/patch_resnet.c`, `include/patch_resnet.h` |
| Patch embedding and positional encoding | `src/patch_resnet.c` |
| 1D convolutional residual feature extraction | `src/patch_resnet.c` |
| Projection/update branch | `patch_resnet_forward()` |
| Edge-side analytical local update | `gram_with_reg_and_inv()` |
| Regularized local solve | `matmul_xxt_with_reg()`, `invert_matrix_gauss_jordan_float()` |
| Block-wise upper-triangular upload | `edge_compute_and_upload_xtx_streaming()` |
| Binary matrix transport | `src/serial_protocol.c`, `docs/protocol.md` |

## Method Flow

The embedded method entry point is `edge_start_training()` in
`src/edge_training.c`.

It follows the same deployment logic as the original STM32 implementation:

1. Initialize the `PatchResNetModel` structure.
2. Attach exported model parameter arrays from `model_params.h`.
3. Convert each received `int16` waveform sample to float and normalize it to
   `[-1, 1]`.
4. Run `patch_resnet_forward()` to extract learned features and concatenate the
   projection branch `[raw waveform | extracted feature]`.
5. Compute the local analytical classifier with
   `gram_with_reg_and_inv()`, corresponding to
   `W = X^T (X X^T + rg I)^(-1) Y`.
6. Send the local `W` matrix and block-wise upper-triangular `X^T X` blocks to
   the host through the SLIP binary protocol.

The host side can then reconstruct the uploaded matrices and perform the cloud
aggregation logic described in the Python implementation.

## Integration Steps

1. Add `include/`, `src/`, and `params/` to your embedded project.
2. Provide CMSIS-DSP and include `arm_math.h`.
3. Replace `params/model_params_template.c` with model parameters exported from
   your own PyTorch checkpoint.
4. Configure dimensions in `include/edge_config.h` if your model differs from
   the default release settings.
5. Bind UART functions through `edge_serial_set_io()`.
6. Optionally bind a board-specific SDRAM allocator by defining:

```c
#define EDGE_MALLOC(size) your_sdram_malloc(size)
#define EDGE_FREE(ptr) your_sdram_free(ptr)
#define EDGE_HEAP_RESET() your_platform_heap_reset()
```

7. Call `edge_start_training()` after a client batch has been received.

## Parameter Template

The file `params/model_params_template.c` intentionally contains zero-filled
arrays only. It documents the expected parameter names and shapes but does not
publish the trained deployment weights.

Before real deployment, replace the template arrays with exported weights for:

- Patch embedding token convolution.
- Positional encoding.
- Convolution and batch-normalization layers.
- Residual block layers.
- Output convolution and batch normalization.
- Projection-branch analytical classifier parameters.

## Public API

Main method entry point:

```c
int edge_start_training(
    int client_id,
    int16_t data[][MAX_FEATURES],
    int samples,
    int signal_features,
    float scale
);
```

The input matrix is expected to contain `INPUT_SEQ_LEN` waveform values followed
by one label column at `LABEL_INDEX`.

## Notes

- The code is provided as a method implementation reference.
- It requires platform adaptation before flashing to a real STM32 device.
- The default serial protocol sends `W_SHAPE` and upper-triangular `XTX_BLOCK`
  payloads using SLIP-framed binary floats.
