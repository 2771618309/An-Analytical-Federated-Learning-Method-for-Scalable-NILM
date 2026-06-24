# Embedded Serial Protocol

The embedded method code uses a small text-header plus SLIP-binary protocol for
matrix exchange. This document describes only the method-related messages.

## SLIP Framing

Binary payloads are wrapped with:

- `0xC0`: frame boundary
- `0xDB 0xDC`: escaped `0xC0`
- `0xDB 0xDD`: escaped `0xDB`
- `0xDB 0xDE`: escaped line feed `0x0A`
- `0xDB 0xDF`: escaped carriage return `0x0D`

Payload values are transmitted as raw little-endian `float` arrays unless the
host platform explicitly changes the implementation.

## Local Analytical Weight Upload

Header:

```text
W_SHAPE <rows> <cols> BIN
```

Payload:

```text
rows * cols * sizeof(float)
```

Terminator:

```text
END
```

For the default configuration, `rows = PROJECTION_DIM` and
`cols = NUM_CLASSES`.

## Block-Wise XTX Upload

Start message:

```text
XTX_START <features> <total_blocks> BIN
```

Each block:

```text
XTX_BLOCK <row_start> <col_start> <block_rows> <block_cols> BIN
<SLIP-framed binary float block>
BLOCK_END
```

End message:

```text
XTX_END
```

Only upper-triangular blocks are sent. The host reconstructs the symmetric
matrix by mirroring off-diagonal blocks.

## Host-Side Responsibilities

The host should:

- Parse text headers line by line.
- Read the following SLIP frame for binary payloads.
- Decode payloads as row-major `float` matrices.
- Reconstruct the full `X^T X` matrix from upper-triangular blocks.
- Combine uploaded local quantities according to the analytical aggregation
  logic in the Python implementation.
