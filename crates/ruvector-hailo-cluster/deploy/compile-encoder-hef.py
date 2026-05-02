#!/usr/bin/env python3
"""Compile the encoder-only ONNX (iter 139) to Hailo-8 .hef.

Companion to compile-hef.py. Uses the encoder-only export from
export-minilm-encoder-onnx.py — no Gather/Where/Expand ops, just clean
MatMul/Softmax/Add/Mul/Reshape encoder primitives that Hailo can fuse.

If this compile succeeds, the HEF surgery in ADR-167 is unblocked.
The host-side embedding lookup + mask construction will be wired in
HailoEmbedder in a follow-up iter.

Usage: python3 compile-encoder-hef.py <encoder_onnx> <out_hef>
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

from hailo_sdk_client import ClientRunner
import numpy as np

HW_ARCH = "hailo8"
NET_NAME = "minilm_encoder"
SEQ_LEN = 128
HIDDEN = 384


def main(onnx_path: str, out_hef: str) -> None:
    onnx_path = Path(onnx_path).resolve()
    out_hef = Path(out_hef).resolve()
    work = out_hef.parent

    print(f"==> [parse] {onnx_path}", flush=True)
    runner = ClientRunner(hw_arch=HW_ARCH)
    runner.translate_onnx_model(
        str(onnx_path),
        net_name=NET_NAME,
        start_node_names=["hidden_states", "extended_attention_mask"],
        end_node_names=["last_hidden_state"],
        net_input_shapes={
            "hidden_states": [1, SEQ_LEN, HIDDEN],
            "extended_attention_mask": [1, 1, 1, SEQ_LEN],
        },
    )

    parsed_har = work / f"{NET_NAME}_parsed.har"
    runner.save_har(str(parsed_har))
    print(f"    parsed HAR → {parsed_har}", flush=True)

    print("==> [optimize] random calibration set (FP→INT8)", flush=True)
    rng = np.random.default_rng(seed=42)
    calib = {
        "hidden_states": rng.standard_normal((64, SEQ_LEN, HIDDEN), dtype=np.float32),
        "extended_attention_mask": np.zeros((64, 1, 1, SEQ_LEN), dtype=np.float32),
    }
    runner.optimize(calib)
    opt_har = work / f"{NET_NAME}_optimized.har"
    runner.save_har(str(opt_har))
    print(f"    optimized HAR → {opt_har}", flush=True)

    print("==> [compile] hailo8 placement + scheduling (slow — minutes)", flush=True)
    hef = runner.compile()
    out_hef.write_bytes(hef)
    size = out_hef.stat().st_size
    print(f"    {size} bytes → {out_hef}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"usage: {sys.argv[0]} <encoder_onnx> <out_hef>", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
