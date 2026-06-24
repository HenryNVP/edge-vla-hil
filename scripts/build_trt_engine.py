#!/usr/bin/env python3
"""Build a TensorRT engine from an ONNX ACT policy. Run ON THE JETSON (matches the device's TRT).

    python3 scripts/build_trt_engine.py --onnx outputs/act.onnx --out outputs/act.engine --fp16

The resulting .engine is loaded by evh_controller's TensorRTBackend. FP16 is the default fast path
for the Orin Nano; INT8 needs a calibration cache and is optional.

TODO: implement with the tensorrt python API:
    builder = trt.Builder(logger); network = builder.create_network(EXPLICIT_BATCH)
    parser = trt.OnnxParser(network, logger); parser.parse(open(onnx,'rb').read())
    config = builder.create_builder_config()
    if fp16: config.set_flag(trt.BuilderFlag.FP16)
    engine = builder.build_serialized_network(network, config)
    open(out,'wb').write(engine)
Log the achieved inference latency here so Phase 3's "honest FPS" number is recorded at build time.
"""
import argparse


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--onnx', required=True)
    p.add_argument('--out', default='outputs/act.engine')
    p.add_argument('--fp16', action='store_true')
    p.add_argument('--int8', action='store_true')
    args = p.parse_args()
    raise SystemExit(
        'STUB: implement ONNX->TRT engine build. See module docstring. '
        f'(onnx={args.onnx} out={args.out} fp16={args.fp16} int8={args.int8})')


if __name__ == '__main__':
    main()
