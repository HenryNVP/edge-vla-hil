#!/usr/bin/env python3
"""Export a LeRobot ACT checkpoint to ONNX for TensorRT compilation.

Run on the x86 host (not the Jetson). The exported graph takes (image, joint_state) and returns an
action chunk. Keep batch/seq dims static where possible — dynamic axes make the TRT build harder.

    python scripts/export_onnx.py --ckpt <dir> --out outputs/act.onnx

TODO: wire to the real ACT policy:
    from lerobot.common.policies.act.modeling_act import ACTPolicy
    policy = ACTPolicy.from_pretrained(args.ckpt).eval()
    wrap policy.forward in a thin nn.Module exposing (image, state) -> action_chunk
    torch.onnx.export(..., opset_version=17, input_names=['image','state'],
                      output_names=['action_chunk'])
Then sanity-check with onnxruntime before building the engine.
"""
import argparse


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True, help='ACT checkpoint dir')
    p.add_argument('--out', default='outputs/act.onnx')
    p.add_argument('--opset', type=int, default=17)
    args = p.parse_args()
    raise SystemExit(
        'STUB: implement ACT->ONNX export. See module docstring. '
        f'(ckpt={args.ckpt} out={args.out} opset={args.opset})')


if __name__ == '__main__':
    main()
