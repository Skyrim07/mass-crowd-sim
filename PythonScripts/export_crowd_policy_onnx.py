import argparse
from pathlib import Path

import torch

from train_crowd_policy import CrowdPolicy, FEATURE_COLUMNS, NormalizedCrowdPolicy


def export(args):
    checkpoint = torch.load(args.input, map_location="cpu")
    feature_columns = checkpoint.get("feature_columns", FEATURE_COLUMNS)
    feature_mean = checkpoint["feature_mean"].float()
    feature_std = checkpoint["feature_std"].float()

    model = CrowdPolicy(len(feature_columns))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    export_model = NormalizedCrowdPolicy(model, feature_mean, feature_std)
    export_model.eval()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    dummy_input = torch.zeros(1, len(feature_columns), dtype=torch.float32)
    torch.onnx.export(
        export_model,
        dummy_input,
        args.output,
        input_names=["observations"],
        output_names=["desired_velocity"],
        dynamic_axes={
            "observations": {0: "batch"},
            "desired_velocity": {0: "batch"},
        },
        opset_version=12,
        dynamo=False,
    )
    print(f"saved {args.output}")


def parse_args():
    parser = argparse.ArgumentParser(description="Export a trained crowd policy checkpoint to normalized ONNX.")
    parser.add_argument("--input", type=Path, default=Path("models/crowd_policy.pt"))
    parser.add_argument("--output", type=Path, default=Path("models/crowd_policy.onnx"))
    return parser.parse_args()


if __name__ == "__main__":
    export(parse_args())
