import argparse
import csv
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


FEATURE_COLUMNS = [
    "target_offset_x",
    "target_offset_z",
    "target_distance",
    "velocity_x",
    "velocity_z",
    "speed",
    "nearest_neighbor_offset_x",
    "nearest_neighbor_offset_z",
    "nearest_neighbor_distance",
    "neighbor_count",
    "boundary_distance_x",
    "boundary_distance_z",
]

TARGET_COLUMNS = [
    "desired_velocity_x",
    "desired_velocity_z",
]


class CrowdPolicy(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, len(TARGET_COLUMNS)),
        )

    def forward(self, x):
        return self.network(x)


class NormalizedCrowdPolicy(nn.Module):
    def __init__(self, policy, feature_mean, feature_std):
        super().__init__()
        self.policy = policy
        self.register_buffer("feature_mean", feature_mean)
        self.register_buffer("feature_std", feature_std)

    def forward(self, x):
        return self.policy((x - self.feature_mean) / self.feature_std)


def load_csv(path):
    features = []
    targets = []

    with path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)

        for row in reader:
            features.append([float(row[column]) for column in FEATURE_COLUMNS])
            targets.append([float(row[column]) for column in TARGET_COLUMNS])

    if not features:
        raise ValueError(f"No training rows found in {path}")

    return torch.tensor(features, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32)


def split_dataset(features, targets, validation_fraction):
    row_count = features.shape[0]
    permutation = torch.randperm(row_count)
    validation_count = max(1, int(row_count * validation_fraction))
    validation_indices = permutation[:validation_count]
    train_indices = permutation[validation_count:]

    return (
        features[train_indices],
        targets[train_indices],
        features[validation_indices],
        targets[validation_indices],
    )


def normalize(train_features, validation_features):
    mean = train_features.mean(dim=0)
    std = train_features.std(dim=0).clamp_min(1e-6)
    return (train_features - mean) / std, (validation_features - mean) / std, mean, std


def train(args):
    torch.manual_seed(args.seed)

    features, targets = load_csv(args.input)
    train_features, train_targets, validation_features, validation_targets = split_dataset(
        features,
        targets,
        args.validation_fraction,
    )
    train_features, validation_features, feature_mean, feature_std = normalize(train_features, validation_features)

    train_loader = DataLoader(
        TensorDataset(train_features, train_targets),
        batch_size=args.batch_size,
        shuffle=True,
    )

    model = CrowdPolicy(train_features.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0

        for batch_features, batch_targets in train_loader:
            predicted = model(batch_features)
            loss = loss_fn(predicted, batch_targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * batch_features.shape[0]

        model.eval()
        with torch.no_grad():
            validation_loss = loss_fn(model(validation_features), validation_targets).item()

        train_loss = train_loss_sum / max(1, train_features.shape[0])

        if epoch == 1 or epoch % args.log_interval == 0 or epoch == args.epochs:
            print(f"epoch={epoch:04d} train_mse={train_loss:.6f} validation_mse={validation_loss:.6f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_columns": FEATURE_COLUMNS,
            "target_columns": TARGET_COLUMNS,
            "feature_mean": feature_mean,
            "feature_std": feature_std,
        },
        args.output,
    )
    print(f"saved {args.output}")

    if args.onnx_output:
        args.onnx_output.parent.mkdir(parents=True, exist_ok=True)
        export_model = NormalizedCrowdPolicy(model, feature_mean, feature_std)
        export_model.eval()
        dummy_input = torch.zeros(1, len(FEATURE_COLUMNS), dtype=torch.float32)
        torch.onnx.export(
            export_model,
            dummy_input,
            args.onnx_output,
            input_names=["observations"],
            output_names=["desired_velocity"],
            dynamic_axes={
                "observations": {0: "batch"},
                "desired_velocity": {0: "batch"},
            },
            opset_version=12,
        )
        print(f"saved {args.onnx_output}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a small imitation policy from crowd teacher CSV data.")
    parser.add_argument("input", type=Path, help="Path to crowd_training_data.csv")
    parser.add_argument("--output", type=Path, default=Path("models/crowd_policy.pt"))
    parser.add_argument("--onnx-output", type=Path, default=Path("models/crowd_policy.onnx"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--log-interval", type=int, default=5)
    parser.add_argument("--seed", type=int, default=12345)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
