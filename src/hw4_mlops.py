from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm


DEFAULT_BUCKET = "hw4-models"
DEFAULT_ENDPOINT_URL = "http://localhost:9000"


class ImageCsvDataset(Dataset):
    def __init__(self, csv_file: str | Path, root_dir: str | Path, transform: Any | None = None) -> None:
        self.data = pd.read_csv(csv_file)
        self.root_dir = Path(root_dir)
        self.transform = transform

        required = {"file_name", "label"}
        missing = required.difference(self.data.columns)
        if missing:
            raise ValueError(f"{csv_file} is missing columns: {sorted(missing)}")

    def __len__(self) -> int:
        return len(self.data)

    def _resolve_path(self, file_name: str) -> Path:
        raw_path = Path(str(file_name))
        candidates = []
        if raw_path.is_absolute():
            candidates.append(raw_path)
        else:
            candidates.extend(
                [
                    self.root_dir / raw_path,
                    self.root_dir / raw_path.name,
                    self.root_dir / "train_data" / raw_path.name,
                    self.root_dir / "test_data" / raw_path.name,
                ]
            )

        for candidate in candidates:
            if candidate.exists():
                return candidate

        raise FileNotFoundError(
            f"Image {file_name!r} was not found under {self.root_dir}. "
            f"Tried: {[str(p) for p in candidates]}"
        )

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.data.iloc[idx]
        img_path = self._resolve_path(row["file_name"])
        image = Image.open(img_path).convert("RGB")
        label = int(row["label"])

        if self.transform:
            image = self.transform(image)

        return image, label


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def build_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_transform, test_transform


def build_model(pretrained: bool = True, freeze_backbone: bool = False) -> nn.Module:
    try:
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
    except AttributeError:
        model = models.resnet18(pretrained=pretrained)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    return model


def classification_metrics(labels: list[int], preds: list[int]) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, average="binary", zero_division=0)),
        "precision": float(precision_score(labels, preds, average="binary", zero_division=0)),
        "recall": float(recall_score(labels, preds, average="binary", zero_division=0)),
    }


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: optim.Optimizer | None = None,
    max_batches: int | None = None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    running_loss = 0.0
    total = 0
    all_preds: list[int] = []
    all_labels: list[int] = []
    phase = "train" if is_train else "eval"

    with torch.set_grad_enabled(is_train):
        for step, (images, labels) in enumerate(tqdm(dataloader, desc=phase)):
            if max_batches is not None and step >= max_batches:
                break

            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_train:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            total += batch_size

            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.detach().cpu().tolist())
            all_labels.extend(labels.detach().cpu().tolist())

    metrics = classification_metrics(all_labels, all_preds)
    metrics["loss"] = float(running_loss / max(total, 1))
    return metrics


def make_dataloader(
    csv_file: str | Path,
    root_dir: str | Path,
    transform: Any,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    dataset = ImageCsvDataset(csv_file=csv_file, root_dir=root_dir, transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )


def save_json(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def load_model_state(model: nn.Module, checkpoint_path: str | Path, device: torch.device) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)


def get_s3_client(endpoint_url: str | None = None) -> Any:
    import boto3

    endpoint = endpoint_url or os.getenv("S3_ENDPOINT_URL", DEFAULT_ENDPOINT_URL)
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "minioadmin"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin"),
        region_name=os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
    )


def ensure_bucket(client: Any, bucket: str) -> None:
    from botocore.exceptions import ClientError

    try:
        client.head_bucket(Bucket=bucket)
    except ClientError:
        client.create_bucket(Bucket=bucket)


def upload_to_s3(path: str | Path, key: str, bucket: str, endpoint_url: str | None = None) -> str:
    client = get_s3_client(endpoint_url)
    ensure_bucket(client, bucket)
    client.upload_file(str(path), bucket, key)
    uri = f"s3://{bucket}/{key}"
    print(f"Uploaded {path} to {uri}")
    return uri


def download_from_s3(key: str, output: str | Path, bucket: str, endpoint_url: str | None = None) -> Path:
    client = get_s3_client(endpoint_url)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    client.download_file(bucket, key, str(output))
    print(f"Downloaded s3://{bucket}/{key} to {output}")
    return output


def train_command(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_transform, test_transform = build_transforms(args.image_size)
    train_loader = make_dataloader(
        args.train_csv,
        args.train_root,
        train_transform,
        args.batch_size,
        True,
        args.num_workers,
        device,
    )
    test_loader = make_dataloader(
        args.test_csv,
        args.test_root,
        test_transform,
        args.batch_size,
        False,
        args.num_workers,
        device,
    )

    model = build_model(pretrained=not args.no_pretrained, freeze_backbone=args.freeze_backbone)
    model = model.to(device)
    if args.base_checkpoint:
        load_model_state(model, args.base_checkpoint, device)

    criterion = nn.CrossEntropyLoss()
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    writer = None
    if not args.no_tensorboard:
        from torch.utils.tensorboard import SummaryWriter

        tb_dir = Path(args.tb_logdir) / args.experiment_name
        writer = SummaryWriter(log_dir=str(tb_dir))
        writer.add_text("hparams", json.dumps(vars(args), indent=2, ensure_ascii=False), 0)

    history: list[dict[str, Any]] = []
    started_at = time.time()
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            max_batches=args.max_train_batches,
        )
        scheduler.step()

        row = {"epoch": epoch, **train_metrics, "lr": float(scheduler.get_last_lr()[0])}
        history.append(row)
        print(f"epoch={epoch} train={row}")

        if writer:
            for key, value in train_metrics.items():
                writer.add_scalar(f"train/{key}", value, epoch)
            writer.add_scalar("train/lr", row["lr"], epoch)

    test_metrics = run_epoch(
        model,
        test_loader,
        criterion,
        device,
        optimizer=None,
        max_batches=args.max_test_batches,
    )
    print(f"test={test_metrics}")

    if writer:
        for key, value in test_metrics.items():
            writer.add_scalar(f"test/{key}", value, args.epochs)
        writer.add_text("test_metrics", json.dumps(test_metrics, indent=2), args.epochs)
        writer.flush()
        writer.close()

    model_path = output_dir / "model.pt"
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "experiment_name": args.experiment_name,
        "model": "resnet18",
        "num_classes": 2,
        "args": vars(args),
        "train_history": history,
        "test_metrics": test_metrics,
    }
    torch.save(checkpoint, model_path)

    s3_uri = None
    if args.s3_upload_key:
        s3_uri = upload_to_s3(model_path, args.s3_upload_key, args.s3_bucket, args.s3_endpoint_url)

    result = {
        "experiment_name": args.experiment_name,
        "device": str(device),
        "model_path": str(model_path),
        "s3_uri": s3_uri,
        "elapsed_sec": time.time() - started_at,
        "train_history": history,
        "test_metrics": test_metrics,
    }
    save_json(output_dir / "metrics.json", result)
    return result


def compare_command(args: argparse.Namespace) -> None:
    base = json.loads(Path(args.base_metrics).read_text())
    fine_tuned = json.loads(Path(args.finetuned_metrics).read_text())

    rows = []
    for name, payload in [("base_train1", base), ("finetuned_train2", fine_tuned)]:
        test = payload["test_metrics"]
        last_train = payload["train_history"][-1]
        rows.append(
            {
                "version": name,
                "train_loss": last_train["loss"],
                "train_accuracy": last_train["accuracy"],
                "train_f1": last_train["f1"],
                "test_loss": test["loss"],
                "test_accuracy": test["accuracy"],
                "test_f1": test["f1"],
                "test_precision": test["precision"],
                "test_recall": test["recall"],
            }
        )

    df = pd.DataFrame(rows)
    csv_output = Path(args.csv_output)
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_output, index=False)

    delta = df.iloc[1].drop(labels=["version"]) - df.iloc[0].drop(labels=["version"])
    lines = [
        "# Model comparison",
        "",
        df.to_markdown(index=False),
        "",
        "## Delta finetuned - base",
        "",
        delta.to_frame("delta").to_markdown(),
        "",
    ]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines))
    print(f"Saved comparison to {output} and {csv_output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HW4 MLOps training, S3 and DVC helpers")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="Train or fine-tune ResNet18")
    train.add_argument("--experiment-name", required=True)
    train.add_argument("--train-csv", required=True)
    train.add_argument("--train-root", required=True)
    train.add_argument("--test-csv", required=True)
    train.add_argument("--test-root", required=True)
    train.add_argument("--output-dir", required=True)
    train.add_argument("--epochs", type=int, default=5)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--lr", type=float, default=1e-3)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--step-size", type=int, default=3)
    train.add_argument("--gamma", type=float, default=0.1)
    train.add_argument("--image-size", type=int, default=224)
    train.add_argument("--num-workers", type=int, default=4)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--base-checkpoint")
    train.add_argument("--freeze-backbone", action="store_true")
    train.add_argument("--no-pretrained", action="store_true")
    train.add_argument("--no-tensorboard", action="store_true")
    train.add_argument("--tb-logdir", default="tb_logs")
    train.add_argument("--s3-upload-key")
    train.add_argument("--s3-bucket", default=os.getenv("S3_BUCKET", DEFAULT_BUCKET))
    train.add_argument("--s3-endpoint-url", default=os.getenv("S3_ENDPOINT_URL", DEFAULT_ENDPOINT_URL))
    train.add_argument("--max-train-batches", type=int)
    train.add_argument("--max-test-batches", type=int)
    train.add_argument("--cpu", action="store_true")

    download = subparsers.add_parser("download", help="Download model artifact from S3")
    download.add_argument("--s3-key", required=True)
    download.add_argument("--output", required=True)
    download.add_argument("--s3-bucket", default=os.getenv("S3_BUCKET", DEFAULT_BUCKET))
    download.add_argument("--s3-endpoint-url", default=os.getenv("S3_ENDPOINT_URL", DEFAULT_ENDPOINT_URL))

    upload = subparsers.add_parser("upload", help="Upload model artifact to S3")
    upload.add_argument("--path", required=True)
    upload.add_argument("--s3-key", required=True)
    upload.add_argument("--s3-bucket", default=os.getenv("S3_BUCKET", DEFAULT_BUCKET))
    upload.add_argument("--s3-endpoint-url", default=os.getenv("S3_ENDPOINT_URL", DEFAULT_ENDPOINT_URL))

    compare = subparsers.add_parser("compare", help="Compare base and fine-tuned metrics")
    compare.add_argument("--base-metrics", required=True)
    compare.add_argument("--finetuned-metrics", required=True)
    compare.add_argument("--output", required=True)
    compare.add_argument("--csv-output", required=True)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "train":
        train_command(args)
    elif args.command == "download":
        download_from_s3(args.s3_key, args.output, args.s3_bucket, args.s3_endpoint_url)
    elif args.command == "upload":
        upload_to_s3(args.path, args.s3_key, args.s3_bucket, args.s3_endpoint_url)
    elif args.command == "compare":
        compare_command(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
