"""Toy pretraining entry point for the analytical NILM method.

This script replaces private NILM waveforms with synthetic tensors while
keeping the same public method flow: build the patch residual feature
extractor, train it with CE + supervised contrastive loss, and save a
checkpoint for the analytical federated learning demo.
"""

import argparse
from pathlib import Path
import sys


# Allow the example to be launched directly from the repository root.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analytical_nilm.pretrain.models import PatchResidualFeatureExtractor
from analytical_nilm.pretrain.trainer import pretrain_feature_extractor
from analytical_nilm.utils import (
    get_device,
    load_config,
    make_pretrain_dataloaders,
    save_json,
    save_text,
    set_seed,
)


def build_model(config):
    """Build the shared feature extractor used by pretraining and AFL.

    Args:
        config: Pretraining configuration with model architecture fields such
            as class_num, length, patch_len, patch_stride, patch_embed_dim,
            out_channel, conv_out_channels, hidden_dim, and dropout.

    Returns:
        A PatchResidualFeatureExtractor instance.
    """

    return PatchResidualFeatureExtractor(
        num_classes=config.class_num,
        input_length=config.length,
        patch_len=config.patch_len,
        patch_stride=config.patch_stride,
        patch_embed_dim=config.patch_embed_dim,
        out_channels=config.out_channel,
        conv_out_channels=config.conv_out_channels,
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
    )


def main():
    """Run synthetic pretraining and write the resulting checkpoint.

    Args:
        None. Command-line arguments are parsed inside the function.

    Returns:
        None.
    """

    parser = argparse.ArgumentParser(description="Toy pretraining demo with synthetic data.")
    parser.add_argument(
        "--config",
        default="analytical_nilm/configs/pretrain_config.yaml",
        help="Path to pretraining config YAML.",
    )
    args = parser.parse_args()

    # Load public demo settings and fix randomness for reproducible toy output.
    config = load_config(args.config)
    set_seed(config.random)
    device = get_device(config.device)

    # Synthetic loaders have the same tensor format as the private NILM loader.
    train_loader, verify_loader = make_pretrain_dataloaders(config)
    model = build_model(config)

    # The trainer follows the method pretraining path: CE + supervised
    # contrastive loss, SGD, CyclicLR, and state_dict checkpoint saving.
    metrics, model_save_path = pretrain_feature_extractor(
        model,
        train_loader,
        verify_loader,
        config,
        device,
    )

    if getattr(config, "save_metrics", True):
        # Save lightweight demo artifacts only; no private data or experiment logs.
        metrics_path = Path(config.output_dir) / "toy_pretrain_metrics.json"
        save_json(
            {
                "config": args.config,
                "device": str(device),
                "model_save_path": model_save_path,
                "metrics": metrics,
            },
            str(metrics_path),
        )
        final_acc = metrics[-1]["verify_accuracy"] if metrics else None
        summary = (
            "Toy pretraining completed.\n"
            f"Checkpoint: {model_save_path}\n"
            f"Epochs: {config.epoch}\n"
            f"Final verify accuracy: {final_acc}\n"
        )
        save_text(summary, str(Path(config.output_dir) / "toy_pretrain_summary.txt"))

    print("Toy pretraining completed.")
    print(f"Checkpoint saved to: {model_save_path}")


if __name__ == "__main__":
    main()
