"""Toy analytical federated learning entry point.

This script demonstrates the public AFL pipeline with synthetic data:
prepare client splits, load the pretrained feature extractor, compute local
closed-form updates, aggregate the analytical head on the server, optionally
remove the regularization term, and evaluate the resulting classifier.
"""

import argparse
import os
from pathlib import Path
import sys

import torch


# Allow the example to be launched directly from the repository root.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analytical_nilm.federated.analytical_federated import (
    LinearAnalyticalHead,
    aggregation1,
    clean_regularization_projection,
    diff,
    evaluate_projection,
    global_update_projection,
    local_update_projection,
    set_head_weight,
)
from analytical_nilm.pretrain.models import PatchResidualFeatureExtractor
from analytical_nilm.utils import (
    build_federated_dataloaders,
    get_device,
    load_config,
    prepare_synthetic_data,
    save_json,
    save_text,
    set_seed,
)


def build_model(config):
    """Build the shared pretrained feature extractor.

    Args:
        config: Federated configuration with model architecture fields such as
            init_num_class, length, patch_len, patch_stride, patch_embed_dim,
            out_channel, conv_out_channels, hidden_dim, and dropout.

    Returns:
        A PatchResidualFeatureExtractor instance.
    """

    return PatchResidualFeatureExtractor(
        num_classes=config.init_num_class,
        input_length=config.length,
        patch_len=config.patch_len,
        patch_stride=config.patch_stride,
        patch_embed_dim=config.patch_embed_dim,
        out_channels=config.out_channel,
        conv_out_channels=config.conv_out_channels,
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
    )


def load_pretrained_if_needed(model, config, device):
    """Load the toy pretraining checkpoint when requested by the config.

    Args:
        model: Feature extractor instance to update.
        config: Federated configuration with pretrained, model_save_dir, and
            model_name fields.
        device: torch.device used as the checkpoint loading target.

    Returns:
        The loaded checkpoint path when pretraining is enabled; otherwise None.
    """

    if not getattr(config, "pretrained", False):
        return None
    model_pretrain_path = os.path.join(config.model_save_dir, config.model_name)
    state_dict = torch.load(model_pretrain_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    return model_pretrain_path


def main():
    """Run the synthetic analytical federated learning workflow.

    Args:
        None. Command-line arguments are parsed inside the function.

    Returns:
        None.
    """

    parser = argparse.ArgumentParser(description="Toy analytical federated learning demo.")
    parser.add_argument(
        "--config",
        default="analytical_nilm/configs/federated_config.yaml",
        help="Path to federated config YAML.",
    )
    args = parser.parse_args()

    # Load public demo settings and fix randomness for reproducible client splits.
    config = load_config(args.config)
    set_seed(config.seed)
    device = get_device(getattr(config, "gpu", getattr(config, "device", "auto")))

    # Synthetic data replaces private NILM data while preserving the prepare-data
    # style interface used by the method implementation.
    train_total, train_data_idx, test_set = prepare_synthetic_data(config)
    client_loaders, all_train_loader, test_loader, train_dataset = build_federated_dataloaders(
        config,
        train_total,
        train_data_idx,
        test_set,
    )

    # The analytical stage uses frozen pretrained features.
    feature_extractor = build_model(config).to(device)
    loaded_checkpoint = load_pretrained_if_needed(feature_extractor, config, device)
    feature_extractor.eval()

    # The projection version concatenates normalized learned features with the
    # input waveform branch before solving the analytical classifier.
    feature_dim = feature_extractor.feature_dim
    config.feat_size = feature_dim
    analytical_dim = feature_dim + config.length
    global_head = LinearAnalyticalHead(analytical_dim, config.num_classes).to(device)

    # Each client computes its local closed-form update independently.
    local_weights, local_R, local_C = [], [], []
    local_shapes = []
    for client_id, loader in enumerate(client_loaders):
        W, R, C, reps_all, label_onehot_all = local_update_projection(
            loader,
            feature_extractor,
            global_head,
            config,
            device,
        )
        local_weights.append(W.cpu())
        local_R.append(R)
        local_C.append(C)
        local_shapes.append(
            {
                "client_id": client_id,
                "num_samples": reps_all.size(0),
                "weight": list(W.shape),
                "R": list(R.shape),
                "C": list(C.shape),
                "features": list(reps_all.shape),
                "labels_onehot": list(label_onehot_all.shape),
            }
        )

    # The server aggregates local analytical solutions into a global head.
    global_weight, global_R, global_C = aggregation1(local_weights, local_R, local_C, config, device)

    # Evaluate the aggregated analytical classifier before regularization cleanup.
    set_head_weight(global_head, global_weight)
    aggregated_train_acc = evaluate_projection(
        all_train_loader,
        feature_extractor,
        global_head,
        config,
        device,
    )
    aggregated_test_acc = evaluate_projection(
        test_loader,
        feature_extractor,
        global_head,
        config,
        device,
    )

    clean_train_acc = None
    clean_test_acc = None
    clean_diff = None
    if getattr(config, "clean_reg", False):
        # Remove the regularization contribution introduced in local updates.
        clean_weight = clean_regularization_projection(global_weight, global_C, config, device)
        set_head_weight(global_head, clean_weight)
        clean_train_acc = evaluate_projection(
            all_train_loader,
            feature_extractor,
            global_head,
            config,
            device,
        )
        clean_test_acc = evaluate_projection(
            test_loader,
            feature_extractor,
            global_head,
            config,
            device,
        )

        # Compare with a centralized closed-form solution on the same synthetic
        # training set as a sanity check for the toy pipeline.
        theory_head = LinearAnalyticalHead(analytical_dim, config.num_classes).to(device)
        theory_weight, _, _ = global_update_projection(
            all_train_loader,
            feature_extractor,
            theory_head,
            config,
            device,
        )
        clean_diff = float(diff(theory_weight.to(device), clean_weight).detach().cpu())

    metrics = {
        "config": args.config,
        "device": str(device),
        "loaded_checkpoint": loaded_checkpoint,
        "num_clients": config.num_clients,
        "num_train_samples": len(train_total),
        "num_test_samples": len(test_set),
        "analytical_dim": analytical_dim,
        "aggregated_train_accuracy": aggregated_train_acc,
        "aggregated_test_accuracy": aggregated_test_acc,
        "clean_train_accuracy": clean_train_acc,
        "clean_test_accuracy": clean_test_acc,
        "clean_weight_difference_to_theory": clean_diff,
        "global_weight_shape": list(global_weight.shape),
        "global_R_shape": list(global_R.shape),
        "global_C_shape": list(global_C.shape),
    }

    if getattr(config, "save_metrics", True):
        # Save compact demo artifacts only; no private data or full experiment logs.
        output_dir = Path(config.output_dir)
        save_json(metrics, str(output_dir / "toy_federated_metrics.json"))
        save_json(local_shapes, str(output_dir / "toy_local_update_shapes.json"))
        summary = (
            "Toy analytical federated learning completed.\n"
            f"Checkpoint: {loaded_checkpoint}\n"
            f"Aggregated train accuracy: {aggregated_train_acc}\n"
            f"Aggregated test accuracy: {aggregated_test_acc}\n"
            f"Clean train accuracy: {clean_train_acc}\n"
            f"Clean test accuracy: {clean_test_acc}\n"
            f"Clean weight difference to theory: {clean_diff}\n"
        )
        save_text(summary, str(output_dir / "toy_federated_summary.txt"))

    print("Toy analytical federated learning completed.")
    print(f"Aggregated test accuracy: {aggregated_test_acc:.4f}")
    if clean_test_acc is not None:
        print(f"Clean test accuracy: {clean_test_acc:.4f}")


if __name__ == "__main__":
    main()
