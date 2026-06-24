"""Training loop for the shared NILM feature extractor."""

import os

import torch
import torch.nn.functional as F
from torch.optim import lr_scheduler

from analytical_nilm.pretrain.losses import (
    SupConLoss,
    cross_entropy_loss,
)
from analytical_nilm.utils import ensure_dir, evaluate_pretrain_model, normalize_per_sample


def pretrain_feature_extractor(model, train_loader, verify_loader, config, device):
    """Pretrain the patch residual feature extractor with CE + supervised
    contrastive loss.

    Args:
        model: Feature extractor with a classifier branch.
        train_loader: DataLoader yielding synthetic training waveform batches.
        verify_loader: DataLoader yielding synthetic verification batches.
        config: Pretraining configuration with optimizer, scheduler, loss, and
            checkpoint fields such as lr, base_lr, epoch, model_save_dir, and
            model_name.
        device: torch.device used for training.

    Returns:
        A tuple (metrics, model_save_path). metrics is a list of per-epoch
        dictionaries. model_save_path is the saved state_dict checkpoint path.
    """

    model.to(device)
    model.train()

    loss_fn = SupConLoss(temperature=getattr(config, "temperature", 0.07))
    loss_fn1 = cross_entropy_loss

    optimizer = torch.optim.SGD(model.parameters(), config.lr)
    scheduler = lr_scheduler.CyclicLR(
        optimizer,
        base_lr=getattr(config, "base_lr", 0.001),
        max_lr=config.lr,
        step_size_up=5,
        step_size_down=5,
    )

    input_norm = getattr(config, "input_norm", True)
    metrics = []
    for epoch in range(config.epoch):
        running_loss = 0.0
        loss_ce_total = 0.0
        loss_supcon_total = 0.0
        total_samples = 0

        for x, targets in train_loader:
            x = x.to(device)
            targets = targets.to(device)
            if input_norm:
                x, _, _ = normalize_per_sample(x)

            optimizer.zero_grad()
            outputs, features = model(x)

            proj_features = F.normalize(features, dim=1)
            loss_supcon = loss_fn(proj_features.unsqueeze(1), targets)
            loss_ce = loss_fn1(outputs, targets)
            loss_value = loss_supcon + loss_ce

            loss_value.backward()
            optimizer.step()

            batch_size = targets.size(0)
            total_samples += batch_size
            running_loss += loss_value.item() * batch_size
            loss_ce_total += loss_ce.item() * batch_size
            loss_supcon_total += loss_supcon.item() * batch_size

        if epoch > 0 and epoch % 10 == 0:
            scheduler.max_lrs = [lr * 0.9 for lr in scheduler.max_lrs]
        scheduler.step()

        verify_acc = evaluate_pretrain_model(model, verify_loader, device, input_norm=input_norm)
        metrics.append(
            {
                "epoch": epoch + 1,
                "total_loss": running_loss / max(total_samples, 1),
                "ce_loss": loss_ce_total / max(total_samples, 1),
                "supcon_loss": loss_supcon_total / max(total_samples, 1),
                "verify_accuracy": verify_acc,
                "lr": optimizer.param_groups[0]["lr"],
            }
        )

    model_save_path = os.path.join(config.model_save_dir, config.model_name)
    ensure_dir(config.model_save_dir)
    torch.save(model.state_dict(), model_save_path)
    return metrics, model_save_path
