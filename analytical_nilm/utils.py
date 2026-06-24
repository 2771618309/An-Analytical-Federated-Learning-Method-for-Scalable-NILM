"""Utility functions for configs, synthetic data, normalization, and metrics."""

import json
import math
import os
import random
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset, TensorDataset


def _to_namespace(value):
    """Recursively convert dictionaries into attribute-access namespaces.

    Args:
        value: A dictionary, list, or scalar loaded from a YAML config file.

    Returns:
        The same data structure with dictionaries converted to SimpleNamespace
        objects.
    """

    if isinstance(value, dict):
        return SimpleNamespace(**{key: _to_namespace(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_to_namespace(item) for item in value]
    return value


def load_config(config_path):
    """Load a YAML configuration file as an attribute-access object.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        A SimpleNamespace tree containing the configuration fields.

    Raises:
        RuntimeError: If the file cannot be decoded with the supported encodings.
        ValueError: If the YAML root is not a mapping.
    """

    encodings = ["utf-8-sig", "utf-8", "gbk"]
    last_exc = None
    for encoding in encodings:
        try:
            with open(config_path, "r", encoding=encoding) as handle:
                config_dict = yaml.safe_load(handle)
            break
        except UnicodeDecodeError as exc:
            last_exc = exc
            continue
    else:
        raise RuntimeError(f"Cannot read config file {config_path}: {last_exc}")

    if not isinstance(config_dict, dict):
        raise ValueError(f"Config file {config_path} must contain a mapping.")
    return _to_namespace(config_dict)


def ensure_dir(path):
    """Create a directory if it does not already exist.

    Args:
        path: Directory path to create.

    Returns:
        None.
    """

    os.makedirs(path, exist_ok=True)


def save_json(payload, path):
    """Write a JSON artifact to disk.

    Args:
        payload: JSON-serializable object to save.
        path: Destination file path.

    Returns:
        None.
    """

    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def save_text(text, path):
    """Write a text artifact to disk.

    Args:
        text: Text content to save.
        path: Destination file path.

    Returns:
        None.
    """

    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def set_seed(seed):
    """Set random seeds for Python, NumPy, and PyTorch.

    Args:
        seed: Integer seed. If None, the function leaves random states unchanged.

    Returns:
        None.
    """

    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_value="auto"):
    """Resolve a config device value into a torch.device.

    Args:
        device_value: Device string such as "cpu", "cuda:0", or "auto".

    Returns:
        A torch.device. "auto" selects CUDA when available, otherwise CPU.
    """

    if device_value in (None, "auto"):
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device_value)


def normalize_per_sample(data):
    """Normalize each waveform independently to the range [-1, 1].

    Args:
        data: Tensor with shape [batch_size, length].

    Returns:
        A tuple of (normalized_data, min_val, max_val), where min_val and
        max_val can be reused for inverse normalization.
    """

    min_val = data.min(dim=1, keepdim=True).values
    max_val = data.max(dim=1, keepdim=True).values
    normalized = (data - min_val) / (max_val - min_val + 1e-8)
    normalized = normalized * 2 - 1
    return normalized, min_val, max_val


def denormalize_per_sample(normalized_data, min_val, max_val):
    """Invert per-sample [-1, 1] normalization.

    Args:
        normalized_data: Normalized tensor with shape [batch_size, length].
        min_val: Per-sample minimum values returned by normalize_per_sample.
        max_val: Per-sample maximum values returned by normalize_per_sample.

    Returns:
        Tensor restored to the original per-sample value range.
    """

    scaled_data = (normalized_data + 1) / 2
    return scaled_data * (max_val - min_val) + min_val


def make_synthetic_waveform_dataset(
    num_samples=600,
    num_classes=6,
    input_length=150,
    noise_std=0.03,
    seed=42,
):
    """Build a small synthetic NILM-like dataset for code-path validation.

    This does not reproduce paper experiments. It only replaces private data
    with generated waveforms so the method implementation can be executed.

    Args:
        num_samples: Total number of generated waveform samples.
        num_classes: Number of synthetic appliance classes.
        input_length: Length of each one-dimensional waveform.
        noise_std: Standard deviation of additive Gaussian noise.
        seed: Random seed used by the PyTorch generator.

    Returns:
        A tuple (x, y), where x has shape [num_samples, input_length] and y has
        shape [num_samples].
    """

    generator = torch.Generator().manual_seed(seed)
    samples_per_class = int(math.ceil(num_samples / num_classes))
    t = torch.linspace(0, 1, input_length)
    xs = []
    ys = []

    for label in range(num_classes):
        freq = 1.0 + label
        phase = 0.2 * label
        base_sine = torch.sin(2 * math.pi * freq * t + phase)
        harmonic = 0.35 * torch.sin(2 * math.pi * (freq + 1) * t)
        square_like = torch.sign(torch.sin(2 * math.pi * freq * t + phase))
        pulse = torch.zeros_like(t)
        center = 0.15 + 0.7 * ((label % max(num_classes, 1)) / max(num_classes - 1, 1))
        pulse += torch.exp(-((t - center) ** 2) / (2 * 0.01))

        if label % 4 == 0:
            template = base_sine + harmonic
        elif label % 4 == 1:
            template = 0.7 * square_like + 0.3 * base_sine
        elif label % 4 == 2:
            template = pulse + 0.2 * harmonic
        else:
            template = base_sine * (0.6 + 0.4 * torch.cos(2 * math.pi * t)) + pulse

        for _ in range(samples_per_class):
            amplitude = 0.8 + 0.4 * torch.rand((), generator=generator)
            offset = 0.05 * torch.randn((), generator=generator)
            noise = noise_std * torch.randn(input_length, generator=generator)
            xs.append(amplitude * template + offset + noise)
            ys.append(label)

    x = torch.stack(xs, dim=0)[:num_samples].float()
    y = torch.tensor(ys[:num_samples], dtype=torch.long)
    permutation = torch.randperm(num_samples, generator=generator)
    return x[permutation], y[permutation]


def make_pretrain_dataloaders(config):
    """Build train and verification dataloaders for the toy pretraining demo.

    Args:
        config: Configuration object containing num_samples, class_num, length,
            noise_std, random, verify_ratio, and batch_size.

    Returns:
        A tuple (train_loader, verify_loader).
    """

    x, y = make_synthetic_waveform_dataset(
        num_samples=config.num_samples,
        num_classes=config.class_num,
        input_length=config.length,
        noise_std=getattr(config, "noise_std", 0.03),
        seed=config.random,
    )
    num_total = x.size(0)
    num_verify = max(1, int(num_total * config.verify_ratio))
    train_x, verify_x = x[:-num_verify], x[-num_verify:]
    train_y, verify_y = y[:-num_verify], y[-num_verify:]
    train_loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
    )
    verify_loader = DataLoader(
        TensorDataset(verify_x, verify_y),
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
    )
    return train_loader, verify_loader


def make_federated_dataloaders(config):
    """Build federated dataloaders directly from synthetic data.

    Args:
        config: Federated-learning configuration object.

    Returns:
        A tuple (client_loaders, all_train_loader, test_loader, train_dataset).
    """

    train_total, train_data_idx, test_set = prepare_synthetic_data(config)
    return build_federated_dataloaders(config, train_total, train_data_idx, test_set)


def prepare_synthetic_data(config):
    """Prepare synthetic train/test data and client index splits.

    It keeps the same high-level contract:
        train_total, train_data_idx, test_set = prepare_data(config)

    The original code reads Excel data and then creates client index lists. This
    version generates toy waveforms first, then applies the same class-wise test
    isolation and IID/non-IID client-index allocation logic.

    Args:
        config: Federated-learning configuration. The function reads fields such
            as num_classes, num_clients, test_ratio, niid, balance, alpha,
            least_samples, num_samples, length, noise_std, seed, and print_flags.

    Returns:
        A tuple (train_total, data_idx, test_set). train_total and test_set are
        TensorDataset instances. data_idx is a list of tensors, one per client,
        containing indices into train_total.
    """

    num_classes = config.num_classes
    num_clients = config.num_clients
    test_ratio = config.test_ratio
    niid = getattr(config, "niid", False)
    balance = getattr(config, "balance", True)
    alpha = getattr(config, "alpha", 0.1)
    least_samples = getattr(config, "least_samples", 1)
    print_flags = getattr(config, "print_flags", [0, 0, 0, 0])

    x_all, y_all = make_synthetic_waveform_dataset(
        num_samples=config.num_samples,
        num_classes=num_classes,
        input_length=config.length,
        noise_std=getattr(config, "noise_std", 0.03),
        seed=config.seed,
    )

    np.random.seed(config.seed)
    labels_all = y_all.numpy()
    test_idx = []
    for label in range(num_classes):
        idx_k = np.where(labels_all == label)[0]
        np.random.shuffle(idx_k)
        n_test = int(np.ceil(test_ratio * len(idx_k)))
        test_idx.extend(idx_k[:n_test])

    test_idx = np.array(sorted(set(test_idx)))
    train_mask = np.ones(len(labels_all), dtype=bool)
    train_mask[test_idx] = False
    train_idx = np.where(train_mask)[0]

    train_x = x_all[torch.from_numpy(train_idx).long()]
    train_y = y_all[torch.from_numpy(train_idx).long()]
    test_x = x_all[torch.from_numpy(test_idx).long()]
    test_y = y_all[torch.from_numpy(test_idx).long()]

    train_total = TensorDataset(train_x, train_y)
    test_set = TensorDataset(test_x, test_y)

    # Preserve the original side effect style for downstream code.
    config.num_classes = num_classes
    config.num_clients = num_clients
    config.batch_size = config.batch_size
    config.test_ratio = test_ratio
    config.niid = niid
    config.balance = balance
    config.alpha = alpha
    config.least_samples = least_samples

    if len(print_flags) > 0 and print_flags[0]:
        print("\n" + "=" * 30 + " Synthetic data preparation " + "=" * 30)
        print(f"Samples: train={len(train_total)}, test={len(test_set)}")
        print(f"Classes: {num_classes}, clients: {num_clients}")
        print(f"Split mode: {'non-IID Dirichlet' if niid else 'IID'}")

    dataset_label = train_y.numpy()
    dataidx_map = {}

    if not niid:
        idxs = np.array(range(len(dataset_label)))
        idx_for_each_class = [idxs[dataset_label == label] for label in range(num_classes)]
        class_num_per_client = [num_classes for _ in range(num_clients)]

        for label in range(num_classes):
            selected_clients = [
                client for client in range(num_clients) if class_num_per_client[client] > 0
            ]
            num_all_samples = len(idx_for_each_class[label])
            if num_all_samples < len(selected_clients):
                selected_clients = list(
                    np.random.choice(selected_clients, size=num_all_samples, replace=False)
                )
            num_selected_clients = len(selected_clients)
            if num_selected_clients == 0:
                continue

            num_per = num_all_samples / num_selected_clients
            if balance:
                num_samples = [int(num_per) for _ in range(num_selected_clients - 1)]
            else:
                if int(num_per) <= 1:
                    num_samples = [1 for _ in range(num_selected_clients - 1)]
                else:
                    while True:
                        num_samples = np.random.randint(
                            least_samples, int(num_per), num_selected_clients - 1
                        ).tolist()
                        if sum(num_samples) < num_all_samples:
                            break
            num_samples.append(num_all_samples - sum(num_samples))

            start = 0
            for client, num_sample in zip(selected_clients, num_samples):
                current = idx_for_each_class[label][start : start + num_sample]
                if client not in dataidx_map:
                    dataidx_map[client] = current
                else:
                    dataidx_map[client] = np.append(dataidx_map[client], current, axis=0)
                start += num_sample
                class_num_per_client[client] -= 1

    elif niid is True:
        min_size = 0
        total_train = len(dataset_label)
        while min_size < least_samples:
            idx_batch = [[] for _ in range(num_clients)]
            for label in range(num_classes):
                idx_k = np.where(dataset_label == label)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
                proportions = np.array(
                    [
                        p * (len(idx_j) < total_train / num_clients)
                        for p, idx_j in zip(proportions, idx_batch)
                    ]
                )
                if proportions.sum() == 0:
                    proportions = np.repeat(1.0 / num_clients, num_clients)
                else:
                    proportions = proportions / proportions.sum()
                split_points = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [
                    idx_j + idx.tolist()
                    for idx_j, idx in zip(idx_batch, np.split(idx_k, split_points))
                ]
            min_size = min(len(idx_j) for idx_j in idx_batch)

        for client in range(num_clients):
            dataidx_map[client] = idx_batch[client]
    else:
        raise NotImplementedError("Only IID (niid=False) and non-IID (niid=True) are supported.")

    data_idx = []
    for client in range(num_clients):
        idxs = np.array(dataidx_map.get(client, []), dtype=np.int64)
        data_idx.append(torch.from_numpy(idxs))
        if len(print_flags) > 1 and print_flags[1]:
            labels_client = dataset_label[idxs] if len(idxs) else np.array([])
            print(f"\nClient {client} data distribution:")
            for label in np.unique(labels_client):
                print(f"  class {int(label)}: {int(np.sum(labels_client == label))}")
            print(f"  total: {len(labels_client)}")

    return train_total, data_idx, test_set


def build_federated_dataloaders(config, train_total, train_data_idx, test_set):
    """Create PyTorch dataloaders from prepared federated splits.

    Args:
        config: Federated-learning configuration with batch_size and num_clients.
        train_total: TensorDataset containing all client training samples.
        train_data_idx: List of client-specific index tensors into train_total.
        test_set: TensorDataset containing the held-out test samples.

    Returns:
        A tuple (client_loaders, all_train_loader, test_loader, train_dataset).
        client_loaders contains one DataLoader per client. train_dataset contains
        the corresponding Subset objects.
    """

    dataset_test = Subset(test_set, list(range(len(test_set))))
    test_loader = DataLoader(
        dataset_test,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
    )

    dataset_train_all = Subset(train_total, list(range(len(train_total))))
    all_train_loader = DataLoader(
        dataset_train_all,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
    )

    train_dataset = []
    client_loaders = []
    for idx in range(config.num_clients):
        train_dataset_idx = Subset(train_total, train_data_idx[idx])
        train_dataset.append(train_dataset_idx)
        client_loaders.append(
            DataLoader(
                train_dataset_idx,
                batch_size=config.batch_size,
                drop_last=False,
                shuffle=True,
                num_workers=0,
            )
        )

    return client_loaders, all_train_loader, test_loader, train_dataset


def evaluate_pretrain_model(model, data_loader, device, input_norm=True):
    """Evaluate the pretrained feature extractor classifier branch.

    Args:
        model: Patch residual feature extractor with a classifier head.
        data_loader: DataLoader yielding (waveform, label) batches.
        device: torch.device used for evaluation.
        input_norm: Whether to apply per-sample normalization before inference.

    Returns:
        Classification accuracy as a float in [0, 1].
    """

    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device)
            if input_norm:
                x, _, _ = normalize_per_sample(x)
            logits, _ = model(x)
            pred = torch.argmax(logits, dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return correct / max(total, 1)
