"""Analytical local update, cloud aggregation, and evaluation functions."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from analytical_nilm.utils import normalize_per_sample


class LinearAnalyticalHead(nn.Module):
    """Linear analytical classifier head.

    The identity branch returns the analytical representation, and the linear
    branch returns class logits.

    Args:
        in_features: Analytical representation dimension.
        num_classes: Number of output appliance classes.
    """

    def __init__(self, in_features, num_classes):
        """Initialize a bias-free linear analytical classifier.

        Args:
            in_features: Analytical representation dimension.
            num_classes: Number of output appliance classes.
        """

        super().__init__()
        self.act = nn.Identity()
        self.fc = nn.Linear(in_features, num_classes, bias=False)

    def forward(self, x):
        """Return the analytical representation and class logits.

        Args:
            x: Analytical representation tensor with shape
                [batch_size, in_features].

        Returns:
            A tuple (x_act, x_fc), where x_act is the identity representation
            and x_fc contains class logits with shape [batch_size, num_classes].
        """

        x_act = self.act(x)
        x_fc = self.fc(x_act)
        return x_act, x_fc


def init_local(config, device):
    """Initialize the analytical head without the projection branch.

    Args:
        config: Federated configuration with feat_size and num_classes.
        device: torch.device where the head should be placed.

    Returns:
        A LinearAnalyticalHead instance on the selected device.
    """

    return LinearAnalyticalHead(config.feat_size, config.num_classes).to(device)


def init_local_projection(config, device):
    """Initialize the analytical head for the projection branch.

    The projection branch concatenates the raw waveform and extracted feature,
    so the analytical dimension is feature_dim + input_length.

    Args:
        config: Federated configuration with feat_size, length, and num_classes.
        device: torch.device where the head should be placed.

    Returns:
        A LinearAnalyticalHead instance on the selected device.
    """

    return LinearAnalyticalHead(config.feat_size + config.length, config.num_classes).to(device)


def local_update(train_loader, model, global_model, config, device):
    """Compute one client's closed-form local update.

    Steps:
    1. Extract local representations from each client batch.
    2. Convert labels to one-hot vectors.
    3. Compute C = X^T X + lambda I.
    4. Compute R = C^{-1}.
    5. Compute W = R X^T Y.

    Args:
        train_loader: Client DataLoader yielding (waveform, label) batches.
        model: Frozen feature extractor.
        global_model: Analytical head used to expose the identity representation.
        config: Federated configuration with input_norm, feature_scale, rg,
            feat_size, and num_classes.
        device: torch.device used for matrix computation.

    Returns:
        A tuple (W, R, C, reps_all, label_onehot_all), where W is the local
        analytical weight matrix, R is the inverse regularized covariance, C is
        the regularized covariance, reps_all is the local representation matrix,
        and label_onehot_all is the local one-hot label matrix.
    """

    with torch.no_grad():
        # Accumulate representations and one-hot labels batch by batch.
        reps_list = []
        label_onehot_list = []

        for train_x, train_y in train_loader:
            train_x = train_x.to(device).float()
            train_y = train_y.long().to(device)

            if config.input_norm:
                train_x, train_x_min, train_x_max = normalize_per_sample(train_x)

            _, reps = model(train_x, train=False)
            if config.input_norm and config.feature_scale:
                reps = reps * (train_x_max - train_x_min)

            reps, _ = global_model(reps)
            reps = reps.double()
            label_onehot = F.one_hot(train_y, config.num_classes).double()

            reps_list.append(reps.detach().cpu())
            label_onehot_list.append(label_onehot.detach().cpu())

        # Concatenate all local batches on the selected device.
        if len(reps_list) > 0:
            reps_all = torch.cat(reps_list, dim=0).to(device)
        else:
            reps_all = torch.empty((0, config.feat_size), dtype=torch.double, device=device)

        if len(label_onehot_list) > 0:
            label_onehot_all = torch.cat(label_onehot_list, dim=0).to(device)
        else:
            label_onehot_all = torch.empty((0, config.num_classes), dtype=torch.double, device=device)

        # Build the regularized covariance matrix and solve the analytical head.
        xtx_reg = reps_all.t() @ reps_all + config.rg * torch.eye(
            reps_all.size(1), dtype=torch.double, device=device
        )
        chol = torch.linalg.cholesky(xtx_reg)
        R = torch.cholesky_inverse(chol)
        W = R @ reps_all.t().double() @ label_onehot_all.double()
        C = xtx_reg.cpu()
        R = R.cpu()

    return W, R, C, reps_all, label_onehot_all


def local_update_projection(train_loader, model, global_model, config, device):
    """Compute one client's closed-form local update with projection features.

    The raw input waveform is concatenated with the extracted representation
    before solving the analytical classifier:

        X = concat(raw_waveform, extracted_feature)

    Args:
        train_loader: Client DataLoader yielding (waveform, label) batches.
        model: Frozen feature extractor.
        global_model: Analytical head used to expose the identity representation.
        config: Federated configuration with input_norm, feature_scale, rg,
            feat_size, length, and num_classes.
        device: torch.device used for matrix computation.

    Returns:
        A tuple (W, R, C, reps_all, label_onehot_all), where W is the local
        analytical weight matrix, R is the inverse regularized covariance, C is
        the regularized covariance, reps_all is the local projection-branch
        representation matrix, and label_onehot_all is the local one-hot label
        matrix.
    """

    with torch.no_grad():
        # Accumulate projection-branch representations and one-hot labels.
        reps_list = []
        label_onehot_list = []

        for train_x, train_y in train_loader:
            train_x = train_x.to(device).float()
            train_y = train_y.long().to(device)

            if config.input_norm:
                train_x_norm, train_x_min, train_x_max = normalize_per_sample(train_x)
            else:
                train_x_norm = train_x

            _, reps = model(train_x_norm, train=False)
            if config.input_norm and config.feature_scale:
                reps = reps * (train_x_max - train_x_min)

            reps = torch.cat([train_x, reps], dim=1)
            reps, _ = global_model(reps)
            reps = reps.double()
            label_onehot = F.one_hot(train_y, config.num_classes).double()

            reps_list.append(reps.detach().cpu())
            label_onehot_list.append(label_onehot.detach().cpu())

        # Concatenate all local samples before computing the closed-form update.
        if len(reps_list) > 0:
            reps_all = torch.cat(reps_list, dim=0).to(device)
        else:
            reps_all = torch.empty(
                (0, config.feat_size + config.length), dtype=torch.double, device=device
            )

        if len(label_onehot_list) > 0:
            label_onehot_all = torch.cat(label_onehot_list, dim=0).to(device)
        else:
            label_onehot_all = torch.empty((0, config.num_classes), dtype=torch.double, device=device)

        # Build the regularized covariance matrix and solve the analytical head.
        xtx_reg = reps_all.t() @ reps_all + config.rg * torch.eye(
            reps_all.size(1), dtype=torch.double, device=device
        )
        chol = torch.linalg.cholesky(xtx_reg)
        R = torch.cholesky_inverse(chol)
        W = R @ reps_all.t().double() @ label_onehot_all.double()
        C = xtx_reg.cpu()
        R = R.cpu()

    return W, R, C, reps_all, label_onehot_all


def global_update(train_loader, model, global_model, config, device):
    """Compute the direct centralized analytical solution.

    This computes the classifier using all training samples at once. It is used
    to compare the aggregated model with the direct global analytical solution.

    Args:
        train_loader: DataLoader over all training samples.
        model: Frozen feature extractor.
        global_model: Analytical head used to expose the identity representation.
        config: Federated configuration with input_norm, feature_scale, rg, and
            num_classes.
        device: torch.device used for matrix computation.

    Returns:
        A tuple (W, reps_all, label_onehot_all), where W is the centralized
        analytical weight matrix, reps_all is the full representation matrix,
        and label_onehot_all is the full one-hot label matrix.
    """

    with torch.no_grad():
        # Collect all representations and labels from the full training loader.
        reps_list = []
        label_onehot_list = []

        for train_x, train_y in train_loader:
            train_x = train_x.to(device).float()
            train_y = train_y.long().to(device)

            if config.input_norm:
                train_x, train_x_min, train_x_max = normalize_per_sample(train_x)

            _, reps = model(train_x, train=False)
            if config.input_norm and config.feature_scale:
                reps = reps * (train_x_max - train_x_min)
            reps, _ = global_model(reps)

            reps_list.append(reps.double().detach().cpu())
            label_onehot_list.append(F.one_hot(train_y, config.num_classes).double().detach().cpu())

        reps_all = torch.cat(reps_list, dim=0).to(device).double()
        label_onehot_all = torch.cat(label_onehot_list, dim=0).to(device).double()

    # Solve X^T X W = X^T Y. Add regularization if X^T X is rank deficient.
    XtX = reps_all.t() @ reps_all
    XtY = reps_all.t() @ label_onehot_all
    rank = torch.linalg.matrix_rank(XtX)
    n = XtX.size(0)
    if rank == n:
        try:
            W = torch.linalg.solve(XtX, XtY)
        except RuntimeError:
            W = torch.pinverse(reps_all) @ label_onehot_all
    else:
        A = XtX + float(config.rg) * torch.eye(n, dtype=torch.double, device=device)
        try:
            W = torch.linalg.solve(A, XtY)
        except RuntimeError:
            W = torch.pinverse(A) @ XtY

    return W, reps_all, label_onehot_all


def global_update_projection(train_loader, model, global_model, config, device):
    """Compute the direct centralized analytical solution with projection features.

    The raw waveform and extracted representation are concatenated before the
    direct global analytical solution is computed.

    Args:
        train_loader: DataLoader over all training samples.
        model: Frozen feature extractor.
        global_model: Analytical head used to expose the identity representation.
        config: Federated configuration with input_norm, feature_scale, rg,
            length, and num_classes.
        device: torch.device used for matrix computation.

    Returns:
        A tuple (W, reps_all, label_onehot_all), where W is the centralized
        analytical weight matrix, reps_all is the full projection-branch
        representation matrix, and label_onehot_all is the full one-hot label
        matrix.
    """

    with torch.no_grad():
        # Collect all projection-branch representations and labels.
        reps_list = []
        label_onehot_list = []

        for train_x, train_y in train_loader:
            train_x = train_x.to(device).float()
            train_y = train_y.long().to(device)

            if config.input_norm:
                train_x_norm, train_x_min, train_x_max = normalize_per_sample(train_x)
            else:
                train_x_norm = train_x

            _, reps = model(train_x_norm, train=False)
            if config.input_norm and config.feature_scale:
                reps = reps * (train_x_max - train_x_min)
            reps = torch.cat([train_x, reps], dim=1)
            reps, _ = global_model(reps)

            reps_list.append(reps.double().detach().cpu())
            label_onehot_list.append(F.one_hot(train_y, config.num_classes).double().detach().cpu())

        reps_all = torch.cat(reps_list, dim=0).to(device).double()
        label_onehot_all = torch.cat(label_onehot_list, dim=0).to(device).double()

    # Solve X^T X W = X^T Y. Add regularization if X^T X is rank deficient.
    XtX = reps_all.t() @ reps_all
    XtY = reps_all.t() @ label_onehot_all
    rank = torch.linalg.matrix_rank(XtX)
    n = XtX.size(0)
    if rank == n:
        try:
            W = torch.linalg.solve(XtX, XtY)
        except RuntimeError:
            W = torch.pinverse(reps_all) @ label_onehot_all
    else:
        A = XtX + float(config.rg) * torch.eye(n, dtype=torch.double, device=device)
        try:
            W = torch.linalg.solve(A, XtY)
        except RuntimeError:
            W = torch.pinverse(A) @ XtY

    return W, reps_all, label_onehot_all


def aggregation1(W, R, C, config, device):
    """Aggregate client analytical solutions on the cloud side.

        Ct = sum_k Ck
        St = sum_k Ck @ Wk
        Rt = Ct^{-1}
        Wt = Rt @ St

    Args:
        W: List of local analytical weight matrices.
        R: List of local inverse covariance matrices. Used directly when there
            is only one client.
        C: List of local regularized covariance matrices.
        config: Federated configuration object. Kept for API consistency.
        device: torch.device used for aggregation.

    Returns:
        A tuple (Wt, Rt, Ct), where Wt is the aggregated analytical weight
        matrix, Rt is the inverse aggregated covariance, and Ct is the aggregated
        covariance.
    """

    if len(W) < 2:
        return (
            W[0].to(device, dtype=torch.double),
            R[0].to(device, dtype=torch.double),
            C[0].to(device, dtype=torch.double),
        )

    # Initialize accumulated covariance and weighted classifier sum.
    Ct = C[0].to(device, dtype=torch.double)
    St = Ct @ W[0].to(device, dtype=torch.double)

    # Accumulate all clients.
    for idx in range(1, len(W)):
        Ck = C[idx].to(device, dtype=torch.double)
        Wk = W[idx].to(device, dtype=torch.double)
        Ct = Ct + Ck
        St = St + Ck @ Wk

    # Compute the aggregated analytical classifier.
    Rt = torch.inverse(Ct)
    Wt = Rt @ St
    return Wt, Rt, Ct


def clean_regularization(W, C, config, device):
    """Remove the regularization term from an aggregated non-projection solution.

    Args:
        W: Aggregated analytical weight matrix.
        C: Aggregated regularized covariance matrix.
        config: Federated configuration with num_clients, rg, and feat_size.
        device: torch.device used for matrix inversion.

    Returns:
        Analytical weight matrix after regularization cleanup.
    """

    # Recover the unregularized covariance estimate.
    x = C.to(device, dtype=torch.double) - config.num_clients * config.rg * torch.eye(
        config.feat_size, dtype=torch.double, device=device
    )
    # Avoid singular diagonal entries before inversion.
    diag = torch.diag(x)
    zero_diag_idx = (diag == 0).nonzero(as_tuple=True)[0]
    if zero_diag_idx.numel() > 0:
        x[zero_diag_idx, zero_diag_idx] = 1e-8
    # Apply the regularization-cleaning transform to the aggregated weights.
    R_origin = torch.inverse(x)
    return W.to(device, dtype=torch.double) + (config.num_clients * config.rg * R_origin) @ W.to(
        device, dtype=torch.double
    )


def clean_regularization_projection(W, C, config, device):
    """Remove the regularization term from a projection-branch solution.

    Args:
        W: Aggregated analytical weight matrix.
        C: Aggregated regularized covariance matrix.
        config: Federated configuration with num_clients, rg, feat_size, and
            length.
        device: torch.device used for matrix inversion.

    Returns:
        Projection-branch analytical weight matrix after regularization cleanup.
    """

    # Projection branch uses raw waveform plus extracted feature.
    dim = config.feat_size + config.length
    # Recover the unregularized covariance estimate.
    x = C.to(device, dtype=torch.double) - config.num_clients * config.rg * torch.eye(
        dim, dtype=torch.double, device=device
    )
    # Avoid singular diagonal entries before inversion.
    diag = torch.diag(x)
    zero_diag_idx = (diag == 0).nonzero(as_tuple=True)[0]
    if zero_diag_idx.numel() > 0:
        x[zero_diag_idx, zero_diag_idx] = 1e-8
    # Apply the regularization-cleaning transform to the aggregated weights.
    R_origin = torch.inverse(x)
    return W.to(device, dtype=torch.double) + (config.num_clients * config.rg * R_origin) @ W.to(
        device, dtype=torch.double
    )


def set_head_weight(head, W):
    """Write an analytical weight matrix into a LinearAnalyticalHead.

    Args:
        head: LinearAnalyticalHead instance to update.
        W: Analytical weight matrix with shape [in_features, num_classes].

    Returns:
        The same head instance with fc.weight updated.
    """

    head.fc.weight = torch.nn.Parameter(torch.t(W.float()).contiguous())
    return head


def diff(W_agg, W_total):
    """Compute the L1 difference between two analytical weight matrices.

    Args:
        W_agg: First analytical weight matrix.
        W_total: Second analytical weight matrix.

    Returns:
        Scalar tensor equal to sum(abs(W_total - W_agg)).
    """

    return torch.sum(torch.abs(W_total - W_agg))


def evaluate_projection(val_loader, model, global_model, config, device):
    """Evaluate an analytical classifier with projection-branch features.

    Args:
        val_loader: DataLoader yielding (waveform, label) batches.
        model: Frozen feature extractor.
        global_model: Analytical head whose weights are already set.
        config: Federated configuration with input_norm.
        device: torch.device used for inference.

    Returns:
        Classification accuracy as a float in [0, 1].
    """

    model.eval()
    global_model.eval()
    total = 0
    correct = 0

    with torch.no_grad():
        for train_x, train_y in val_loader:
            train_x = train_x.to(device).float()
            train_y = train_y.long().to(device)

            if config.input_norm:
                train_x_norm, _, _ = normalize_per_sample(train_x)
            else:
                train_x_norm = train_x

            _, reps = model(train_x_norm, train=False)
            reps = torch.cat([train_x, reps], dim=1)
            _, logits = global_model(reps)
            pred = torch.argmax(logits, dim=1)
            correct += (pred == train_y).sum().item()
            total += train_y.numel()

    return correct / max(total, 1)


def evaluate(val_loader, model, global_model, config, device):
    """Evaluate an analytical classifier without projection-branch features.

    Args:
        val_loader: DataLoader yielding (waveform, label) batches.
        model: Frozen feature extractor.
        global_model: Analytical head whose weights are already set.
        config: Federated configuration with input_norm.
        device: torch.device used for inference.

    Returns:
        Classification accuracy as a float in [0, 1].
    """

    model.eval()
    global_model.eval()
    total = 0
    correct = 0

    with torch.no_grad():
        for train_x, train_y in val_loader:
            train_x = train_x.to(device).float()
            train_y = train_y.long().to(device)

            if config.input_norm:
                train_x, _, _ = normalize_per_sample(train_x)

            _, reps = model(train_x, train=False)
            _, logits = global_model(reps)
            pred = torch.argmax(logits, dim=1)
            correct += (pred == train_y).sum().item()
            total += train_y.numel()

    return correct / max(total, 1)
