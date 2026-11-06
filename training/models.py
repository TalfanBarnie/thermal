"""Model candidates. Each `fit_*` returns the artefact dict understood by thermal.model."""

from __future__ import annotations

import itertools
import logging
from datetime import UTC, datetime
from typing import Any

import numpy as np

from thermal.model import LOGVAR_MAX, LOGVAR_MIN, poly_features

log = logging.getLogger(__name__)


def standardise(X: np.ndarray, y: np.ndarray):
    x_mean = X.mean(axis=0).astype(np.float64)
    x_std = X.std(axis=0).astype(np.float64)
    x_std[x_std < 1e-8] = 1.0
    y_mean, y_std = float(y.mean()), float(y.std() or 1.0)
    return x_mean, x_std, y_mean, y_std


def _meta(model_type: str, names: list[str], sat: dict[str, Any], platform: str, **extra):
    return {
        "model_type": model_type,
        "feature_names": list(names),
        "target": sat["mir_channel"],
        "sat": sat["key"],
        "platform": platform,
        "created": datetime.now(UTC).isoformat(),
        **extra,
    }


def monomial_powers(n_features: int, degree: int) -> np.ndarray:
    """All monomials of total degree 1..degree (no constant term)."""
    rows = []
    for d in range(1, degree + 1):
        for combo in itertools.combinations_with_replacement(range(n_features), d):
            p = np.zeros(n_features, dtype=np.int64)
            for i in combo:
                p[i] += 1
            rows.append(p)
    return np.array(rows)


def fit_ridge_poly(
    X: np.ndarray,
    y: np.ndarray,
    names: list[str],
    sat,
    platform: str,
    degree: int = 2,
    alpha: float = 1.0,
) -> dict[str, Any]:
    from sklearn.linear_model import Ridge

    x_mean, x_std, y_mean, y_std = standardise(X, y)
    xs = (X - x_mean) / x_std
    ys = (y - y_mean) / y_std
    powers = monomial_powers(X.shape[1], degree)
    P = poly_features(xs, powers)
    p_mean, p_std = P.mean(axis=0), P.std(axis=0)
    p_std[p_std < 1e-8] = 1.0
    Pn = (P - p_mean) / p_std
    reg = Ridge(alpha=alpha).fit(Pn, ys)
    resid = ys - reg.predict(Pn)
    reg_v = Ridge(alpha=alpha * 10).fit(Pn, np.log(resid**2 + 1e-6))
    # fold the P standardisation into the weights: w_n/(p_std) and bias - sum(w*p_mean/p_std)
    w_mean = reg.coef_ / p_std
    b_mean = float(reg.intercept_ - np.sum(reg.coef_ * p_mean / p_std))
    w_logvar = reg_v.coef_ / p_std
    b_logvar = float(reg_v.intercept_ - np.sum(reg_v.coef_ * p_mean / p_std))
    return {
        "meta": _meta("ridge_poly", names, sat, platform, degree=degree, alpha=alpha),
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "powers": powers,
        "w_mean": w_mean,
        "b_mean": b_mean,
        "w_logvar": w_logvar,
        "b_logvar": b_logvar,
    }


def fit_mlp(
    X: np.ndarray,
    y: np.ndarray,
    names: list[str],
    sat,
    platform: str,
    hidden: tuple[int, ...] = (128, 128, 128),
    epochs: int = 30,
    batch_size: int = 4096,
    lr: float = 1e-3,
    seed: int = 0,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    patience: int = 5,
) -> dict[str, Any]:
    import torch
    from torch import nn

    torch.manual_seed(seed)
    x_mean, x_std, y_mean, y_std = standardise(X, y)
    xs = torch.tensor((X - x_mean) / x_std, dtype=torch.float32)
    ys = torch.tensor((y - y_mean) / y_std, dtype=torch.float32)
    layers: list[nn.Module] = []
    d = X.shape[1]
    for h in hidden:
        layers += [nn.Linear(d, h), nn.GELU(approximate="tanh")]
        d = h
    layers.append(nn.Linear(d, 2))
    net = nn.Sequential(*layers)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    def nll(out, target):
        mean, logvar = out[:, 0], out[:, 1].clamp(LOGVAR_MIN, LOGVAR_MAX)
        return (0.5 * (logvar + (target - mean) ** 2 * torch.exp(-logvar))).mean()

    val = None
    if X_val is not None and y_val is not None and len(y_val):
        val = (
            torch.tensor((X_val - x_mean) / x_std, dtype=torch.float32),
            torch.tensor((y_val - y_mean) / y_std, dtype=torch.float32),
        )
    best_state, best_val, bad = None, float("inf"), 0
    n = len(ys)
    history = []
    for epoch in range(epochs):
        net.train()
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, batch_size):
            b = perm[i : i + batch_size]
            opt.zero_grad()
            loss = nll(net(xs[b]), ys[b])
            loss.backward()
            opt.step()
            total += float(loss) * len(b)
        sched.step()
        rec = {"epoch": epoch + 1, "train_nll": total / n}
        if val is not None:
            net.eval()
            with torch.no_grad():
                v = float(nll(net(val[0]), val[1]))
            rec["val_nll"] = v
            if v < best_val - 1e-4:
                best_val, bad = v, 0
                best_state = {k: t.clone() for k, t in net.state_dict().items()}
            else:
                bad += 1
        history.append(rec)
        log.info("epoch %s", rec)
        if val is not None and bad >= patience:
            log.info("early stopping")
            break
    if best_state is not None:
        net.load_state_dict(best_state)
    arrays: dict[str, Any] = {
        "meta": _meta(
            "mlp",
            names,
            sat,
            platform,
            hidden=list(hidden),
            n_layers=len(hidden) + 1,
            epochs_run=len(history),
            history=history,
        ),
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
    }
    linear = [m for m in net if isinstance(m, nn.Linear)]
    for i, lin in enumerate(linear):
        arrays[f"W{i}"] = lin.weight.detach().numpy().T.astype(np.float32)
        arrays[f"b{i}"] = lin.bias.detach().numpy().astype(np.float32)
    return arrays
