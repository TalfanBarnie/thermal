import numpy as np
import pytest

from thermal import config
from thermal.model import BackgroundModel, gelu_tanh, poly_features, save_model
from training import models


def _synthetic(n=5000, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 3)).astype(np.float32)
    y = (
        1.5 * X[:, 0] - 0.5 * X[:, 1] ** 2 + 0.3 * X[:, 0] * X[:, 2] + 0.05 * rng.standard_normal(n)
    ).astype(np.float32)
    return X, y


def test_monomial_powers_count():
    p = models.monomial_powers(3, 2)
    assert p.shape == (3 + 6, 3)
    assert poly_features(np.ones((1, 3)), p).shape == (1, 9)


def test_ridge_poly_fit_export_roundtrip(tmp_path):
    X, y = _synthetic()
    sat = config.satellite("msg-0deg")
    arrays = models.fit_ridge_poly(X, y, ["a", "b", "c"], sat, "MSG3", degree=2, alpha=1e-3)
    path = tmp_path / "m.npz"
    save_model(path, arrays)
    m = BackgroundModel.load(path)
    mean, std = m.predict(X, ["a", "b", "c"])
    assert np.sqrt(np.mean((mean - y) ** 2)) < 0.08
    assert np.all(std > 0) and np.median(std) < 0.2
    with pytest.raises(ValueError):
        m.predict(X, ["c", "b", "a"])


def test_mlp_numpy_forward_matches_torch(tmp_path):
    torch = pytest.importorskip("torch")
    X, y = _synthetic(n=3000)
    sat = config.satellite("msg-0deg")
    arrays = models.fit_mlp(
        X,
        y,
        ["a", "b", "c"],
        sat,
        "MSG3",
        hidden=(16, 16),
        epochs=3,
        batch_size=512,
        X_val=X[:500],
        y_val=y[:500],
    )
    path = tmp_path / "mlp.npz"
    save_model(path, arrays)
    m = BackgroundModel.load(path)
    mean, std = m.predict(X[:10], ["a", "b", "c"])
    # rebuild the torch net from the exported weights and compare
    xs = torch.tensor((X[:10] - arrays["x_mean"]) / arrays["x_std"], dtype=torch.float64)
    h = xs
    for i in range(3):
        h = h @ torch.tensor(arrays[f"W{i}"], dtype=torch.float64) + torch.tensor(
            arrays[f"b{i}"], dtype=torch.float64
        )
        if i < 2:
            h = torch.nn.functional.gelu(h, approximate="tanh")
    mean_t = h[:, 0].numpy() * arrays["y_std"] + arrays["y_mean"]
    assert np.allclose(mean, mean_t, atol=1e-6)
    assert np.all(std > 0)


def test_gelu_tanh_basic():
    assert gelu_tanh(np.array([0.0]))[0] == 0.0
    assert gelu_tanh(np.array([3.0]))[0] == pytest.approx(3.0, abs=1e-2)
