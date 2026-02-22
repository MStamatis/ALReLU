import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

torch = pytest.importorskip("torch", exc_type=ImportError)

from alrelu_torch import ALReLU, TrainableALReLU, alrelu


def test_alrelu_default_formula():
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=torch.float32)
    y = alrelu(x)
    expected = torch.tensor([0.02, 0.01, 0.0, 1.0, 2.0], dtype=torch.float32)
    assert torch.allclose(y, expected, atol=1e-6, rtol=1e-6)


def test_fixed_module_respects_alpha():
    layer = ALReLU(alpha=0.2)
    x = torch.tensor([-2.0, 2.0], dtype=torch.float32)
    y = layer(x)
    expected = torch.tensor([0.4, 2.0], dtype=torch.float32)
    assert torch.allclose(y, expected, atol=1e-6, rtol=1e-6)


def test_trainable_alpha_changes_with_optimizer():
    layer = TrainableALReLU(alpha_init=0.01, non_negative=False)
    optimizer = torch.optim.SGD(layer.parameters(), lr=0.1)

    x = torch.tensor([[-1.0], [-2.0], [-3.0]], dtype=torch.float32)
    target = torch.tensor([[0.5], [1.0], [1.5]], dtype=torch.float32)

    alpha_before = float(layer.alpha.detach().cpu())
    for _ in range(5):
        optimizer.zero_grad()
        pred = layer(x)
        loss = torch.mean((pred - target) ** 2)
        loss.backward()
        optimizer.step()
    alpha_after = float(layer.alpha.detach().cpu())

    assert alpha_after != pytest.approx(alpha_before)


def test_non_negative_constraint_is_applied():
    layer = TrainableALReLU(alpha_init=0.01, non_negative=True)
    with torch.no_grad():
        layer.alpha.copy_(torch.tensor(-0.5))
    x = torch.tensor([-2.0], dtype=torch.float32)
    y = layer(x)
    expected = torch.tensor([0.0], dtype=torch.float32)
    assert torch.allclose(y, expected, atol=1e-6, rtol=1e-6)
