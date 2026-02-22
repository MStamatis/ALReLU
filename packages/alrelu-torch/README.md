# alrelu-torch

PyPI-installable PyTorch package for the ALReLU activation with 2 variants:

1. `ALReLU` (default): fixed `alpha=0.01`
2. `TrainableALReLU`: trainable `alpha` parameter

Formula:

`ALReLU(x, alpha) = max(abs(alpha * x), x)`

## Installation

After publishing:

```bash
pip install alrelu-torch
```

From source:

```bash
pip install .
```

## Usage

### 1) Functional API

```python
import torch
from alrelu_torch import alrelu

x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
y = alrelu(x)  # alpha=0.01 by default
```

### 2) Fixed module

```python
import torch.nn as nn
from alrelu_torch import ALReLU

model = nn.Sequential(
    nn.Linear(32, 64),
    ALReLU(alpha=0.01),
    nn.Linear(64, 10),
)
```

### 3) Trainable alpha module

```python
import torch.nn as nn
from alrelu_torch import TrainableALReLU

model = nn.Sequential(
    nn.Linear(32, 64),
    TrainableALReLU(alpha_init=0.01, non_negative=True),
    nn.Linear(64, 10),
)
```

## Development

Install dev tools:

```bash
pip install -e .[dev]
```

Run tests:

```bash
pytest
```

Build package:

```bash
python -m build
```

## Publish to PyPI

1. Bump version in `pyproject.toml` and `src/alrelu_torch/version.py`.
2. Build distributions:

```bash
python -m build
```

3. Validate metadata:

```bash
python -m twine check dist/*
```

4. Upload to TestPyPI (recommended first):

```bash
python -m twine upload --repository testpypi dist/*
```

5. Upload to PyPI:

```bash
python -m twine upload dist/*
```
