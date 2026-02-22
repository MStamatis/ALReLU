# ALReLU

Monorepo with two separate PyPI packages for ALReLU:

1. `alrelu-keras` (repo root)
2. `alrelu-torch` (`packages/alrelu-torch`)

Both use the same core formula:

`ALReLU(x, alpha) = max(abs(alpha * x), x)`

## Package 1: alrelu-keras

Install:

```bash
pip install alrelu-keras
```

Keras with TensorFlow backend:

```bash
pip install alrelu-keras[tensorflow]
```

Docs:

- PyPI README source: `README.keras.md`
- Code: `src/alrelu_keras`

## Package 2: alrelu-torch

Install:

```bash
pip install alrelu-torch
```

Code and docs:

- Package root: `packages/alrelu-torch`
- PyPI README source: `packages/alrelu-torch/README.md`

## Separate PyPI Publishing

### Publish `alrelu-keras` (from repo root)

```bash
python -m build
python -m twine check dist/*
python -m twine upload dist/*
```

### Publish `alrelu-torch` (from its folder)

```bash
cd packages/alrelu-torch
python -m build
python -m twine check dist/*
python -m twine upload dist/*
```

TestPyPI first (recommended):

```bash
python -m twine upload --repository testpypi dist/*
```

## License

MIT License (`LICENSE`).
