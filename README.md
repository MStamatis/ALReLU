# ALReLU

Monorepo with two separate PyPI packages for ALReLU:

1. `alrelu-keras` (repo root)
2. `alrelu-torch` (`packages/alrelu-torch`)

Both use the same core formula:

`ALReLU(x, alpha) = max(abs(alpha * x), x)`

## Reference

ALReLU paper:

`ALReLU: A different approach on Leaky ReLU activation function to improve Neural Networks Performance`

https://arxiv.org/abs/2012.07564

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

## Training Examples (MNIST)

Scripts:

- Keras + TensorFlow: `scripts/train_mnist_keras_tf.py`
- PyTorch: `scripts/train_mnist_torch.py`

Direct links:

- https://github.com/MStamatis/ALReLU/blob/main/scripts/train_mnist_keras_tf.py
- https://github.com/MStamatis/ALReLU/blob/main/scripts/train_mnist_torch.py

Run commands (from repo root):

```bash
# Keras + TensorFlow
python scripts/train_mnist_keras_tf.py --epochs 5 --variant fixed --alpha 0.01
python scripts/train_mnist_keras_tf.py --epochs 5 --variant learnable --alpha 0.01

# PyTorch
python scripts/train_mnist_torch.py --epochs 5 --variant fixed --alpha 0.01
python scripts/train_mnist_torch.py --epochs 5 --variant learnable --alpha 0.01
```

## License

MIT License (`LICENSE`).
