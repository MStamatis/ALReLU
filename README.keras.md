# alrelu-keras

PyPI-installable Keras package for the ALReLU activation with 2 variants:

1. `ALReLU` (default): current behavior from this project, with fixed `alpha=0.01`.
2. `TrainableALReLU`: same formula with trainable `alpha`.

Formula used by both variants:

`ALReLU(x, alpha) = max(abs(alpha * x), x)`

## Reference

ALReLU paper:

`ALReLU: A different approach on Leaky ReLU activation function to improve Neural Networks Performance`

https://arxiv.org/abs/2012.07564

## Installation

```bash
pip install alrelu-keras
```

Keras requires a backend. For the common TensorFlow backend:

```bash
pip install alrelu-keras[tensorflow]
```

From source:

```bash
pip install .
```

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

## Usage

### 1) Default ALReLU (current implementation behavior)

```python
import keras
from alrelu_keras import ALReLU, alrelu

# As activation function
x = keras.ops.array([-2.0, -1.0, 0.0, 1.0], dtype="float32")
y = alrelu(x)  # alpha=0.01 by default

# As Keras layer
model = keras.Sequential(
    [
        keras.layers.Input(shape=(32,)),
        keras.layers.Dense(64),
        ALReLU(),  # fixed alpha
        keras.layers.Dense(10),
    ]
)
```

### 2) Trainable alpha variant

```python
import keras
from alrelu_keras import TrainableALReLU

model = keras.Sequential(
    [
        keras.layers.Input(shape=(32,)),
        keras.layers.Dense(64),
        TrainableALReLU(alpha_initializer=0.01),  # alpha is trainable
        keras.layers.Dense(10),
    ]
)
```

## Serialization / model loading

Both `ALReLU`, `TrainableALReLU`, and `alrelu` are registered as Keras-serializable objects.
You can save/load models without manually passing `custom_objects` in standard Keras workflows.

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

## License

MIT License. See `LICENSE`.
