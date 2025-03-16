# Neural Network Implementation

This repository contains a flexible implementation of feedforward neural networks with various optimization algorithms.

## Features

- Support for multiple datasets (MNIST, Fashion MNIST)
- Various optimization algorithms (SGD, Momentum, Nesterov, RMSProp, Adam, Nadam)
- Different activation functions (Sigmoid, Tanh, ReLU, Identity)
- Weight initialization methods (Random, Xavier)
- Experiment tracking with Weights & Biases

## Requirements

```
numpy
wandb
keras
tensorflow
pandas
```

### Install the requirements:

```bash
pip install numpy wandb keras tensorflow pandas
```

## Usage

### Download Required Files

To execute the program, ensure you have downloaded both `main.py` and `train.py` files.

### Basic Training

To train a model with default parameters:

```bash
python train.py --wandb_entity myname --wandb_project myprojectname
```

### Hyperparameter Tuning

The script supports various hyperparameters:

```bash
python train.py \
  --wandb_entity myname \
  --wandb_project myprojectname \
  --dataset fashion_mnist \
  --epochs 10 \
  --batch_size 32 \
  --optimizer adam \
  --learning_rate 0.0001 \
  --num_layers 4 \
  --hidden_size 128 \
  --activation tanh \
  --weight_init xavier \
  --weight_decay 0.0005
```

## Supported Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `-wp`, `--wandb_project` | myprojectname | Project name for Weights & Biases |
| `-we`, `--wandb_entity` | myname | Wandb Entity name |
| `-d`, `--dataset` | fashion_mnist | Dataset choice: `["mnist", "fashion_mnist"]` |
| `-e`, `--epochs` | 10 | Number of training epochs |
| `-b`, `--batch_size` | 32 | Batch size for training |
| `-l`, `--loss` | cross_entropy | Loss function: `["mean_squared_error", "cross_entropy"]` |
| `-o`, `--optimizer` | adam | Optimizer: `["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"]` |
| `-lr`, `--learning_rate` | 0.0001 | Learning rate |
| `-m`, `--momentum` | 0.5 | Momentum parameter |
| `-beta`, `--beta` | 0.5 | Beta parameter for RMSProp |
| `-beta1`, `--beta1` | 0.5 | Beta1 parameter for Adam/Nadam |
| `-beta2`, `--beta2` | 0.5 | Beta2 parameter for Adam/Nadam |
| `-eps`, `--epsilon` | 0.000001 | Epsilon parameter for numerical stability |
| `-w_d`, `--weight_decay` | 0.0 | Weight decay (L2 regularization) |
| `-w_i`, `--weight_init` | xavier | Weight initialization: `["random", "Xavier"]` |
| `-nhl`, `--num_layers` | 4 | Number of hidden layers |
| `-sz`, `--hidden_size` | 128 | Number of neurons per hidden layer |
| `-a`, `--activation` | tanh | Activation function: `["identity", "sigmoid", "tanh", "ReLU"]` |

## Suggested Hyperparameter Combinations

### Experiment 1: Effect of Hidden Layers

```bash
python train.py --num_layers 3 --hidden_size 64 --optimizer adam
python train.py --num_layers 4 --hidden_size 64 --optimizer adam
python train.py --num_layers 5 --hidden_size 64 --optimizer adam
```

### Experiment 2: Effect of Optimizer

```bash
python train.py --optimizer sgd --learning_rate 0.001
python train.py --optimizer momentum --learning_rate 0.001
python train.py --optimizer nag --learning_rate 0.001
python train.py --optimizer rmsprop --learning_rate 0.001
python train.py --optimizer adam --learning_rate 0.001
python train.py --optimizer nadam --learning_rate 0.001
```

### Experiment 3: Effect of Activation Function

```bash
python train.py --activation sigmoid
python train.py --activation tanh
python train.py --activation ReLU
```

### Experiment 4: Effect of Weight Initialization

```bash
python train.py --weight_init random
python train.py --weight_init Xavier
```

### Experiment 5: Effect of Weight Decay

```bash
python train.py --weight_decay 0
python train.py --weight_decay 0.0005
python train.py --weight_decay 0.5
```

## Viewing Results

Results will be logged to your Weights & Biases account. You can view them by going to:

```
https://wandb.ai/[your-username]/[your-project-name]
```
