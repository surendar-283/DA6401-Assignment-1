import argparse
import numpy as np
import wandb
from keras.datasets import mnist, fashion_mnist
from keras.utils import to_categorical
import main

def load_dataset(dataset_name):
    """Load and preprocess the specified dataset."""
    if dataset_name == "mnist":
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
    elif dataset_name == "fashion_mnist":
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Normalize pixel values to between 0 and 1
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    
    # Reshape to (samples, features)
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)
    
    # Convert class vectors to binary class matrices
    num_classes = 10
    
    return X_train, X_test, y_train, y_test, num_classes

def train_neural_network(config):
    # Print configuration details
    print("\n" + "="*50)
    print("TRAINING WITH THE FOLLOWING CONFIGURATION:")
    print("="*50)
    for key, value in vars(config).items():
        print(f"{key:20s}: {value}")
    print("="*50 + "\n")
    
    # Load dataset
    X_train, X_test, y_train, y_test, num_classes = load_dataset(config.dataset)
    
    # Create train/val split
    X_train, X_val, y_train, y_val = main.create_train_val_split(X_train, y_train, val_ratio=0.1)
    
    # Define network architecture
    input_size = X_train.shape[1]
    output_size = num_classes
    layer_sizes = [input_size]
    
    # Add hidden layers
    for _ in range(config.num_layers):
        layer_sizes.append(config.hidden_size)
    
    # Add output layer
    layer_sizes.append(output_size)
    
    # Initialize parameters
    parameters = main.initialize_parameters(layer_sizes, init_method=config.weight_init, weight_decay=config.weight_decay)
    
    # Map loss choice to loss type
    loss_type = "cross_entropy" if config.loss == "cross_entropy" else "mse"
    
    # Map optimizer choice to optimizer function
    optimizer_map = {
        "sgd": main.sgd,
        "momentum": main.momentum,
        "nag": main.nesterov,
        "rmsprop": main.rmsprop,
        "adam": main.adam,
        "nadam": main.nadam
    }
    
    optimizer_func = optimizer_map[config.optimizer]

    run_name = f"hl_{config.num_layers}_bs_{config.batch_size}_opt_{config.optimizer}_ac_{config.activation}_wd_{config.weight_decay}"    
    wandb.init(project=config.wandb_project, entity=config.wandb_entity, config=vars(config) , name=run_name)

    
    # Train the model
    optimizer_func(config, X_val, y_val, X_train, y_train, parameters)
    
    # Evaluate on test set
    test_pred, _ = main.forward(X_test, parameters, activation=config.activation)
    test_loss = main.compute_loss(test_pred, y_test, parameters, config.weight_decay, loss_type=loss_type)
    test_acc = np.mean(np.argmax(test_pred, axis=1) == y_test)
    
    wandb.log({"test_loss": test_loss, "test_accuracy": test_acc})
    wandb.finish()
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

def main_function():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train a neural network with various optimizers and track with Weights & Biases")
    
    # Wandb arguments
    parser.add_argument("-wp", "--wandb_project", type=str, default="myprojectname", help="Project name used to track experiments in Weights & Biases dashboard")
    parser.add_argument("-we", "--wandb_entity", type=str, default="myname", help="Wandb Entity used to track experiments in the Weights & Biases dashboard")
    # Dataset arguments
    parser.add_argument("-d", "--dataset", type=str, default="fashion_mnist", choices=["mnist", "fashion_mnist"], help="Dataset to use for training")
    
    # Training arguments
    parser.add_argument("-e", "--epochs", type=int, default=10, help="Number of epochs to train neural network")
    parser.add_argument("-b", "--batch_size", type=int, default=32, help="Batch size used to train neural network")
    
    # Loss function
    parser.add_argument("-l", "--loss", type=str, default="cross_entropy", choices=["mean_squared_error", "cross_entropy"], help="Loss function to use")
    
    # Optimizer arguments
    parser.add_argument("-o", "--optimizer", type=str, default="adam", choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"], help="Optimizer to use")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.0001, help="Learning rate used to optimize model parameters")
    parser.add_argument("-m", "--momentum", type=float, default=0.5, help="Momentum used by momentum and nag optimizers")
    parser.add_argument("-beta", "--beta", type=float, default=0.5, help="Beta used by rmsprop optimizer")
    parser.add_argument("-beta1", "--beta1", type=float, default=0.5, help="Beta1 used by adam and nadam optimizers")
    parser.add_argument("-beta2", "--beta2", type=float, default=0.5, help="Beta2 used by adam and nadam optimizers")
    parser.add_argument("-eps", "--epsilon", type=float, default=0.000001, help="Epsilon used by optimizers")
    parser.add_argument("-w_d", "--weight_decay", type=float, default=0.0, help="Weight decay used by optimizers")
    
    # Model architecture arguments
    parser.add_argument("-w_i", "--weight_init", type=str, default="xavier", choices=["random", "Xavier"], help="Weight initialization method")
    parser.add_argument("-nhl", "--num_layers", type=int, default=4, help="Number of hidden layers used in feedforward neural network")
    parser.add_argument("-sz", "--hidden_size", type=int, default=128, help="Number of hidden neurons in a feedforward layer")
    parser.add_argument("-a", "--activation", type=str, default="tanh", choices=["identity", "sigmoid", "tanh", "ReLU"], help="Activation function to use")
    
    args = parser.parse_args()
    
    # Convert "Xavier" to "xavier" for consistency with main.py
    if args.weight_init == "Xavier":
        args.weight_init = "xavier"
    
    # Convert "ReLU" to "relu" for consistency with main.py
    if args.activation == "ReLU":
        args.activation = "relu"
    
    # Run training
    train_neural_network(args)

if __name__ == "__main__":
    main_function()