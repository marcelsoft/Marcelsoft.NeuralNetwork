# Marcelsoft.NeuralNetwork

A high-performance, multi-layer neural network library for C# built on top of [Math.NET Numerics](https://numerics.mathdotnet.com/). This library provides a flexible framework for building, training, and deploying neural networks with support for multiple high-performance computing backends.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Features

- **Multi-layer Neural Networks**: Easily create networks with multiple fully-connected layers
- **Flexible Training**: Multiple training algorithms including backpropagation and simulated annealing
- **High-Performance Computing**: Automatic support for CUDA, MKL, and OpenBLAS providers
- **Persistence**: Save and load trained networks to/from disk
- **Modern C#**: Built with .NET 10.0 using latest C# language features
- **Math.NET Integration**: Leverages Math.NET Numerics for efficient matrix operations

## Quick Start

### Installation

Clone the repository:
```bash
git clone https://github.com/yourusername/Marcelsoft.NeuralNetwork.git
cd Marcelsoft.NeuralNetwork
```

Build the project:
```bash
dotnet build
```

### Basic Usage

```csharp
using Marcelsoft.NeuralNetwork;
using MathNet.Numerics.LinearAlgebra;

// Create a neural network
var network = new NeuralNetwork();
network.CreateLayer(4, 400);    // Input: 4,  Hidden: 400
network.CreateLayer(400, 30);   // Hidden: 400, Hidden: 30
network.CreateLayer(30, 4);     // Hidden: 30,  Output: 4

// Forward pass
var input = Matrix<double>.Build.Dense(1, 4, new[] { 0.1, 0.2, 0.3, 0.4 });
var output = network.Forward(input);

// Save the network
Storage.Save(network, "models", "my_network");

// Load the network
var loadedNetwork = Storage.Load("models", "my_network");
```

## Project Structure

```
src/
├── Marcelsoft.NeuralNetwork/          # Main library
│   ├── NeuralNetwork.cs               # Neural network container
│   ├── NetworkLayer.cs                # Individual layer implementation
│   ├── Storage.cs                     # Network persistence
│   ├── Utils.cs                       # Utility functions
│   └── Training/                      # Training algorithms
│       ├── Trainer.cs                 # Backpropagation trainer
│       ├── TrainerBase.cs             # Base class for trainers
│       └── SimulatedAnnealingTrainer.cs
└── TestAppNN/                         # Example application
    └── Program.cs                     # Usage examples
```

## Core Components

### NeuralNetwork
The main class for building and using neural networks. Supports stacking multiple layers and provides forward propagation methods.

**Key Methods:**
- `CreateLayer(inputSize, outputSize)`: Add a new fully-connected layer
- `Forward(input)`: Forward pass with derivative computation (for training)
- `FastForward(input)`: Optimized forward pass (for inference)
- `Clone()`: Create an independent copy of the network

### NetworkLayer
Represents a single fully-connected layer with weights, biases, and sigmoid activation.

**Features:**
- Random weight initialization
- sigmoid activation function with precomputed derivatives
- Weight and bias updates for gradient descent training
- Layer cloning for network duplication

### Training
Multiple training strategies for optimizing network weights:
- **Backpropagation**: Classic gradient descent-based training
- **Simulated Annealing**: Metaheuristic optimization for escaping local minima

### Storage
Utilities for persisting neural networks to disk and loading them back into memory.

## High-Performance Computing

This library automatically detects and uses available high-performance computing providers:

- **CUDA**: GPU acceleration via NVIDIA CUDA
- **MKL**: Intel Math Kernel Library for CPU optimization
- **OpenBLAS**: Open-source BLAS library for CPU acceleration

The library gracefully falls back to managed computation if specialized providers are unavailable.

## Example: Training a Network

See [TestAppNN/Program.cs](src/TestAppNN/Program.cs) for a complete example that:
1. Creates a neural network
2. Trains it on sample time-series data
3. Saves the trained network
4. Loads and verifies the network

```bash
cd src
dotnet run --project TestAppNN
```

## Requirements

- **.NET 10.0** or later
- Optional: NVIDIA CUDA toolkit for GPU acceleration
- Optional: Intel MKL or OpenBLAS for CPU optimization

## Dependencies

- [MathNet.Numerics](https://www.nuget.org/packages/MathNet.Numerics) (v5.0.0)
- [MathNet.Numerics.Data.Text](https://www.nuget.org/packages/MathNet.Numerics.Data.Text) (v5.0.0)
- [MathNet.Numerics.Providers.CUDA](https://www.nuget.org/packages/MathNet.Numerics.Providers.CUDA) (v5.0.0)
- [MathNet.Numerics.Providers.MKL](https://www.nuget.org/packages/MathNet.Numerics.Providers.MKL) (v5.0.0)
- [MathNet.Numerics.Providers.OpenBLAS](https://www.nuget.org/packages/MathNet.Numerics.Providers.OpenBLAS) (v5.0.0)

## Building

### Debug Build
```bash
dotnet build
```

### Release Build
```bash
dotnet build -c Release
```

### Run Tests/Examples
```bash
dotnet run --project src/TestAppNN
```

## Usage Examples

### Creating and Training a Network

```csharp
var network = new NeuralNetwork();
network.CreateLayer(4, 100);
network.CreateLayer(100, 50);
network.CreateLayer(50, 4);

// Prepare training data
var inputs = new List<Vector<double>>();
var outputs = new List<Vector<double>>();
// ... populate inputs and outputs ...

// Train using backpropagation
var trainer = new Trainer();
trainer.Train(
    network, 
    inputs, 
    outputs, 
    learningRate: 0.05, 
    epochs: 5000
);
```

### Cloning Networks

```csharp
var originalNetwork = new NeuralNetwork();
// ... configure and train ...

var clonedNetwork = originalNetwork.Clone();
```

## Architecture Notes

- **Activation Function**: Currently uses sigmoid activation exclusively
- **Matrix Operations**: All computations use Math.NET Numerics matrices
- **Input Format**: Accepts row vectors (1 × n matrices) as input
- **Output Format**: Returns row vectors matching the output layer size

## Future Enhancements

- [ ] Support for additional activation functions (ReLU, Tanh, etc.)
- [ ] Convolutional layers
- [ ] Recurrent layers (LSTM, GRU)
- [ ] Batch normalization
- [ ] Dropout for regularization
- [ ] Cross-platform support (Windows, macOS)
- [ ] Performance benchmarking suite

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For issues, questions, or suggestions, please [open an issue](https://github.com/yourusername/Marcelsoft.NeuralNetwork/issues) on GitHub.

