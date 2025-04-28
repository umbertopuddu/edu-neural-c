# Introduction to Neural Networks
 
[![Honorable mention](https://img.youtube.com/vi/PAZTIAfaNr8/0.jpg)](https://www.youtube.com/watch?v=PAZTIAfaNr8)
> Click on the image to play - MUST WATCH !!

A neural network (NN), is your *best guess of an unknown function*. They are inspired by the biological neural networks that constitute your brain and the foundation of **Deep Learning**.

At their core, NNs consist of layers of interconnected nodes, or "neurons". Each connection, like the synapses of a brain, transmits a signal from one neuron to another. The neuron that receives a signal reprocesses it itself, and then signals downstream once more, until the destination is reached. In most NNs, signals travel from an input layer, through one or more so-called hidden layers (i.e., the layers between input and output), to an output layer. Each of this layers can be though as an array of many neurons, which have different connection with the neurons from layers before and after. This assignment treats *fully-connected NNs*, meaning that every neuron of a given layer is connected to all the neurons of the layer right before and right after (i.e., receives from all neurons in the previous one and transmits to all of the ones in the next one). The network learns by adjusting the strengths (weights) of these connections based on input data and desired outputs, enabling it to recognize patterns, make predictions, or classify information.

## Approximating the World with Functions

A key theoretical underpinning of neural networks is their ability to approximate complex functions. With enough neurons and the right structure, a neural network can approximate virtually *any* continuous function mapping inputs to outputs. This is known as the **Universal Approximation Theorem** — basically, with the right setup, your network can learn to model virtually *ANY* reasonable process, given enough data, capacity (layers and neurons per layer) and training time. Just as with the right method, amount of time, and effort you have learn and master any skill you needed (actively or subconsciously).

For a deeper dive into this concept, Michael Nielsen’s online book "Neural Networks and Deep Learning" provides an excellent explanation (see Chapter 4 on Universality):
👉 [Neural Networks and Deep Learning - Universality](http://neuralnetworksanddeeplearning.com/chap4.html)

## How Neural Networks Learn: Gradient Descent

Neural networks learn by minimizing something called an "error" or "loss" function. This function measures how wrong the network's predictions are compared to the real target values. Training the network means adjusting the model’s parameters (weights and biases) so that the predictions get closer and closer to the actual targets, thereby reducing this error.

The most common tool used for minimizing the loss is called **Gradient Descent**.

**The Core Idea:**
Imagine the loss function as a huge landscape of hills and valleys. Where you are on this landscape depends on your current choice of parameters. The lowest valley is where the error is smallest — that's where we want to be. Gradient Descent is basically the art of finding the fastest way down the hill toward that lowest valley. You could think of the Gradient Descent as a way to represent mathematically the *urgency* you get to study before an exam, which depends on how many mistakes or tears you have shed on the practice.

I guess it's time for some math now... but come on, you knew this was coming, right?

**The Formula:**
The update rule for a parameter (or weight) $\theta$ is:

$$
\theta_{\text{new}} = \theta_{\text{old}} - \eta \nabla J(\theta)
$$

Where:
- $\theta$ represents a parameter (a weight or a bias) in the network.
- $\eta$ (called the **learning rate**) controls how big of a step we take (i.e., how much in consideration are you taking that urgency).
- $J(\theta)$ is the **loss function** we want to minimize.
- $\nabla J(\theta)$ is the **gradient** of the loss function with respect to $\theta$ (i.e., the urgency).

---

### Breaking It Down:

- **The Gradient (∇J(θ)):**
  This tells us the direction in which the loss is increasing the fastest. It's like having a local map of the landscape that says, "If you move in this direction, you'll go uphill."

- **The Partial Derivative (dJ/dθ):**
  For a single parameter, the derivative tells us:
  > *If I nudge θ a little bit, how much will J (the error) change?*

- **Why the minus sign?**
  The gradient points *uphill* — toward *increasing* loss. Since we want *less* error, we go the **opposite** way — that's why we subtract the gradient.

---

# Provided tools

## Overview

This repository contains a foundational implementation of a feedforward neural network (Multi-Layer Perceptron) written entirely in standard C (C99). It provides fundamental building blocks for creating, training, and using simple neural networks for tasks like regression and binary classification.

The library uses a **neuron-centric** approach, where computations explicitly iterate through neurons and their connections. This design is intended to be clear and educational but is generally less performant than matrix-based approaches for large-scale problems.

## Repository Contents

- **`nn.h`** — Public header file. Include this in your project. It provides:
  - Type definitions: `pfloat`, `LogLevel`, `Model`, `Params`, `FinalizedModel`, and others.
  - Public function prototypes: `make_model`, `model_train`, `model_finalize`, `finalized_model_predict`, `model_save_params`, `model_load_params`, etc.
  - Constants for standard activation functions (`nnSigmoid`, `nnTanh`, `nnRelu`, `nnIdentity`) and loss functions (`nnMSE`, `nnBCE`).

- **`nn.c`** — Source file implementing all functions declared in `nn.h`.

- **Testers** — Example programs for validating the framework (e.g., absolute value, XOR, etc.).

## The API

### Float Precision

* **`pfloat`**: Defined via `typedef` in `nn.h` (currently `long double`). Controls the precision of all calculations. Rendered as: $pfloat$. Adjust depending on your needs.

> *A smaller datatype will decrease the memory usage, but possibly also make your weights meaningless.*

### Configuration (`Params`)

Before creating a model, you need to configure its *training* properties (or hyperparameters) using the `Params` struct:

```c
typedef struct params {
    const nnLoss *loss;          // Pointer to the loss function (e.g., &nnMSE, &nnBCE, or custom defined one).
    Optimizer *optimizer;        // Pointer to the optimizer settings (only SGD and mini-batch for now).
    int seed;                    // Seed for random weight initialization, crucial for reproducability (<= 0 uses time).
    size_t log_frequency_epochs; // Log average loss every N epochs during model_train (0 disables).
    bool randomize_batches;      // Shuffle dataset indices each epoch in model_train? To improve generalisation and reduce bias introduced by a following a specific sequence.
} Params;
```
* **Note:**
* The `Params` struct itself is usually created on the stack.

**The Optimizer:**

```c
// Creates optimizer (currently only holds Learning Rate for Stochastic Gradient Descent).
Optimizer *make_optimizer(pfloat learning_rate);
```

* **Note:**
* You need to create and manage the `Optimizer` struct (using `make_optimizer` and `free`).
* The `Params` struct itself is usually created on the stack.

### Model Creation (`make_model`)

The trainable `Model` is created using `make_model`:

```c
Model *make_model(size_t n_inputs, size_t n_outputs, const Params *params, ...);
```

* Takes the number of inputs (`n_inputs`) and outputs (`n_outputs`) for the model.
* Takes a pointer to the configured `Params` struct (containing training settings).
* Variable arguments (`...`) define the hidden and output layers sequentially. Meaning that no matter how many layers you design your model to have, as long as you follow the correct syntax, it'll be able to parse it.
* Each layer is defined by a pair: (`size_t num_neurons`, `const nnFunc *activation_func`).
* The list *must* be terminated by a `(size_t)0`.
* The number of neurons in the *last* specified layer must match the `n_outputs` argument.

**Example:**
```c
// Creates a model with:
// - 1 input feature
// - 1 output value
// - 1 hidden layer with 64 neurons using Tanh activation
// - 1 output layer with 1 neuron using Identity activation
// - Uses training parameters defined in 'params'
model = make_model(1, 1, &params,
                   64, &nnTanh,     // Hidden Layer 1 (`nnRelu`usually performs faster)
                   1, &nnIdentity, // Output Layer
                   0);             // Terminator
```

### Training (`model_train`)

The primary training function handles the entire training loop:

```c
bool model_train(Model *model, const pfloat *all_inputs, const pfloat *all_targets,
                 size_t num_samples, size_t epochs, size_t batch_size);
```

* Takes the initialized model, pointers to the entire input and target datasets (flattened), the total number of samples, the number of epochs, and the desired `batch_size`.
* **Internal Loop:** Iterates through epochs. For each epoch:
    * Optionally shuffles data indices if `params->randomize_batches` is true (using an internal buffer `model->shuffled_indices`).
    * Iterates through the dataset in batches of size `batch_size`.
    * For each batch, it internally calls helpers to:
        1.  Zero gradients (`model_zero_grads`).
        2.  Perform forward and backward passes for each sample in the batch, accumulating gradients.
        3.  Apply gradient updates using `model_apply_gradients`, averaging by the actual number of samples processed in the batch (handles partial batches at the end). This step implements the Gradient Descent update explained earlier.
    * Logs the average loss over the *entire* dataset every `params->log_frequency_epochs` using `model_calculate_average_loss`.
* Returns `true` if all epochs complete without error, `false` otherwise.

### Inference (`forward` / `finalized_model_predict`)

You can get predictions in two ways:

## Making Predictions

### 1. Using the Trainable Model
```
const pfloat *forward(Model *m, const pfloat *input_vec);
```

- Processes a single input sample (`input_vec`).
- Returns a pointer to the model’s internal output buffer (valid only until the next `forward` call or `free_model`).
- Suitable for use *during* or *after* training within the same process.

### 2. Using a Finalized Model (for Deployment)
```
pfloat *finalized_model_predict(const FinalizedModel *fm, const pfloat *input_vec, pfloat *output_buffer);
```
- Processes a single input sample (`input_vec`).
- Requires a user-provided `output_buffer` large enough to hold `n_outputs` `pfloat` values.
- Returns a pointer to the user's `output_buffer` on success, or `NULL` on error.
- **Preferred for deployment**, as it uses an optimized, inference-only structure that removes all unnecessary data from memory.


### The Finalized Model (Inference Optimization)

The `FinalizedModel` provides an optimized, inference-only representation of a trained network.

## Finalized Models

- **Creation**
```
FinalizedModel *model_finalize(const Model *m);
```
  - Takes a trained `Model` and creates a new `FinalizedModel`.

- **Content**
  - Stores only the essentials for inference: dimensions, weights, biases, and activation function pointers (`FinalizedLayer`).
  - Discards gradients, training parameters, and any unnecessary training data.

- **Benefits**
  - Smaller memory footprint.
  - Clearer separation of concerns: inference vs. training.
  - Allows freeing the original `Model` memory without losing the trained weights.

- **Usage**
  - Use `finalized_model_predict` for making predictions.

- **Cleanup**
  - Use `free_finalized_model(FinalizedModel *fm)` to deallocate a finalized model.

### Parameter Persistence (`model_save_params`, `model_load_params`)

The provided library allows saving/loading only the model parameters and essential dimensions. It **does not** save activation functions or training configuration.

* `bool model_save_params(const Model *m, const char *filename)`: Saves weights, biases, and layer dimensions from a trainable `Model`.
* `bool model_load_params(Model *m, const char *filename)`: Loads parameters into a pre-existing `Model`. The caller must first create `Model *m` using `make_model` with the **identical architecture** (layer sizes, inputs per neuron, activation functions) as the saved model.
* `finalized_model_save_params` / `finalized_model_load_params`: Similar functions exist for the `FinalizedModel` structure, also requiring a pre-existing, architecturally identical structure for loading.

## Available Components

### Activation Functions:

* `nnSigmoid`: Sigmoid function: $\sigma(x) = \frac{1}{1 + e^{-x}}$
* `nnTanh`: Hyperbolic tangent: $\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$
* `nnRelu`: Rectified Linear Unit: $\text{ReLU}(x) = \max(0, x)$
* `nnIdentity`: Linear / Identity function: $f(x) = x$. Use for regression output layers or linear transformations (i.e. don't need non-linear activation function).

### Implemented Losses:

* `nnMSE`: Mean Squared Error (for regression): $J = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2$
* `nnBCE`: Binary Cross-Entropy (for binary classification, use with `nnSigmoid` output): $J = -\frac{1}{N} \sum_{i=1}^{N} [y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)]$
    (where $y$ is the true label (0 or 1) and $\hat{y}$ is the predicted probability)

## How to Compile and Run

A `Makefile` is provided.

* **Compile & Build:** `make`
* **Run Testers:** `make test`
* **Clean:** `make clean` (removes executable, object files, and saved parameters and weights)

## Limitations

* **Performance:** Neuron-centric design is significantly slower than matrix/vectorized operations for large models/datasets.
* **Features:** Implements only SGD and mini-batch SGD. Lacks therefore advanced optimizers (Adam, RMSprop), different layer types (Convolutional, Recurrent), regularization techniques (Dropout, L1/L2), etc.
* **Parameter Loading:** Requires manual reconstruction of the *exact* model architecture (layer sizes, activation functions) before loading saved parameters. The saved parameters files only store weights, biases, and dimensions, so for running a saved model, you need to \"know\" (#include ...) or redefine the model architecture.
* **Parallelization:** Single-threaded CPU execution only. With mutiple core CPUs and increasingly more available GPUs, sequential computation is a bottleneck of the given implementation, supporting CUDA or Apple MLX would improve the speed of training and inference.

# Your Task

Looking at the provided testers, source files, headers, and the API described in this repo:

👉 **Define your dataset**:
You are free to choose the number of samples, but the data must satisfy:
-100 < Re(z) < 100 and -100 < Im(z) < 100.

👉 **Define a model** using the provided framework and loss functions, aimed at approximating the **complex square root function** (supporting negative numbers as well).

👉 **Train the model**.

To get started, consider these questions:
- How many outputs should your model have?
- What kind of activation function(s) seem to perform best for your testcases?
- When should you stop training?

> *Grading will be based on the accuracy your model achieves on unseen data (generalization/extrapolation) and even outside the training range ("in the wild").
> Any model achieving beyond 95% accuracy will award full credit: the goal of a neural network is to achieve "good enough" and not perfect.*

Good luck — and have fun 😁 !
