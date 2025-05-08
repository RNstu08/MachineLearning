# Topic 21: Artificial Neural Networks (ANNs)

## 1. Overview

Artificial Neural Networks (ANNs) are a class of machine learning models inspired by the structure and function of biological neural networks in the brain. They form the foundation of **Deep Learning**. ANNs are capable of learning complex patterns from data and are used for a wide variety of tasks, including classification, regression, image recognition, natural language processing, and more.

## 2. Core Concepts

### a. Biological Inspiration & The Artificial Neuron

* **Biological Neuron:** Receives signals via dendrites, processes them in the soma, and if a threshold is met, fires an output signal down an axon to other neurons via synapses.
* **Artificial Neuron (Perceptron/Node):** A mathematical function that models a biological neuron. It takes multiple weighted inputs, adds a bias, and passes the result through an **activation function** to produce an output.

### b. The Perceptron

* The simplest form of a neural network, consisting of a single artificial neuron.
* **Structure:** Inputs, weights, bias, a weighted sum, and a step activation function.
* **Learning:** Uses the Perceptron Learning Algorithm to adjust weights and bias based on prediction errors.
* **Limitation:** Can only learn **linearly separable** patterns (e.g., cannot solve the XOR problem).

### c. Multi-Layer Perceptrons (MLPs)

To overcome the limitations of single perceptrons, neurons are organized into layers:
* **Input Layer:** Receives the raw input features.
* **Hidden Layer(s):** One or more layers between the input and output layers. Neurons in hidden layers apply weighted sums, biases, and **non-linear activation functions** to their inputs. These layers enable the network to learn complex, non-linear relationships and hierarchical features. Networks with multiple hidden layers are called "deep" neural networks.
* **Output Layer:** Produces the final prediction. Its structure (number of neurons, activation function) depends on the task (e.g., sigmoid for binary classification, softmax for multi-class classification, linear for regression).
* **Feedforward Networks:** Information flows in one direction, from input to output, without cycles.

### d. Activation Functions

Activation functions introduce **non-linearity** into the network, allowing it to learn complex patterns. Without them, an MLP would behave like a simple linear model.
* **Sigmoid ($\sigma(z) = 1 / (1 + e^{-z})$):** Output (0, 1). Used in output layer for binary classification. Prone to vanishing gradients in deep hidden layers.
* **Tanh ($\tanh(z)$):** Output (-1, 1). Zero-centered, often preferred over sigmoid in hidden layers, but still has vanishing gradient issues.
* **ReLU (Rectified Linear Unit - $ReLU(z) = \max(0, z)$):** Output $[0, \infty)$. Most popular for hidden layers. Computationally efficient, helps with vanishing gradients (for positive inputs). Can suffer from "dying ReLU" problem.
* **Variants of ReLU (Leaky ReLU, PReLU, ELU):** Address the dying ReLU problem by allowing a small, non-zero gradient for negative inputs.
* **Softmax:** Used in the output layer for **multi-class classification**. Converts raw scores (logits) into a probability distribution over K classes, where outputs sum to 1.
    `Softmax(zᵢ) = e^(zᵢ) / Σ e^(zⱼ)`

## 3. How ANNs Learn: Forward & Backward Propagation

### a. Forward Propagation

* The process of passing input data through the network, layer by layer, to compute the final output (prediction).
* At each neuron: `output = activation_function( (inputs • weights) + bias )`.

### b. Backpropagation (Backward Propagation of Errors)

* The algorithm used to train ANNs by adjusting weights and biases to minimize the error (loss).
1.  **Forward Pass:** Calculate the network's prediction for a training instance.
2.  **Calculate Loss:** Compute the error between the prediction and the true target using a **loss function**.
3.  **Backward Pass:**
    * Propagate the error signal backward through the network, from the output layer to the input layer.
    * Uses the **chain rule** from calculus to compute the gradient (partial derivative) of the loss function with respect to each weight and bias in the network ($\partial L / \partial w$, $\partial L / \partial b$). These gradients indicate how much each parameter contributed to the error.
4.  **Update Parameters:** Use an **optimizer** (e.g., Gradient Descent) to adjust the weights and biases in the direction opposite to their gradients, scaled by a **learning rate ($\eta$)**.
    `w_new = w_old - η * (∂L / ∂w_old)`

This process is repeated for many **epochs** (passes through the entire training dataset), typically using **mini-batches** of data for each update.

## 4. Loss Functions

Quantify the difference between the network's predictions and the true target values. The goal is to minimize this function.
* **For Regression:**
    * **Mean Squared Error (MSE / L2 Loss):** $L = (1/N) Σ (y_i - \hat{y}_i)^2$. Sensitive to outliers.
    * **Mean Absolute Error (MAE / L1 Loss):** $L = (1/N) Σ |y_i - \hat{y}_i|$. Less sensitive to outliers.
    * **Huber Loss:** Combines MSE (for small errors) and MAE (for large errors).
* **For Classification:**
    * **Binary Cross-Entropy (Log Loss):** For binary classification. $L = -[y \log(\hat{p}) + (1-y) \log(1-\hat{p})]$.
    * **Categorical Cross-Entropy:** For multi-class classification. $L = -Σ y_k \log(\hat{p}_k)$.

## 5. Optimizers

Algorithms that use the gradients computed by backpropagation to update the network's weights and biases.
* **Gradient Descent Variants:**
    * **Batch Gradient Descent:** Uses the entire dataset for each update. Slow for large datasets.
    * **Stochastic Gradient Descent (SGD):** Uses one sample at a time. Noisy updates, can escape local minima.
    * **Mini-Batch Gradient Descent:** Most common. Uses a small batch of samples. Balances stability and efficiency.
* **Advanced Optimizers (to improve convergence and speed):**
    * **Momentum:** Adds a fraction of the previous update to the current one, accelerating convergence and dampening oscillations.
    * **Nesterov Accelerated Gradient (NAG):** A "look-ahead" version of momentum.
    * **AdaGrad:** Adapts learning rates per parameter (larger updates for infrequent features). Can suffer from diminishing learning rates.
    * **RMSProp:** Addresses AdaGrad's diminishing learning rate by using an exponentially decaying average of squared gradients.
    * **Adam (Adaptive Moment Estimation):** Combines ideas from Momentum and RMSProp. Often a good default choice.
    * **AdamW:** Adam with improved weight decay (L2 regularization) handling.
* **Learning Rate Schedules:** Gradually decreasing the learning rate during training can help improve convergence.

## 6. Introduction to PyTorch

PyTorch is a popular open-source deep learning framework known for its flexibility and Python-first approach.
* **Key Components:**
    * **Tensors (`torch.Tensor`):** Multi-dimensional arrays, similar to NumPy arrays, with GPU acceleration capabilities.
    * **Automatic Differentiation (`torch.autograd`):** Tracks operations on tensors and automatically computes gradients for backpropagation when `.backward()` is called on a scalar loss.
    * **Neural Network Modules (`torch.nn`):** Provides building blocks for neural networks:
        * `nn.Module`: Base class for all neural network modules.
        * Layers: `nn.Linear`, `nn.Conv2d`, `nn.LSTM`, etc.
        * Activation Functions: `nn.ReLU`, `nn.Sigmoid`, `nn.Softmax`, etc.
        * Loss Functions: `nn.MSELoss`, `nn.CrossEntropyLoss`, etc.
    * **Optimizers (`torch.optim`):** Implements various optimization algorithms (`optim.SGD`, `optim.Adam`, etc.).
    * **Data Handling (`torch.utils.data`):** `Dataset` and `DataLoader` classes for efficient data loading, batching, and shuffling.
* **Dynamic Computation Graphs (Define-by-Run):** Computation graphs are built on the fly as code executes, offering flexibility and easier debugging.

## 7. Building ANNs (Conceptual Workflow in PyTorch)

1.  **Prepare Data:** Load, preprocess, convert to Tensors, use `Dataset` & `DataLoader`.
2.  **Define Model:** Create a class inheriting from `nn.Module`, define layers in `__init__`, and implement the `forward` pass.
3.  **Define Loss Function:** E.g., `criterion = nn.CrossEntropyLoss()`.
4.  **Define Optimizer:** E.g., `optimizer = optim.Adam(model.parameters(), lr=learning_rate)`.
5.  **Training Loop (epochs):**
    * Iterate through `DataLoader` (mini-batches).
    * Move data to device (CPU/GPU).
    * `optimizer.zero_grad()` (clear old gradients).
    * Forward pass: `outputs = model(inputs)`.
    * Calculate loss: `loss = criterion(outputs, labels)`.
    * Backward pass: `loss.backward()` (compute gradients).
    * Update parameters: `optimizer.step()`.
    * (Optional) Validate on a validation set.
6.  **Evaluation:** Test the trained model on a separate test set.

**Regularization Techniques (e.g., `nn.Dropout`):** Can be added to network layers to prevent overfitting by randomly dropping neurons during training.

---