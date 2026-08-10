# Module 08 | Backpropagation Fundamentals <!-- omit in toc -->



## Problem Statement (Rephrased from [Practice Problems](module08%20Practice%20Problems.pdf))

Design and analyze a **fully connected neural network** with:

* **3 input neurons**
* **3 hidden layers**
* **1 output neuron**

You may choose the number of neurons in each hidden layer and specify the activation function used in each layer.

Complete the following tasks:

1. **Define the Network Architecture**

   * Clearly specify the number of neurons in each layer.
   * Specify the activation function used in each hidden and output layer.
   * Define the dimensions of all weight matrices and bias vectors.

2. **Initialize the Parameters**

   * Initialize all weights and biases with small numerical values.
   * Clearly list the initialized values.

3. **Perform the Forward Pass**

   * Choose one training example.
   * Perform the complete forward pass through the network.
   * Calculate the output step by step, showing the intermediate values at every layer.

4. **Define the Loss Function**

   * Define a suitable loss function, such as **Mean Squared Error (MSE)**.
   * Calculate the loss for the chosen training example.

5. **Derive the Gradients**

   * Manually derive all gradients using the **backpropagation algorithm**.
   * Show the mathematical derivation clearly.

6. **Compute Gradients for All Parameters**

   * Compute the gradients for every weight and bias in every layer.
   * Clearly show the resulting gradient matrices and vectors.

7. **Perform the Complete Backward Pass**

   * Show how the error propagates from the output layer back through all three hidden layers.
   * Explain the role of the chain rule at each step.

8. **Perform One Parameter Update**

   * Choose a suitable learning rate.
   * Perform one complete **gradient descent** update.
   * Calculate and show the updated weights and biases for every layer.

The final solution should contain all intermediate calculations so that the entire **forward pass → loss calculation → backward pass → parameter update** can be followed step by step.

## My Solution

### TASK 01. Define the Network Architecture

#### Input Layer
**Number of Neurons**: $3$
So, any input sample consists of 3 feature values.

**An Input Sample**: $$ x \equiv \begin{bmatrix}
   x_1 \\ x_2 \\ x_3
\end{bmatrix}$$, a $3 \times 1$ matrix, that is, a $3 \times 1$ tensor.

$x$ can be thought as **the output** of the input layer, that is, in general notation, $A^{(0)} = x$.

#### Hidden Layer 1

**Number of Neurons**: $2$, with each neuron having $3$ weights and a bias.

**Vectorization of Weights**:
$$
W^{(1)} \equiv \begin{bmatrix}
   w_{11}^{(1)} &w_{12}^{(1)} &w_{13}^{(1)} \\
   w_{21}^{(1)} &w_{22}^{(1)} &w_{23}^{(1)} \\
\end{bmatrix}
$$, a $2 \times 3$ matrix/tensor. Each row represents a neuron in this layer. And $w_{ij}^{(l)}$ corresponds to the connection between $i$-th neuron of this layer $l$, which is $1$ in this case, and $j$-th neuron of the previous layer.

**Vectorization of Biases**:
$$
B^{(1)} \equiv \begin{bmatrix}
   b_1^{(1)} \\ b_2^{(1)}
\end{bmatrix}
$$, a $2 \times 1$ matrix/tensor. $b_i$ is the bias parameter of $i$-th neuron of this layer.

**Activation Function**:
$$
\text{ReLU}(z) = \begin{cases}
   z, \text{if } z \geq 0 \\
   0, \text{if } z < 0
\end{cases}
$$

**Linear Weighted Sum**:
$$
\begin{aligned}
   Z^{(1)} &\equiv W^{(1)} A^{(0)} +  B^{(1)} \\
   &\equiv W^{(1)} x +  B^{(1)}
\end{aligned}
$$

**Output**:
$$
A^{(1)} \equiv \text{ReLU}({Z^{(1)}})
$$

#### Hidden Layer 2

**Number of Neurons**: $3$, with each neuron having $2$ weights and a bias.

**Vectorization of Weights**:
$$
W^{(2)} \equiv \begin{bmatrix}
   w_{11}^{(2)} &w_{12}^{(2)} \\
   w_{21}^{(2)} &w_{22}^{(2)} \\
   w_{31}^{(2)} &w_{32}^{(2)} \\
\end{bmatrix}
$$, a $3 \times 2$ matrix/tensor. Each row represents a neuron in this layer. And $w_{ij}$ corresponds to the connection between $i$-th neuron of this layer and $j$-th neuron of the previous layer.

**Vectorization of Biases**:
$$
B^{(1)} \equiv \begin{bmatrix}
   b_1^{(2)} \\ b_2^{(2)} \\ b_3^{(2)}
\end{bmatrix}
$$, a $3 \times 1$ matrix/tensor. $b_i$ is the bias parameter of $i$-th neuron of this layer.

**Activation Function**:

$$
\text{ReLU}(z) = \begin{cases}
   z, \text{if } z \geq 0 \\
   0, \text{if } z < 0
\end{cases}
$$

**Linear Weighted Sum**:
$$
Z^{(2)} \equiv W^{(2)} A^{(1)} +  B^{(2)}
$$

**Output**:
$$
A^{(2)} \equiv \text{ReLU}({Z^{(2)}})
$$

#### Hidden Layer 3

**Number of Neurons**: $2$, with each neuron having $3$ weights and a bias.

**Vectorization of Weights**:
$$
W^{(3)} \equiv \begin{bmatrix}
   w_{11}^{(3)} &w_{12}^{(3)} &w_{13}^{(3)} \\
   w_{21}^{(3)} &w_{22}^{(3)} &w_{23}^{(3)} \\
\end{bmatrix}
$$, a $2 \times 3$ matrix/tensor. Each row represents a neuron in this layer. And $w_{ij}$ corresponds to the connection between $i$-th neuron of this layer and $j$-th neuron of the previous layer.

**Vectorization of Biases**:
$$
B^{(3)} \equiv \begin{bmatrix}
   b_1^{(3)} \\ b_2^{(3)}
\end{bmatrix}
$$, a $3 \times 1$ matrix/tensor. $b_i$ is the bias parameter of $i$-th neuron of this layer.

**Activation Function**:

$$
\text{ReLU}(z) = \begin{cases}
   z, \text{if } z \geq 0 \\
   0, \text{if } z < 0
\end{cases}
$$

**Linear Weighted Sum**:
$$
Z^{(3)} \equiv W^{(3)} A^{(2)} +  B^{(3)} 
$$

**Output**:
$$
A^{(3)} \equiv \text{ReLU}({Z^{(3)}})
$$

#### Output Layer
**Number of Neurons**: $1$, with $2$ weights and a bias.

**Vectorization of Weights**:
$$
W^{(4)} \equiv \begin{bmatrix}
   w_{11}^{(4)} &w_{12}^{(4)}
\end{bmatrix}
$$, a $1 \times 2$ matrix/tensor. Each row represents a neuron in this layer. And $w_{ij}$ corresponds to the connection between $i$-th neuron of this layer and $j$-th neuron of the previous layer.

**Vectorization of Biases**:
$$
B^{(3)} \equiv \begin{bmatrix}
   b_1^{(4)}
\end{bmatrix}
$$, a $3 \times 1$ matrix/tensor. $b_i$ is the bias parameter of $i$-th neuron of this layer.

**Activation Function**:
$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

**Linear Weighted Sum**:
$$
Z^{(4)} \equiv W^{(4)} A^{(3)} +  B^{(4)} 
$$

**Output**:
$$
\hat{Y} \equiv A^{(4)} \equiv \sigma({Z^{(4)}})
$$

### TASK 02. Initialize the Parameters

#### Hidden Layer 1
$$
\begin{aligned}
   W^{(1)} &= \begin{bmatrix}
      0.1 &0.1 &0.1 \\
      0.1 &0.1 &0.1 \\
   \end{bmatrix}
   \\ \\
   B^{(1)} &= \begin{bmatrix}
      0.1 \\ 0.1
   \end{bmatrix}
\end{aligned}
$$

#### Hidden Layer 2
$$
\begin{aligned}
   W^{(2)} &= \begin{bmatrix}
      0.1 &0.1 \\
      0.1 &0.1 \\
      0.1 &0.1
   \end{bmatrix}
   \\ \\
   B^{(2)} &= \begin{bmatrix}
      0.1 \\ 0.1 \\ 0.1
   \end{bmatrix}
\end{aligned}
$$
#### Hidden Layer 3
$$
\begin{aligned}
   W^{(3)} &= \begin{bmatrix}
      0.1 &0.1 &0.1 \\
      0.1 &0.1 &0.1 \\
   \end{bmatrix}
   \\ \\
   B^{(3)} &= \begin{bmatrix}
      0.1 \\ 0.1
   \end{bmatrix}
\end{aligned}
$$

#### Output Layer
$$
\begin{aligned}
   W^{(4)} &= \begin{bmatrix}
      0.1 &0.1
   \end{bmatrix}
   \\ \\
   B^{(4)} &= \begin{bmatrix}
      0.1
   \end{bmatrix}
\end{aligned}
$$

### TASK 03. Perform the Forward Pass

#### Choosing a Training Example
$$
x = \begin{bmatrix}
   1 \\ 1 \\ 1
\end{bmatrix}
$$

#### Hidden Layer 1
$$
\begin{aligned}
   Z^{(1)} &\equiv W^{(1)} x + B^{(1)} \\
   &= \begin{bmatrix}
               0.1 &0.1 &0.1 \\
               0.1 &0.1 &0.1
            \end{bmatrix} 
            \begin{bmatrix}
               1 \\ 1 \\ 1
            \end{bmatrix}
            +
            \begin{bmatrix}
               0.1 \\ 0.1
            \end{bmatrix} \\
   &= \begin{bmatrix}
               0.4 \\ 0.4
            \end{bmatrix} \\ \\
   
   A^{(1)} &\equiv \text{ReLU}(Z^{(1)}) \\
   &= \text{ReLU}\Big(\begin{bmatrix}
               0.4 \\ 0.4
            \end{bmatrix}\Big) \\
   &= \begin{bmatrix}
               0.4 \\ 0.4
            \end{bmatrix}
\end{aligned}
$$


#### Hidden Layer 2
$$
\begin{aligned}
   Z^{(2)} &\equiv W^{(2)} A^{(1)} + B^{(2)} \\
   &= \begin{bmatrix}
               0.1 &0.1 \\
               0.1 &0.1 \\
               0.1 &0.1
            \end{bmatrix} 
            \begin{bmatrix}
               0.4 \\ 0.4
            \end{bmatrix}
            +
            \begin{bmatrix}
               0.1 \\ 0.1 \\ 0.1
            \end{bmatrix} \\
   &= \begin{bmatrix}
               0.18 \\ 0.18 \\ 0.18
            \end{bmatrix} \\ \\
   
   A^{(2)} &\equiv \text{ReLU}(Z^{(2)}) \\
   &= \text{ReLU}\Big(\begin{bmatrix}
               0.18 \\ 0.18 \\ 0.18
            \end{bmatrix}\Big) \\
   
   &= \begin{bmatrix}
              0.18 \\ 0.18 \\ 0.18
            \end{bmatrix}
\end{aligned}
$$

#### Hidden Layer 3

$$
\begin{aligned}
   Z^{(3)} &\equiv W^{(3)} A^{(2)} + B^{(3)} \\
   &= \begin{bmatrix}
               0.1 &0.1 &0.1 \\
               0.1 &0.1 &0.1 \\
            \end{bmatrix} 
            \begin{bmatrix}
               0.18 \\ 0.18 \\ 0.18
            \end{bmatrix}
            +
            \begin{bmatrix}
               0.1 \\ 0.1
            \end{bmatrix} \\
   &= \begin{bmatrix}
               0.154 \\ 0.154
            \end{bmatrix} \\ \\
   
   A^{(3)} &\equiv \text{ReLU}(Z^{(3)}) \\
   &= \text{ReLU}\Big(\begin{bmatrix}
               0.154 \\ 0.154
            \end{bmatrix}\Big) \\
   
   &= \begin{bmatrix}
               0.154 \\ 0.154
            \end{bmatrix}
\end{aligned}
$$

#### Output Layer

$$
\begin{aligned}
   Z^{(4)} &\equiv W^{(4)} A^{(3)} + B^{(4)} \\
   &= \begin{bmatrix}
               0.1 &0.1 \\
            \end{bmatrix} 
            \begin{bmatrix}
               0.154 \\ 0.154
            \end{bmatrix}
            +
            \begin{bmatrix}
               0.1
            \end{bmatrix} \\
   &= \begin{bmatrix}
               0.1308
            \end{bmatrix} \\ \\
   
   \hat{Y} \equiv A^{(4)} &\equiv \sigma (Z^{(4)}) \\
   &= \sigma \Big(\begin{bmatrix}
               0.1308
            \end{bmatrix}\Big) \\
   
   &\approx \begin{bmatrix}
               0.533
            \end{bmatrix}
\end{aligned}
$$

### TASK 04. Define the Loss Function and Calculate the Loss

I have already used the sigmoid function in the output layer, so, yes, I am assuming this is a Binary Classification Problem.

I will use **Binary Cross-Entropy (BCE)** as my Loss Function.

$$
L \equiv \text{BCE} \equiv - \big[ y \log{\hat{y}} + (1-y) \log{(1 - \hat{y})} \big]
$$

#### Calculating the Loss

For my input sample $x$, I will assume my true label is $y = 1$. And from **Forward Pass**, $\hat{y}=0.533$.

$$
L = - 1 \times \log{0.533} = - \log{0.533} \approx 0.273
$$


### TASK 05. Derive the Gradients

#### Output Layer

$$
\frac{\partial L}{\partial z^{(4)}} = \hat{Y} - Y = -0.467
$$


$$
\frac{\partial Z^{(4)}}{\partial W^{(4)}} = \begin{bmatrix}
   \frac{\partial z^{(4)}}{\partial w_{ij}^{(4)}}
\end{bmatrix}_{1 \times 2} = \begin{bmatrix}
   a_j^{(3)}
\end{bmatrix}_{1 \times 2} = \begin{bmatrix}
   0.154 &0.154
\end{bmatrix}
$$

$$
\frac{\partial Z^{(4)}}{\partial B^{(4)}} = \begin{bmatrix}
   \frac{\partial z^{(4)}}{\partial b_{i}^{(4)}}
\end{bmatrix}_{1 \times 1} = \begin{bmatrix}
   1
\end{bmatrix}_{1 \times 1} = \begin{bmatrix}
   1
\end{bmatrix}
$$

Hence,
$$
\begin{aligned}
      \frac{\partial L}{\partial W^{(4)}} &= \begin{bmatrix}
      \frac{\partial z^{(4)}}{\partial w_{ij}^{(4)}} \cdot \frac{\partial L}{\partial z^{(4)}} 
   \end{bmatrix}_{1 \times 2} = \frac{\partial L}{\partial z^{(4)}} \cdot \begin{bmatrix}
      \frac{\partial z^{(4)}}{\partial w_{ij}^{(4)}}
   \end{bmatrix}_{1 \times 2}\\ &= -0.467 \cdot \begin{bmatrix}
      0.154 &0.154
   \end{bmatrix}\\ &\approx \begin{bmatrix}
      -0.072 &-0.072
   \end{bmatrix} \\ \\
   \frac{\partial L}{\partial B^{(4)}} &= \begin{bmatrix}
      \frac{\partial z^{(4)}}{\partial b_i^{(4)}} \cdot \frac{\partial L}{\partial z^{(4)}}
   \end{bmatrix}_{1 \times 1} = \frac{\partial L}{\partial z^{(4)}} \cdot \begin{bmatrix}
      \frac{\partial z^{(4)}}{\partial b_i^{(4)}}
   \end{bmatrix}_{1 \times 1} \\
   &=-0.467 \cdot \begin{bmatrix}
   1
\end{bmatrix} \\ &= \begin{bmatrix}
   -0.467
\end{bmatrix}
\end{aligned}
$$

#### Hidden Layer 3

#### Hidden Layer 2

#### Hidden Layer 1



