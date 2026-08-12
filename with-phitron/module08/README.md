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

5. **Perform the Complete Backward Pass and Compute All Gradients**

- Derive the backpropagation equations using the chain rule.
- Propagate the error from the output layer backward through all three hidden layers.
- Compute the gradient for every weight and bias.
- Show the resulting gradient matrices and vectors.
- Clearly explain the mathematical reasoning at each layer.

6. **Perform One Parameter Update**

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
      0.1 &0.2 &0.3 \\
      0.1 &0.2 &0.3 \\
   \end{bmatrix}
   \\ \\
   B^{(1)} &= \begin{bmatrix}
      0.1 \\ 0.2
   \end{bmatrix}
\end{aligned}
$$

#### Hidden Layer 2
$$
\begin{aligned}
   W^{(2)} &= \begin{bmatrix}
      0.1 &0.2 \\
      0.1 &0.2 \\
      0.1 &0.2
   \end{bmatrix}
   \\ \\
   B^{(2)} &= \begin{bmatrix}
      0.1 \\ 0.2 \\ 0.3
   \end{bmatrix}
\end{aligned}
$$
#### Hidden Layer 3
$$
\begin{aligned}
   W^{(3)} &= \begin{bmatrix}
      0.1 &0.2 &0.3 \\
      0.1 &0.2 &0.3 \\
   \end{bmatrix}
   \\ \\
   B^{(3)} &= \begin{bmatrix}
      0.1 \\ 0.2
   \end{bmatrix}
\end{aligned}
$$

#### Output Layer
$$
\begin{aligned}
   W^{(4)} &= \begin{bmatrix}
      0.1 &0.2
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
               0.1 &0.2 &0.3 \\
               0.1 &0.2 &0.3
            \end{bmatrix} 
            \begin{bmatrix}
               1 \\ 1 \\ 1
            \end{bmatrix}
            +
            \begin{bmatrix}
               0.1 \\ 0.2
            \end{bmatrix} \\
   &= \begin{bmatrix}
               0.7 \\ 0.8
            \end{bmatrix} \\ \\
   
   A^{(1)} &\equiv \text{ReLU}(Z^{(1)}) \\
   &= \text{ReLU}\Big(\begin{bmatrix}
               0.7 \\ 0.8
            \end{bmatrix}\Big) \\
   &= \begin{bmatrix}
               0.7 \\ 0.8
            \end{bmatrix}
\end{aligned}
$$


#### Hidden Layer 2
$$
\begin{aligned}
   Z^{(2)} &\equiv W^{(2)} A^{(1)} + B^{(2)} \\
   &= \begin{bmatrix}
               0.1 &0.2 \\
               0.1 &0.2 \\
               0.1 &0.2
            \end{bmatrix} 
            \begin{bmatrix}
               0.7 \\ 0.8
            \end{bmatrix}
            +
            \begin{bmatrix}
               0.1 \\ 0.2 \\ 0.3
            \end{bmatrix} \\
   &= \begin{bmatrix}
               0.33 \\ 0.43 \\ 0.53
            \end{bmatrix} \\ \\
   
   A^{(2)} &\equiv \text{ReLU}(Z^{(2)}) \\
   &= \text{ReLU}\Big(\begin{bmatrix}
               0.33 \\ 0.43 \\ 0.53
            \end{bmatrix}\Big) \\
   
   &= \begin{bmatrix}
              0.33 \\ 0.43 \\ 0.53
            \end{bmatrix}
\end{aligned}
$$

#### Hidden Layer 3

$$
\begin{aligned}
   Z^{(3)} &\equiv W^{(3)} A^{(2)} + B^{(3)} \\
   &= \begin{bmatrix}
               0.1 &0.2 &0.3 \\
               0.1 &0.2 &0.3 \\
            \end{bmatrix} 
            \begin{bmatrix}
               0.33 \\ 0.43 \\ 0.53
            \end{bmatrix}
            +
            \begin{bmatrix}
               0.1 \\ 0.2
            \end{bmatrix} \\
   &= \begin{bmatrix}
               0.378 \\ 0.478
            \end{bmatrix} \\ \\
   
   A^{(3)} &\equiv \text{ReLU}(Z^{(3)}) \\
   &= \text{ReLU}\Big(\begin{bmatrix}
               0.378 \\ 0.478
            \end{bmatrix}\Big) \\
   
   &= \begin{bmatrix}
               0.378 \\ 0.478
            \end{bmatrix}
\end{aligned}
$$

#### Output Layer

$$
\begin{aligned}
   Z^{(4)} &\equiv W^{(4)} A^{(3)} + B^{(4)} \\
   &= \begin{bmatrix}
               0.1 &0.2 \\
            \end{bmatrix} 
            \begin{bmatrix}
               0.378 \\ 0.478
            \end{bmatrix}
            +
            \begin{bmatrix}
               0.1
            \end{bmatrix} \\
   &= \begin{bmatrix}
               0.2334
            \end{bmatrix} \\ \\
   
   \hat{Y} \equiv A^{(4)} &\equiv \sigma (Z^{(4)}) \\
   &= \sigma \Big(\begin{bmatrix}
               0.2334
            \end{bmatrix}\Big) \\
   
   &\approx \begin{bmatrix}
               0.558
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

For my input sample $x$, I will assume my true label is $y = 1$. And from **Forward Pass**, $\hat{y}=0.558$.

$$
L = - 1 \times \log{0.558} = - \log{0.558} \approx 0.253
$$


### TASK 05. Perform the Complete Backward Pass and Compute All Gradients

#### Proud Documentation of My Enlightening Exploration
I belive my mentor wanted me to calculate the gradients of all scalers $w$ and $b$, individually. He aimed for us to understand **what is truly happening** *under the 'magical'* **Autodifferentiation** provided by frameworks.

But, as a Linear Algebra fan, I couldn't resist **Vectorization**, which is a must for efficient implementation of backpropagation.

In my last commit, I started calculating gradients in vectorized approach, keeping the **'scaler' reality** parallelly in my head. Soon, I noticed 'head' is not enough. So, I started scribbling on paper, of which I am proud. I have decided to keep it as a part of this repo. See [pen-paper.pdf](assets/pen-paper.pdf). I truly learned a lot though this.

I also consulted with Gemini. See my chats with Gemini: [Matrix Multiplication Variants Explained](assets/Chat%20with%20Gemini,%20Matrix%20Multiplication%20Variants%20Explained.pdf) and [Vectorized Backpropagation](assets/Chat%20with%20Gemini,%20Vectorized%20Backpropagation%20for%20Neural%20Network.pdf).

Now, I know what is truly happening as well as I have clarity on vectorized formulas. I am happy. I believe my mentor would be too. I am not lazy. I am differently industrious.

#### Summary of My Exploration
The following summary is tied to my definition of architecture in [TASK 01. Define the Network Architecture](#task-01-define-the-network-architecture).

Let's calculate the derivative of loss $L$ with respect to a (scaler) weight $w^{(l)}_{ij}$, corresponding to the connection between $i$-th neuron of layer $l$ and $j$-th neuron of the previous layer.

$$
\begin{aligned}
   \frac{\partial L}{\partial w^{(l)}_{ij}}
   &= \frac{\partial L}{\partial a^{(l)}_{i}} \cdot
   \color{green} \frac{\partial a^{(l)}_{i}}{\partial z^{(l)}_{i}} \cdot
   \color{red} \frac{\partial z^{(l)}_{i}}{\partial w^{(l)}_{ij}}
   \\
   &= \left(
         \sum_{k=1}^{n^{(l+1)}}
         \left(
            \color{orange}
            \frac{\partial L}{\partial z^{(l+1)}_{k}} \color{black} \cdot
            \color{blue}
            \frac {\partial z^{(l+1)}_{k}}{\partial a^{(l)}_{i}}
            \color{black}
         \right)
      \right) \cdot
   \color{green} \frac{\partial a^{(l)}_{i}}{\partial z^{(l)}_{i}} \color{black} \cdot
   \color{red} \frac{\partial z^{(l)}_{i}}{\partial w^{(l)}_{ij}}

\end{aligned}
$$

As expressed by 'Back' in '**Back**propagation', we have to combine derivatives from left.

Yes, this is the chain rule for function composition from calculus.

The Sigma ($\sum$) on the 2nd line is because the neuron is connected to each of the neurons in the next layer, which can have more than one neurons.

Let's vectorize them now.

$$
\begin{aligned}
   \color{orange}
   \frac{\partial L}{\partial Z^{(l+1)}}
   &= \frac{\partial L}
   {\partial 
   \begin{bmatrix}
      z_1^{(l+1)} \\ z_2^{(l+1)} \\ \vdots \\ z_{n^{(l+1)}}^{(l+1)}
   \end{bmatrix}_{n^{(l+1)} \times 1}
   }
   \\ \\
   &= \begin{bmatrix}
      \frac{\partial L}{\partial z_1} \\
      \frac{\partial L}{\partial z_2} \\
      \vdots \\
      \frac{\partial L}{\partial z_{n^{(l+1)}}}
   \end{bmatrix}_{n^{(l+1)} \times 1}
\end{aligned}
$$

$$
\begin{aligned}
   \color{blue}
   \frac{\partial Z^{(l+1)}}{\partial A^{(l)}}
   &= \frac{\partial 
   \begin{bmatrix}
      z_1^{(l+1)} \\ z_2^{(l+1)} \\ \vdots \\ z_{n^{(l+1)}}^{(l+1)}
   \end{bmatrix}_{n^{(l+1)} \times 1}
   }
   {\partial 
   \begin{bmatrix}
      a_1^{(l)} \\ a_2^{(l)} \\ \vdots \\ a_{n^{(l)}}^{(l)}
   \end{bmatrix}_{n^{(l)} \times 1}
   }
   \\ \\
   &= \begin{bmatrix}
      \frac{\partial z_1^{(l+1)}}{\partial a_1^{(l)}}
      &\frac{\partial z_1^{(l+1)}}{\partial a_2^{(l)}}
      &\cdots
      &\frac{\partial z_1^{(l+1)}}{\partial a_{n^{(l)}}^{(l)}} \\
      \frac{\partial z_2^{(l+1)}}{\partial a_1^{(l)}}
      &\frac{\partial z_2^{(l+1)}}{\partial a_2^{(l)}}
      &\cdots
      &\frac{\partial z_2^{(l+1)}}{\partial a_{n^{(l)}}^{(l)}} \\
      \vdots &\vdots &\ddots &\vdots \\
      \frac{\partial z_{n^{(l+1)}}^{(l+1)}}{\partial a_1^{(l)}}
      &\frac{\partial z_{n^{(l+1)}}^{(l+1)}}{\partial a_2^{(l)}}
      &\cdots
      &\frac{\partial z_{n^{(l+1)}}^{(l+1)}}{\partial a_{n^{(l)}}^{(l)}}
   \end{bmatrix}_{n^{(l+1)} \times n^{(l)}}
\\ \\
&= \begin{bmatrix}
   w^{(l+1)}_{11} & w^{(l+1)}_{12} &\cdots &w^{(l+1)}_{1, n^{(l)}} \\
   w^{(l+1)}_{21} & w^{(l+1)}_{22} &\cdots &w^{(l+1)}_{2, n^{(l)}} \\
   \vdots &\vdots &\ddots &\vdots \\
   w^{(l+1)}_{n^{(l+1)},1} & w^{(l+1)}_{n^{(l+1)},2} &\cdots &w^{(l+1)}_{n^{(l+1)}, n^{(l)}} \\
\end{bmatrix}_{n^{(l+1)} \times n^{(l)}}
\\ \\
&= W^{(l+1)}_{n^{(l+1)} \times n^{(l)}}
\end{aligned}
$$

How do we combine $\color{orange} \frac{\partial L}{\partial Z^{(l+1)}}$ and $\color{blue} \frac{\partial Z^{(l+1)}}{\partial A^{(l)}}$? **We have to combine them with matrix operations in a way that is consistent with the underlying scaler operations**. Such a way is:
$$
 \left( \color{blue} \frac{\partial Z^{(l+1)}}{\partial A^{(l)}} \color{black} \right)^T
 \cdot
\color{orange} \frac{\partial L}{\partial Z^{(l+1)}}
$$
, which gives a matrix of order $n^{(l)} \times 1$. To understand this, keep the orders of the factor matrices in mind, and check if what happens when you multiply is consistent with underlying scaler operations.

$$
\begin{aligned}
   \color{green} \frac{\partial A^{(l)}}{\partial Z^{(l)}}
   &= \frac
   {\partial
      \begin{bmatrix}
         a_1^{(l)} \\ a_2^{(l)} \\ \vdots \\ a_{n^{(l)}}^{(l)}
      \end{bmatrix}_{n^{(l)} \times 1}
   }
   {\partial
      \begin{bmatrix}
         z_1^{(l)} \\ z_2^{(l)} \\ \vdots \\ z_{n^{(l)}}^{(l)}
      \end{bmatrix}_{n^{(l)} \times 1}
   }
   \\ \\
   &= \begin{bmatrix}
      \frac{\partial a_1^{(l)}}{\partial z_1^{(l)}}
      &\frac{\partial a_1^{(l)}}{\partial z_2^{(l)}}
      &\cdots
      &\frac{\partial a_1^{(l)}}{\partial z_{n^{(l)}}^{(l)}} \\
      \frac{\partial a_2^{(l)}}{\partial z_1^{(l)}}
      &\frac{\partial a_2^{(l)}}{\partial z_2^{(l)}}
      &\cdots
      &\frac{\partial a_2^{(l)}}{\partial z_{n^{(l)}}^{(l)}} \\
      \vdots &\vdots &\ddots &\vdots \\
      \frac{\partial a_{n^{(l)}}^{(l)}}{\partial z_1^{(l)}}
      &\frac{\partial a_{n^{(l)}}^{(l)}}{\partial z_2^{(l)}}
      &\cdots
      &\frac{\partial a_{n^{(l)}}^{(l)}}{\partial z_{n^{(l)}}^{(l)}} \\
   \end{bmatrix}_{n^{(l)} \times n^{(l)}}
   \\ \\
   &= \begin{bmatrix}
      \frac{\partial a_1^{(l)}}{\partial z_1^{(l)}}
      &0
      &\cdots
      &0 \\
      0
      &\frac{\partial a_2^{(l)}}{\partial z_2^{(l)}}
      &\cdots
      &0 \\
      \vdots &\vdots &\ddots &\vdots \\
      0
      &0
      &\cdots
      &\frac{\partial a_{n^{(l)}}^{(l)}}{\partial z_{n^{(l)}}^{(l)}} \\
   \end{bmatrix}_{n^{(l)} \times n^{(l)}}
\end{aligned}
$$
, which is a diagonal matrix. We can combine it with incoming gradient this way:

$$
 \left( \color{blue} \frac{\partial Z^{(l+1)}}{\partial A^{(l)}} \color{black} \right)^T
 \cdot
\color{orange} \frac{\partial L}{\partial Z^{(l+1)}}
\color{black} \odot 
\text{diag} \left( \color{green} \frac{\partial A^{(l)}}{\partial Z^{(l)}} \color{black} \right)
$$

Yes, $\odot$ denotes point-wise multiplication of matrices, that is, **Hadamard Multiplication**. And, **diag(A)** is the diagonal vector of a diagonal matrix **A**. This combination give a matrix of order $n^{(l)} \times 1$.


$$
\begin{aligned}
   \color{red} \frac{\partial Z^{(l)}}{\partial W^{(l)}}
   &= \frac
   {\partial
      \begin{bmatrix}
        z^{{(l)}}_1 \\ z^{{(l)}}_2 \\ \vdots \\ z^{{(l)}}_{n^{(l)}}
      \end{bmatrix}_{n^{(l)} \times 1}
   }
   {\partial
      \begin{bmatrix}
         w^{(l)}_{11} &w^{(l)}_{12} &\cdots &w^{(l)}_{1, n^{(l-1)}} \\
         w^{(l)}_{21} &w^{(l)}_{22} &\cdots &w^{(l)}_{2, n^{(l-1)}} \\
         \cdots &\cdots &\ddots &\vdots \\
         w^{(l)}_{n^{(l)}, 1} &w^{(l)}_{n^{(l)}, 2} &\cdots &w^{(l)}_{n^{(l)}, n^{(l-1)}}
      \end{bmatrix}_{n^{(l)} \times n^{(l-1)}}
   }
   \\ \\
   &= \begin{bmatrix}
      \frac{\partial z^{{(l)}}_1}{\partial w^{(l)}_{11}}
      &\frac{\partial z^{{(l)}}_1}{\partial w^{(l)}_{12}}
      &\cdots
      &\frac{\partial z^{{(l)}}_1}{\partial w^{(l)}_{1, n^{(l-1)}}} \\
      \frac{\partial z^{{(l)}}_2}{\partial w^{(l)}_{11}}
      &\frac{\partial z^{{(l)}}_2}{\partial w^{(l)}_{12}}
      &\cdots
      &\frac{\partial z^{{(l)}}_2}{\partial w^{(l)}_{1, n^{(l-1)}}} \\
      \vdots &\vdots &\ddots &\vdots \\
      \frac{\partial z^{{(l)}}_{n^{(l)}}}{\partial w^{(l)}_{11}}
      &\frac{\partial z^{{(l)}}_{n^{(l)}}}{\partial w^{(l)}_{12}}
      &\cdots
      &\frac{\partial z^{{(l)}}_{n^{(l)}}}{\partial w^{(l)}_{1, n^{(l-1)}}}
   \end{bmatrix}_{n^{(l)} \times n^{(l-1)}}
   \\ \\
   &= \begin{bmatrix}
      a^{(l)}_{1} & a^{(l)}_{2} &\cdots &a^{(l)}_{n^{(l-1)}} \\
      a^{(l)}_{1} & a^{(l)}_{2} &\cdots &a^{(l)}_{n^{(l-1)}} \\
      \vdots &\vdots &\ddots &\vdots \\
      a^{(l)}_{1} & a^{(l)}_{2} &\cdots &a^{(l)}_{n^{(l-1)}}
   \end{bmatrix}_{n^{(l)} \times n^{(l-1)}}
   \\ \\
   &= \begin{bmatrix}
      a^{(l)}_{1} & a^{(l)}_{2} &\cdots &a^{(l)}_{n^{(l-1)}}
   \end{bmatrix}_{1 \times n^{(l-1)}} \\
   &\text{(Downcasted with no loss in information)}
   \\ \\
   &= \left( A^{(l-1)} \right)^T
\end{aligned}
$$

Yes, we have not consider the full Jacobian Matrix here.

Now we have,
$$
\begin{aligned}
   \frac{\partial L}{\partial W^{(l)}}
   &= \left( \color{blue} \frac{\partial Z^{(l+1)}}{\partial A^{(l)}} \color{black} \right)^T \cdot
   \color{orange} \frac{\partial L}{\partial Z^{(l+1)}}
   \color{black} \odot 
   \text{diag} \left( \color{green} \frac{\partial A^{(l)}}{\partial Z^{(l)}}
   \color{black} \right) \cdot
   \color{red} \frac{\partial Z^{(l)}}{\partial W^{(l)}} \\
   &= \left( \color{blue} W^{(l+1)} \color{black} \right)^T \cdot
   \color{orange} \frac{\partial L}{\partial Z^{(l+1)}}
   \color{black} \odot 
   \text{diag} \left( \color{green} \frac{\partial A^{(l)}}{\partial Z^{(l)}}
   \color{black} \right) \cdot
   \color{red} \left( A^{(l-1)} \right)^T \\
   &= \color{magenta} \frac{\partial L}{\partial Z^{(l)}} \color{black} \cdot \color{red} \left( A^{(l-1)} \right)^T
\end{aligned}
$$

Similiary,
$$
\begin{aligned}
   \frac{\partial L}{\partial B^{(l)}}
   &= \color{red} \frac{\partial Z^{(l)}}{\partial B^{(l)}} \color{black} \cdot
   \color{magenta} \frac{\partial L}{\partial Z^{(l)}} \\
    &= \color{red} I \color{black} \cdot
   \color{magenta} \frac{\partial L}{\partial Z^{(l)}} \\
   &= \color{magenta} \frac{\partial L}{\partial Z^{(l)}}
\end{aligned}
$$


#### Output Layer


#### Hidden Layer 3

#### Hidden Layer 2

#### Hidden Layer 1



