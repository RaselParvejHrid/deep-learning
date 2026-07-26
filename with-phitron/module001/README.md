# Module 001 | Introduction to Deep Learning and Perceptrons

## My Solution to [Practice Problems](module001%20Practice%20Problems.pdf)

### Table of Contents
- [Module 001 | Introduction to Deep Learning and Perceptrons](#module-001--introduction-to-deep-learning-and-perceptrons)
  - [My Solution to Practice Problems](#my-solution-to-practice-problems)
    - [Table of Contents](#table-of-contents)
    - [Bias](#bias)
      - [What is a bias in a perceptron?](#what-is-a-bias-in-a-perceptron)
      - [Why do we add bias to a perceptron?](#why-do-we-add-bias-to-a-perceptron)
    - [Weights](#weights)
      - [What are weights in a perceptron?](#what-are-weights-in-a-perceptron)
      - [How do weights affect the output of a perceptron?](#how-do-weights-affect-the-output-of-a-perceptron)
    - [Perceptron Output](#perceptron-output)
      - [A perceptron has inputs $x_1 = 2$, $x_2 = 3$, weights $w_1 = 0.5$, $w_2 = 1$, and bias $b = 1$. Calculate the weighted sum (before activation).](#a-perceptron-has-inputs-x_1--2-x_2--3-weights-w_1--05-w_2--1-and-bias-b--1-calculate-the-weighted-sum-before-activation)
    - [Step function](#step-function)
      - [What is the step function and how does it work?](#what-is-the-step-function-and-how-does-it-work)
    - [Activation Function](#activation-function)
      - [Why do we need an activation function in a perceptron?](#why-do-we-need-an-activation-function-in-a-perceptron)
      - [Name three common activation functions.](#name-three-common-activation-functions)

---

### Bias

#### What is a bias in a perceptron?

The `Bias` $(b)$ is a parameter of a Perceptron, additional to `Input Weights` $(W = [w_i])$, that the perceptron learns during training, that adjusts the `Weighted Sum` $(W \cdot X)$ to the threshold of the `Activation Function` that follows.

#### Why do we add bias to a perceptron?

We add `Bias` $(b)$ to the `Weighted Sum` ($W \cdot X$) to adjust the weighted sum to the threshold of the `Activation Function` of the perceptron.

For `SKLearn#Perceptron`, the `Activation Function` is the `Step Function` as defined below.

$$
y_{step}(\text{Weighted Sum}) = \begin{cases}
1, & \text{Weighted Sum}\geq {\mathbf{\color{Blue} 0}} \\
0, & \text{Weighted Sum} <  {\mathbf{\color{Blue} 0}} \\
\end{cases}
$$

This ${\mathbf{\color{Blue} 0}}$ is the `(Decision) Threshold` of the `Activation Function`, for `SKLearn#Perceptron`.

During training, `SKLearn#Perceptron` finds optimal `b` so that adding it to the weighted sum adjusts the sum to the threshold ${\mathbf{\color{Blue} 0}}$, for the sake of correct activation decision.

### Weights

#### What are weights in a perceptron?

`Weights` $(W = [w_i])$ are the measures of importance of the individual features/inputs toward activation of the perceptron.

#### How do weights affect the output of a perceptron?
Suppose, from training, the Perceptron learns that the weight of input $x_i$ is $w_i$.

$$
\text{Influence of a Feature on Activation Decision:}
\begin{cases}
\text{None,} & \text{if } w_i \text{ close to } 0 \\
\text{Strongly Direct,} & \text{if } w_i \text{ largely positive} \\
\text{Strongly Inverse,} & \text{if } w_i \text{ largely negative}
\end{cases}
$$

### Perceptron Output

#### A perceptron has inputs $x_1 = 2$, $x_2 = 3$, weights $w_1 = 0.5$, $w_2 = 1$, and bias $b = 1$. Calculate the weighted sum (before activation).

We have,
$$
\begin{aligned}
   X &= \begin{bmatrix} x_1 & x_2 \end{bmatrix} \\
     &= \begin{bmatrix} 2 & 3 \end{bmatrix} \\
   W &= \begin{bmatrix} w_1 & w_2 \end{bmatrix} \\
     &= \begin{bmatrix} 0.5 & 1 \end{bmatrix} \\
   b &= 1 \\
   W \cdot X + b &= \begin{bmatrix} 0.5 & 1 \end{bmatrix} \cdot \begin{bmatrix} 2 & 3 \end{bmatrix} + 1 \\
   &= 1 + 3 + 1 \\
   &= 5
\end{aligned}
$$

So, the weighted sum is $5$.


### Step function

#### What is the step function and how does it work?
Step functions, as defined below, are one kind of activation functions used in a perceptron.

$$
y_{step}(\text{Weighted Sum}) = \begin{cases}
1, & \text{Weighted Sum}\geq {\mathbf{\color{Blue} \text{Threshold}}} \\
0, & \text{Weighted Sum} <  {\mathbf{\color{Blue} \text{Threshold}}} \\
\end{cases}
$$

For `SKLearn#Perceptron`, this $\mathbf{\color{Blue} \text{Threshold}} = 0$.

The `Weighted Sum` is the input to a step function, as to any other activation function. As obvious from the definition above, if the `Weigted Sum` is greater than or equal to the `Thresold` value, the perceptron (or the artificial neuron) fires, or gets activated, that is the output of the activation function is `1`. Otherwise, it is `0`— the neuron does not fire— it remains inactive.

### Activation Function

#### Why do we need an activation function in a perceptron?
To decide if the input values are, collectively, `important enough` for the perceptron to fire.

The `importance` are represented by the `Weighted Sum`, which is the input to the activation function. And the activation function decides, based on `Threshold` inside the function, if the importance is `enough`.

#### Name three common activation functions.
1. Sigmoid
2. tanh
3. ReLU