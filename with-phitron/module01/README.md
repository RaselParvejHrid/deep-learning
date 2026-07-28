# Module 01 | Introduction to Deep Learning and Perceptrons

## My Solution to [Practice Problems](module01%20Practice%20Problems.pdf)

### Table of Contents {ignore=true}

<!-- @import "[TOC]" {cmd="toc" depthFrom=3 depthTo=4 orderedList=True} -->

<!-- code_chunk_output -->

1. [Bias](#bias)
    1. [What is a bias in a perceptron?](#what-is-a-bias-in-a-perceptron)
    2. [Why do we add bias to a perceptron?](#why-do-we-add-bias-to-a-perceptron)
2. [Weights](#weights)
    1. [What are weights in a perceptron?](#what-are-weights-in-a-perceptron)
    2. [How do weights affect the output of a perceptron?](#how-do-weights-affect-the-output-of-a-perceptron)
3. [Perceptron Output](#perceptron-output)
    1. [A perceptron has inputs $x_1 = 2$, $x_2 = 3$, weights $w_1 = 0.5$, $w_2 = 1$, and bias $b = 1$. Calculate the weighted sum (before activation).](#a-perceptron-has-inputs-x_1--2-x_2--3-weights-w_1--05-w_2--1-and-bias-b--1-calculate-the-weighted-sum-before-activation)
4. [Step function](#step-function)
    1. [What is the step function and how does it work?](#what-is-the-step-function-and-how-does-it-work)
5. [Activation Function](#activation-function)
    1. [Why do we need an activation function in a perceptron?](#why-do-we-need-an-activation-function-in-a-perceptron)
    2. [Name three common activation functions.](#name-three-common-activation-functions)

<!-- /code_chunk_output -->



---

<!-- pagebreak -->

### Bias

#### What is a bias in a perceptron?

The `Bias` $(b)$ is a parameter of a Perceptron, additional to `Input Weights` $(W = [w_i])$, that the perceptron learns during training, that adjusts the `Weighted Sum` $(W \cdot X)$ to the threshold of the `Activation Function` that follows.

#### Why do we add bias to a perceptron?

Let's recall a key step in how a trained perception predicts. It starts with `combining the inputs linearly`. The better the `Linear Combination`, the better the prediction. Well, then what decides a linear combination good or bad, and how is a perceptron supposed to have it or acquire it?

A perception learns the optimal linear combination parameters (Weights and Bias) from training.

What ensures optimality? The perceptron learns the parameters, during training, with supervision of truth values and by optimizing a loss function, called 'Perceptron Loss', that is tied to its activation function, namely, the step function.

This Linear Combination Parameters (Weights and Bias) represent a 'Decision Boundary' on the feature space. This decision boundary is a line on 2D, a plane on 3D (generally, a hyperplane on any dimensional feature space).

From coordinate geometry, not from ML, not from DL, from Coordinate Geometry, this dicision boundary has a equation of locus, with Feature values as coordinate variables, and Weights and Bias as equation parameter.

From Coordinate Geometry, the `general equation of n-dimensional hyperplane` is this:
```math
w_1 x_1 + w_2 x_2 + \cdots \cdots \cdots + w_n x_n + b = 0
```

If the decision boundary passes through the coordinate origin, this $\text{Bias} = b = 0$, and we can omit the term, in other wors, no bias added.

If the decision boundary does not pass through the coordinate origin, the bias is not zero. It must be there as put by Coordinate Geometry, again not by ML, not by DL.

We add the `Bias` term while predicting with a perceptron, neither because we want it for mere engineering convenience, nor it is an engineering trick, nor it's a ML/DL technique. It is already in the perception, or not, decided during training depending the distribution of the feature space, to be added (if there) with $\sum w_i x_i$ during prediction.

<!-- pagebreak -->

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

<!-- pagebreak -->

### Perceptron Output

#### A perceptron has inputs $x_1 = 2$, $x_2 = 3$, weights $w_1 = 0.5$, $w_2 = 1$, and bias $b = 1$. Calculate the weighted sum (before activation).

We have,
```math
\begin{aligned}
   X &= \begin{bmatrix} x_1 & x_2 \end{bmatrix} \\
     &= \begin{bmatrix} 2 & 3 \end{bmatrix} \\
     \\
   W &= \begin{bmatrix} w_1 & w_2 \end{bmatrix} \\
     &= \begin{bmatrix} 0.5 & 1 \end{bmatrix} \\
     \\
   b &= 1 \\
   \\
   W \cdot X + b &= \begin{bmatrix} 0.5 & 1 \end{bmatrix} \cdot \begin{bmatrix} 2 & 3 \end{bmatrix} + 1 \\
   &= 1 + 3 + 1 \\
   &= 5
\end{aligned}
```

So, the weighted sum is $5$.

<!-- pagebreak -->

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

<!-- pagebreak -->

### Activation Function

#### Why do we need an activation function in a perceptron?
To decide if the input values are, collectively, `important enough` for the perceptron to fire.

The `importance` is represented by the `Weighted Sum`, which is the input to the activation function. And the activation function decides, based on `Threshold` inside the function, if the importance is `enough`.

#### Name three common activation functions.
1. Sigmoid
2. tanh
3. ReLU