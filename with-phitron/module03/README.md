# Module 03 | Decision Boundaries and Perceptron Learning

## My Solution to [Practice Problems](module03%20Practice%20Problems.pdf)

### Table of Contents {ignore=true}


<!-- @import "[TOC]" {cmd="toc" depthFrom=3 depthTo=4 orderedList=true} -->

<!-- code_chunk_output -->

1. [Conceptual Questions](#conceptual-questions)
    1. [What is a decision boundary, and how does it relate to classification tasks?](#what-is-a-decision-boundary-and-how-does-it-relate-to-classification-tasks)
    2. [How does the bias term in a neuron affect the position of the decision boundary?](#how-does-the-bias-term-in-a-neuron-affect-the-position-of-the-decision-boundary)
    3. [Explain with an example how changing weights affects the orientation of the decision boundary.](#explain-with-an-example-how-changing-weights-affects-the-orientation-of-the-decision-boundary)
    4. [Why is it important to visualize the decision boundary when training a model?](#why-is-it-important-to-visualize-the-decision-boundary-when-training-a-model)
2. [Calculation Problems](#calculation-problems)
    1. [A perceptron has initial Weights $w=[0.2 -0.1]$, initial Bias $b=0.1$ and Learning Rate $\eta = 0.1$. For a training example $x = [1 1]$, target is $t=1$ and the perceptron output is $y=0$. Calculate the new weights and bias after one update, using $w_{\text{new}} \leftarrow w_{\text{old}} + \eta (t-y) x$ and $b_{\text{new}} \leftarrow b_{\text{old}} + \eta (t-y)$.](#a-perceptron-has-initial-weights-w02--01-initial-bias-b01-and-learning-rate-eta--01-for-a-training-example-x--1-1-target-is-t1-and-the-perceptron-output-is-y0-calculate-the-new-weights-and-bias-after-one-update-using-w_textnew-leftarrow-w_textold--eta-t-y-x-and-b_textnew-leftarrow-b_textold--eta-t-y)

<!-- /code_chunk_output -->



### Conceptual Questions

#### What is a decision boundary, and how does it relate to classification tasks?

#### How does the bias term in a neuron affect the position of the decision boundary?

#### Explain with an example how changing weights affects the orientation of the decision boundary.

#### Why is it important to visualize the decision boundary when training a model?

### Calculation Problems

#### A perceptron has initial Weights $w=[0.2 -0.1]$, initial Bias $b=0.1$ and Learning Rate $\eta = 0.1$. For a training example $x = [1 1]$, target is $t=1$ and the perceptron output is $y=0$. Calculate the new weights and bias after one update, using $w_{\text{new}} \leftarrow w_{\text{old}} + \eta (t-y) x$ and $b_{\text{new}} \leftarrow b_{\text{old}} + \eta (t-y)$.