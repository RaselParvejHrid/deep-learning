# Module 04 | Learning in Perceptrons – From Gradient Descent to Loss Functions

## My Solution to [Practice Problems](module04%20Practice%20Problems.pdf)

### Table of Contents {ignore=true}


<!-- @import "[TOC]" {cmd="toc" depthFrom=3 depthTo=3 orderedList=true} -->

<!-- code_chunk_output -->

1. [Q01. Why does the perceptron algorithm use class labels −1 and +1 instead of 0 and 1? Explain how this choice affects the perceptron update rule.](#q01-why-does-the-perceptron-algorithm-use-class-labels-1-and-1-instead-of-0-and-1-explain-how-this-choice-affects-the-perceptron-update-rule)
2. [Q02. Gradient descent usually updates weights by subtracting the gradient. Why does the perceptron update rule add the term $`\eta yx`$ instead?](#q02-gradient-descent-usually-updates-weights-by-subtracting-the-gradient-why-does-the-perceptron-update-rule-add-the-term-eta-yx-instead)
3. [Q03. Explain in your own words how the perceptron learning rule can be seen as a form of gradient descent. Also mention one limitation of the perceptron loss function.](#q03-explain-in-your-own-words-how-the-perceptron-learning-rule-can-be-seen-as-a-form-of-gradient-descent-also-mention-one-limitation-of-the-perceptron-loss-function)

<!-- /code_chunk_output -->



### Q01. Why does the perceptron algorithm use class labels −1 and +1 instead of 0 and 1? Explain how this choice affects the perceptron update rule. 

The question refers to two choices below about Perceptron's Activation Function, both with Weighted Sum, $`z \equiv wx + b`$.

$$
\begin{aligned}

a_{0,1}(z) &\equiv \begin{cases}
1, & \text{, if}\; z \geq 0 \\
0, &\text{, if}\; z < 0\end{cases} \\
\\
a_{\pm 1}(z) &\equiv \begin{cases}
1, & \text{if}\; z \geq 0 \\
-1, &\text{if}\; z < 0\end{cases}

\end{aligned}
$$

#### For $`a_{\pm 1}(z)`$
Standard Loss Function is
$$
L_{W, b}(x) \equiv \max(0, -yz)
$$

In case of Correct Classification, $`y`$ and $`z`$ do not disagree in sign $`\implies yz \geq 0 \implies -yz \leq 0 \implies L \equiv 0`$.

In case of Misclassification, $`y`$ and $`z`$ disagree in sign $`\implies yz < 0 \implies -yz > 0 \implies L \equiv -yz \equiv y(wx+b)`$.

Then, according the Gradient Descent Optimization, the Perceptron Learning Rule is
$$
\begin{aligned}
    w_{\text{new}} &= w_{\text{old}} - \eta  \frac{\partial L}{\partial w}
    \\
    &= \begin{cases}
        w_{\text{old}} - \eta  \frac{\partial}{\partial w} \left( 0 \right), & \text{for Correct Classification}\\
        w_{\text{old}} - \eta  \frac{\partial}{\partial w} \left( y \left( wx + b\right) \right), & \text{for Misclassification}
        \end{cases}
    \\
    &= \begin{cases}
        w_{\text{old}}, & \text{for Correct Classification}\\
        w_{\text{old}} - \eta yx, & \text{for Misclassification}
        \end{cases}
    \\
    \\
    b_{\text{new}} &= b_{\text{old}} - \eta  \frac{\partial L}{\partial b}
    \\
    &= \begin{cases}
        b_{\text{old}} - \eta  \frac{\partial}{\partial b} \left( 0 \right), & \text{for Correct Classification}\\
        b_{\text{old}} - \eta  \frac{\partial}{\partial b} \left( y \left( wx + b\right) \right), & \text{for Misclassification}
        \end{cases}
    \\
    &= \begin{cases}
        b_{\text{old}}, & \text{for Correct Classification}\\
        b_{\text{old}} - \eta y, & \text{for Misclassification}
        \end{cases}
    \\
\end{aligned}
$$

#### For $`a_{0, 1}(z)`$
A standard Loss Function is
$$
L_{W, b}(x) \equiv \left( y- \hat{y} \right)z
$$

In case of Correct Classification, $`\left( y- \hat{y} \right) = 0 \implies L \equiv 0`$.

In case of Misclassification, $`\left( y - \hat{y} \right) \neq 0`$ and $`L \equiv \left( y- \hat{y} \right)z \equiv \left( y- \hat{y} \right)(wx+b)`$.

Then, according the Gradient Descent Optimization, the Perceptron Learning Rule is
$$
\begin{aligned}
    w_{\text{new}} &= w_{\text{old}} - \eta  \frac{\partial L}{\partial w}
    \\
    &= \begin{cases}
        w_{\text{old}} - \eta  \frac{\partial}{\partial w} \left( 0 \right), & \text{for Correct Classification}\\
        w_{\text{old}} - \eta  \frac{\partial}{\partial w} \left( \left( y- \hat{y} \right)(wx+b) \right), & \text{for Misclassification}
        \end{cases}
    \\
    &= \begin{cases}
        w_{\text{old}}, & \text{for Correct Classification}\\
        w_{\text{old}} - \eta \left( y- \hat{y} \right)x, & \text{for Misclassification}
        \end{cases}
    \\
    \\
    b_{\text{new}} &= b_{\text{old}} - \eta  \frac{\partial L}{\partial b}
    \\
    &= \begin{cases}
        b_{\text{old}} - \eta  \frac{\partial}{\partial b} \left( 0 \right), & \text{for Correct Classification}\\
        b_{\text{old}} - \eta  \frac{\partial}{\partial b} \left( ( \left( y- \hat{y} \right)(wx+b) \right), & \text{for Misclassification}
        \end{cases}
    \\
    &= \begin{cases}
        b_{\text{old}}, & \text{for Correct Classification}\\
        b_{\text{old}} - \eta \left( y- \hat{y} \right), & \text{for Misclassification}
        \end{cases}
    \\
\end{aligned}
$$

#### Comparison

<table>
  <tr>
    <th rowspan="2">Perceptron</th>
    <th colspan="2">Activation Function</th>
  </tr>
  <tr>
    <td> $$a_{0,1}(z) \equiv \begin{cases} 1, & \text{, if}\; z \geq 0 \\ 0, &\text{, if}\; z < 0\end{cases}$$
    </td>
    <td>$$a_{\pm 1}(z) \equiv \begin{cases}
            1, & \text{if}\; z \geq 0 \\
            -1, &\text{if}\; z < 0\end{cases}$$</td>
  </tr>
  <tr>
    <th>Loss Function</th>
    <td>$$L_{W, b}(x) \equiv \left( y- \hat{y} \right)z$$</td>
    <td>$$L_{W, b}(x) \equiv \max(0, -yz)$$</td>
  </tr>
  <tr>
    <th>Learning Rule</th>
    <td>$$\begin{aligned}
        w_{\text{new}} &= w_{\text{old}} - \eta \left( y- \hat{y} \right)x \\ \\
        b_{\text{new}} &= b_{\text{old}} - \eta \left( y- \hat{y} \right)
    \end{aligned}$$</td>
    <td>$$\begin{aligned}
        w_{\text{new}} &= w_{\text{old}} - \eta yx \\ \\
        b_{\text{new}} &= b_{\text{old}} - \eta y
    \end{aligned}$$</td>
  </tr>
</table>

So, that's how our choice of class labels for Perceptron affetcts both Loss Function and Learning Rule.


### Q02. Gradient descent usually updates weights by subtracting the gradient. Why does the perceptron update rule add the term $`\eta yx`$ instead? 
Yes, `by subtracting the gradient` of the **Loss Function**.

In the special case of Perceptron, with $a_{\pm 1}(z) \equiv \begin{cases}
            1, & \text{if}\; z \geq 0 \\
            -1, &\text{if}\; z < 0\end{cases}$, as the activation function, a standard loss function is
$$
L_{W, b}(x) \equiv \max(0, -yz).
$$

Its gradients are
$$
\begin{aligned}
\frac{\partial L}{\partial w} &= yx \\
\frac{\partial L}{\partial w} &= y    
\end{aligned}
$$

That's why the Perceptron, using Gradient Descent Optimization in its own special case, updates its `Weights` and `Bias` with a term with $`y`$ and $`x`$ as factors, and also $`\eta`$, as generally suggested by Gradient Descent.
      
### Q03. Explain in your own words how the perceptron learning rule can be seen as a form of gradient descent. Also mention one limitation of the perceptron loss function.

Perceptron Learning Rule, is not an equivalent form of Gradient Descent, but an special form of Gradient Descent. Perceptron Learning Rule is derived applying the General `Gradient Descent Optimization Algorithm` on the special case of perceptron. See [the Derivation](#q01-why-does-the-perceptron-algorithm-use-class-labels-1-and-1-instead-of-0-and-1-explain-how-this-choice-affects-the-perceptron-update-rule).