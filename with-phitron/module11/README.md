# Module 11 | Pytorch Training Pipeline From Scratch

## Problem Statement (from [Practice Problems](module11%20Practice%20Problems.pdf))

Q01. Consider:
```python
x = torch.tensor([2.0], requires_grad=True)
y = x**2 + 3*x + 1
y.backward()
```
What is the value inside x.grad?
Explain how PyTorch computed it step-by-step.

Q2. What happens in this case?
```python
x = torch.tensor([2.0])
y = x**2
y.backward()
```
Why does it fail?

## My solution

### Q01
`x.grad` is the value of the gradient $\frac{\partial y}{\partial x}$ at $x=2.0$.

$$
\begin{aligned}
   \left( \frac{\partial y}{\partial x} \right)_{x=2.0}
   &= \left( 2x + 3 \right)_{x=2.0} \\
   &= 2 \times 2.0 + 3 \\
   &= 7.0
\end{aligned}
$$

With `requires_grad = True` while creating tensor $x$ (in Line 1), PyTorch start building a computational graph of all operations involving $x$ and the result of operations.

$y$ (Line 2) becomes part of that graph, as it is created with operations involving . When we call `.backward()` on $y$, PyTorch traverse back the graph and applies Chain Rule to Compute Derivatives.


### Q02
Tensor $y$ is created from tensor $x$, by squaring. $x$ is not created with `requires_grad = True`. So, PyTorch infers we do not need gradient with respect t $x$, so it builds no computational graph. With no computational graph, no gradient. Hence, `.backward()` fails.