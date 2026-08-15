# Module 10 | From Computational Graph to Autograd Implementation <!-- omit in toc -->





## Problem Statement (Rephrased from [Practice Problems](module10%20Practice%20Problems.pdf))

## SECTION A: Conceptual Understanding

1. Why do we need **autograd** if we already understand calculus and can compute derivatives manually?

2. What problem does autograd solve when working with **deep neural networks containing many layers and parameters**?

3. What is a **computational graph**? Explain it using the simple expression:

   $$
   y = (x^2 + 3x) \times 2
   $$

4. In a computational graph:

   * What do the **nodes** represent?
   * What do the **edges** represent?

5. Why is a computational graph called a **Directed Acyclic Graph (DAG)**?

6. What would happen to gradient flow if the computational graph contained a **cycle**?

7. In your own words, explain how **autograd automatically applies the chain rule** to compute gradients.

---

## SECTION B: Computational Graph and Chain Rule

1. Consider:

   $$
   x = 2
   $$

   $$
   y = x^2
   $$

   $$
   z = 3y
   $$

   $$
   L = z + 4
   $$

   **Tasks:**

   * Draw the computational graph.
   * Manually compute $\frac{dL}{dx}$ using the chain rule.

2. Suppose:

   $$
   y = \operatorname{sigmoid}(wx+b)
   $$

   and

   $$
   L = (y-t)^2
   $$

   Write the **complete chain-rule expression** for $\frac{dL}{dw}$ before simplifying it.

3. Why does the combination of **Sigmoid activation + Binary Cross-Entropy loss** simplify the gradient to:

   $$
   \frac{\partial L}{\partial z}
   = y_{\text{pred}} - y
   $$

4. What would change in the **gradient flow and gradient computation** if we used **ReLU** instead of **Sigmoid**?

## My Solution
