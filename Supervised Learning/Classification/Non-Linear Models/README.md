## Supervised learning with Support Vector Machines

Support Vector Machines (SVMs) are a powerful class of supervised learning algorithms used for classification and regression. They aim to find the hyperplane which best divides a dataset into classes. This method is particularly useful when the classes are not linearly separable. The beauty of SVMs lies in their utilization of geometry and the introduction of the kernel trick to handle non-linear data.

A Support Vector Machine is a supervised learning model that creates a hyperplane or set of hyperplanes in a high- or infinite-dimensional space, which can be used for classification, regression, or other tasks. The main goal is to separate the data in such a way that the margin between the two classes is maximized.

### The Concept of Margin
The margin is defined as the distance between the separating hyperplane and the nearest data point from either class. An SVM aims to maximize this margin, ensuring robustness and generalizability of the classification.

Given a hyperplane defined by:

```math
w⋅x+b=0
```
The distance of a point xi
 to this hyperplane is:
```math
|w⋅xi+b|/‖w‖
```

### Hard vs Soft Margin

**Hard Margin SVM:** Assumes that the data is linearly separable and tries to find the hyperplane without any misclassification. This can lead to overfitting in the presence of noise.

**Soft Margin SVM:** Allows some misclassifications in the hope of achieving a more generalizable model. This is controlled by a parameter, often called C.

### Kernel Trick

While SVM is linear by design, the Kernel trick allows it to work on non-linearly separable data. It does so by mapping the data into a higher-dimensional space, making it possible to find a hyperplane that separates the data.

```math
K(xi,xj)=ϕ(xi)⋅ϕ(xj)
```

#### Popular kernels include:

**1. Linear:**
```math
K(x,y)=x⋅y
```

**2. Polynomial:**
```math
K(x,y)=(1+x⋅y)d
```
**3. Radial basis function (RBF):**
```math
K(x,y)=exp(−γ‖x−y‖2)
```

### What is a support vector machine kernel?

In the context of Support Vector Machines (SVMs), a kernel is a function that computes the dot product between two vectors in a transformed space. Instead of explicitly transforming the vectors into this higher-dimensional space and then calculating the dot product, the kernel provides a shortcut by directly computing the dot product in the original space, which can often be computationally more efficient.

The kernel essentially allows SVMs to create non-linear decision boundaries without explicitly transforming the data.

#### Here’s a breakdown of the kernel’s significance:

Linearity & Non-linearity: SVMs, in their basic form, construct a linear hyperplane to separate data. But, real-world data is often not linearly separable. Kernels help SVMs deal with such non-linear data by implicitly mapping the data into a higher-dimensional space where it becomes linearly separable.

*Types of Kernels:*

**1. Linear Kernel:**

          Represents a linear decision boundary. Mathematically, it’s just the dot product of the two input vectors. (2 Row values)

**2. Polynomial Kernel:**

        Introduces polynomial decision boundaries (e.g., curves, parabolas). The degree of the polynomial determines the complexity of the decision boundary.

**3. Radial Basis Function (RBF) or Gaussian Kernel:**

        Can model very complex decision boundaries. It considers the distance between data points in its calculations.
