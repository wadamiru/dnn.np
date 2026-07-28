# MLP Derivations: Matrix Calculus Reference

This document provides rigorous matrix calculus derivations for key neural network layers and loss functions:
1. **Linear (Fully Connected) Layer**
2. **Gaussian Error Linear Unit (GELU) Activation**
3. **Rectified Linear Unit (ReLU) Activation**
4. **Softmax Cross-Entropy Loss**

---

## Universal Axioms & Matrix Calculus Identities

The following foundational identities apply across all matrix calculus derivations in this document.

> **Identity I (Frobenius Inner Product & Gradient Identification):**  
> For a scalar loss function $L(A)$ over a matrix parameter $A \in \mathbb{R}^{m \times n}$:
> 
> $$\mathrm{d}L = \langle \nabla_A L, \mathrm{d}A \rangle_F = \text{Tr}\left( (\nabla_A L)^T \mathrm{d}A \right)$$
> 
> *Proof:* By definition of the total differential, $\mathrm{d}L = \sum_{i,j} \frac{\partial L}{\partial A_{ij}} \mathrm{d}A_{ij} = \text{Tr}\left( \left(\frac{\partial L}{\partial A}\right)^T \mathrm{d}A \right)$.

> **Identity II (Cyclic Permutation & Transposition of Trace):**  
> For conformable matrices $A, B, C$:
> 
> $$\text{Tr}(ABC) = \text{Tr}(CAB) = \text{Tr}(BCA) \quad \text{and} \quad \text{Tr}(A) = \text{Tr}(A^T)$$

> **Identity III (Hadamard Commutativity & Duality inside Trace):**  
> For matrices $A, B, C \in \mathbb{R}^{m \times n}$:
> 
> $$\text{Tr}\left( A^T (B \circ C) \right) = \text{Tr}\left( (A \circ B)^T C \right)$$
> 
> *Proof:* $\sum_{i,j} A_{ij} (B_{ij} C_{ij}) = \sum_{i,j} (A_{ij} B_{ij}) C_{ij}$.

> **Identity IV (Differential Product Rules):**  
> Matrix multiplication differential: $\mathrm{d}(AB) = (\mathrm{d}A)B + A(\mathrm{d}B)$  
> Hadamard product differential: $\mathrm{d}(A \circ B) = (\mathrm{d}A) \circ B + A \circ (\mathrm{d}B)$

---

## 1. Linear (Fully Connected) Layer Gradients

### 1.1 Mathematical Setup
Consider a mini-batched linear transformation $f: \mathbb{R}^{N \times d_{\text{in}}} \to \mathbb{R}^{N \times d_{\text{out}}}$ mapped by:

$$Y = XW + \mathbf{1}_N b^T$$

Where:
* $X \in \mathbb{R}^{N \times d_{\text{in}}}$ : Batch Input Matrix
* $W \in \mathbb{R}^{d_{\text{in}} \times d_{\text{out}}}$ : Weight Parameter Matrix
* $b \in \mathbb{R}^{d_{\text{out}}}$ : Bias Parameter Vector
* $\mathbf{1}_N \in \mathbb{R}^{N \times 1}$ : Vector of Ones $(1, 1, \dots, 1)^T$
* $Y \in \mathbb{R}^{N \times d_{\text{out}}}$ : Pre-activation Output Matrix
* $\delta Y \equiv \nabla_Y L = \frac{\partial L}{\partial Y} \in \mathbb{R}^{N \times d_{\text{out}}}$ : Upstream Loss Gradient

### 1.2 Total Differential Expansion
Applying Identity IV to $Y = XW + \mathbf{1}_N b^T$:

$$\mathrm{d}Y = (\mathrm{d}X)W + X(\mathrm{d}W) + \mathbf{1}_N (\mathrm{d}b)^T$$

Expressing scalar differential $\mathrm{d}L$ using Identity I and expanding:

$$\mathrm{d}L = \text{Tr}\left( \delta Y^T \mathrm{d}Y \right) = \text{Tr}\left( \delta Y^T (\mathrm{d}X) W \right) + \text{Tr}\left( \delta Y^T X (\mathrm{d}W) \right) + \text{Tr}\left( \delta Y^T \mathbf{1}_N (\mathrm{d}b)^T \right)$$

Let $\mathrm{d}L = \mathcal{T}_X + \mathcal{T}_W + \mathcal{T}_b$.

### 1.3 Derivation of Gradients

#### Input Gradient ($\nabla_X L$)
Applying Identity II to isolate $\mathrm{d}X$:

$$\mathcal{T}_X = \text{Tr}\left( \delta Y^T (\mathrm{d}X) W \right) = \text{Tr}\left( W \delta Y^T \mathrm{d}X \right) = \text{Tr}\left( (\delta Y W^T)^T \mathrm{d}X \right)$$

Matching with Identity I yields:

$$\nabla_X L = \delta Y W^T \in \mathbb{R}^{N \times d_{\text{in}}}$$

#### Weight Gradient ($\nabla_W L$)
Applying Identity II to isolate $\mathrm{d}W$:

$$\mathcal{T}_W = \text{Tr}\left( \delta Y^T X (\mathrm{d}W) \right) = \text{Tr}\left( (X^T \delta Y)^T \mathrm{d}W \right)$$

Matching with Identity I yields:

$$\nabla_W L = X^T \delta Y \in \mathbb{R}^{d_{\text{in}} \times d_{\text{out}}}$$

#### Bias Gradient ($\nabla_b L$)
Applying Identity II to isolate $\mathrm{d}b$:

$$\mathcal{T}_b = \text{Tr}\left( \delta Y^T \mathbf{1}_N (\mathrm{d}b)^T \right) = \text{Tr}\left( (\mathrm{d}b)^T \delta Y^T \mathbf{1}_N \right) = \text{Tr}\left( (\mathbf{1}_N^T \delta Y) \mathrm{d}b \right)$$

Matching with Identity I yields:

$$\nabla_b L = \delta Y^T \mathbf{1}_N = \sum_{i=1}^{N} \delta Y_{i, \cdot} \in \mathbb{R}^{d_{\text{out}}}$$

---

## 2. GELU Layer Gradients

### 2.1 Mathematical Setup
The element-wise Gaussian Error Linear Unit mapping $f: \mathbb{R}^{N \times d} \to \mathbb{R}^{N \times d}$ is defined by:

$$Y = \text{GELU}(X) = X \circ \Phi(X)$$

Where $\Phi(X) = \frac{1}{2} \left( \mathbf{1}_{N \times d} + \text{erf}\left( \frac{X}{\sqrt{2}} \right) \right)$ is the Gaussian CDF, and $\delta Y \equiv \nabla_Y L$.

### 2.2 Derivative Identity
Using the standard normal PDF $\Phi'(x) = \frac{\mathrm{d}\Phi(x)}{\mathrm{d}x} = \frac{1}{\sqrt{2\pi}} e^{-\frac{x^2}{2}}$, the differential of $\Phi(X)$ is:

$$\mathrm{d}\Phi(X) = \Phi'(X) \circ \mathrm{d}X = \left( \frac{1}{\sqrt{2\pi}} \exp\left( -\frac{X^{\circ 2}}{2} \right) \right) \circ \mathrm{d}X$$

### 2.3 Derivation of Gradient

Applying Identity IV (Hadamard differential rule) to $Y = X \circ \Phi(X)$:

$$\mathrm{d}Y = (\mathrm{d}X) \circ \Phi(X) + X \circ \mathrm{d}\Phi(X)$$

Substituting $\mathrm{d}Y$ into $\mathrm{d}L = \text{Tr}\left( \delta Y^T \mathrm{d}Y \right)$ yields two distinct pathways: 
$$\mathrm{d}L = T_{\text{direct}} + T_{\text{rate}}$$

#### Direct Pathway ($\mathcal{T}_{\text{direct}}$)
Using Identity III:

$$\mathcal{T}_{\text{direct}} = \text{Tr}\left( \delta Y^T ((\mathrm{d}X) \circ \Phi(X)) \right) = \text{Tr}\left( (\delta Y \circ \Phi(X))^T \mathrm{d}X \right)$$

#### Rate Pathway ($\mathcal{T}_{\text{rate}}$)
Applying Identity III twice:

$$\mathcal{T}_{\text{rate}} = \text{Tr}\left( \delta Y^T (X \circ \mathrm{d}\Phi(X)) \right) = \text{Tr}\left( (\delta Y \circ X)^T \mathrm{d}\Phi(X) \right)$$

$$\mathcal{T}_{\text{rate}} = \text{Tr}\left( \left( \delta Y \circ \frac{X}{\sqrt{2\pi}} \circ \exp\left( -\frac{X^{\circ 2}}{2} \right) \right)^T \mathrm{d}X \right)$$

#### Recombination ($\nabla_X L$)
Factoring out $\mathrm{d}X$ gives the exact gradient:

$$\nabla_X L = \delta Y \circ \left( \Phi(X) + \frac{X}{\sqrt{2\pi}} \circ \exp\left( -\frac{X^{\circ 2}}{2} \right) \right)$$

### 2.4 Derivation for Fast $\tanh$ Approximation

For the fast approximation, $Y = \text{GELU}_{\text{tanh}}(X) = \frac{1}{2} X \circ \left( \mathbf{1}_{N \times d} + \tanh(U) \right)$, where:

$$U = \sqrt{\frac{2}{\pi}} \left( X + 0.044715 X^{\circ 3} \right)$$

Applying the Hadamard product rule to $Y$:

$$\text{d}Y = \frac{1}{2} (\text{d}X) \circ \left( \mathbf{1}_{N \times d} + \tanh(U) \right) + \frac{1}{2} X \circ \text{d}\tanh(U)$$

#### Differential of the Inner Argument ($\text{d}U$)

Taking the Hadamard differential of $U$ with respect to $X$:

$$\text{d}U = \sqrt{\frac{2}{\pi}} \left( \mathbf{1}_{N \times d} + 3 \cdot 0.044715 X^{\circ 2} \right) \circ \text{d}X = \sqrt{\frac{2}{\pi}} \left( \mathbf{1}_{N \times d} + 0.134145 X^{\circ 2} \right) \circ \text{d}X$$

#### Differential of the $\tanh$ Activation ($\text{d }\tanh(U)$)

Using the element-wise derivative identity $\frac{\text{d}}{\text{d}u} \tanh(u) = 1 - \tanh^2(u)$:

$$\text{d}\tanh(U) = \left( \mathbf{1}_{N \times d} - \tanh^{\circ 2}(U) \right) \circ \text{d}U$$

Substituting $\text{d}U$ yields:

$$\text{d}\tanh(U) = \left( \mathbf{1}_{N \times d} - \tanh^{\circ 2}(U) \right) \circ \sqrt{\frac{2}{\pi}} \left( \mathbf{1}_{N \times d} + 0.134145 X^{\circ 2} \right) \circ \text{d}X$$

#### Total Differential Recombination ($\text{d}Y$)

Substituting $\text{d}\tanh(U)$ back into $\text{d}Y$:

$$\text{d}Y = \left[ 0.5 \left( \mathbf{1}_{N \times d} + \tanh(U) \right) + 0.5 X \circ \left( \mathbf{1}_{N \times d} - \tanh^{\circ 2}(U) \right) \circ \sqrt{\frac{2}{\pi}} \left( \mathbf{1}_{N \times d} + 0.134145 X^{\circ 2} \right) \right] \circ \text{d}X$$

#### Trace Identification ($\nabla_X L$)

Plugging $\text{d}Y$ into $\text{d}L = \text{Tr}\left( \delta Y^T \text{d}Y \right)$ and applying Identity III to shift the Hadamard factor:

$$\text{d}L = \text{Tr}\left( \left( \delta Y \circ \left( 0.5 \left( \mathbf{1}_{N \times d} + \tanh(U) \right) + 0.5 X \circ \left( \mathbf{1}_{N \times d} - \tanh^{\circ 2}(U) \right) \circ \sqrt{\frac{2}{\pi}} \left( \mathbf{1}_{N \times d} + 0.134145 X^{\circ 2} \right) \right) \right)^T \text{d}X \right)$$

Extracting $\nabla_X L = \frac{\partial L}{\partial X}$ directly confirms:

$$\nabla_X L = \delta Y \circ \left( 0.5 \left( \mathbf{1}_{N \times d} + \tanh(U) \right) + 0.5 X \circ \left( \mathbf{1}_{N \times d} - \tanh^{\circ 2}(U) \right) \circ \sqrt{\frac{2}{\pi}} \left( \mathbf{1}_{N \times d} + 0.134145 X^{\circ 2} \right) \right)$$

---

## 3. ReLU Layer Gradients

### 3.1 Mathematical Setup
The element-wise Rectified Linear Unit (ReLU) mapping $f: \mathbb{R}^{N \times d} \to \mathbb{R}^{N \times d}$ is defined by:

$$Y = \text{ReLU}(X) = \max(\mathbf{0}_{N \times d}, X) = X \circ H(X)$$

Where $H(X) = \mathbb{I}(X > 0)$ represents the element-wise Heaviside step indicator function.

### 3.2 Derivation of Gradient
Applying Identity IV to $Y = X \circ H(X)$:

$$\mathrm{d}Y = (\mathrm{d}X) \circ H(X) + X \circ \mathrm{d}H(X)$$

Since $H(X)$ is piecewise constant everywhere except at $X=0$ (a set of measure zero), its differential vanishes almost everywhere ($\mathrm{d}H(X) = \mathbf{0}_{N \times d}$). Thus:

$$\mathrm{d}Y = (\mathrm{d}X) \circ H(X)$$

Applying Identity III to $\mathrm{d}L = \text{Tr}\left( \delta Y^T \mathrm{d}Y \right)$:

$$\mathrm{d}L = \text{Tr}\left( \delta Y^T ((\mathrm{d}X) \circ H(X)) \right) = \text{Tr}\left( (\delta Y \circ \mathbb{I}(X > 0))^T \mathrm{d}X \right)$$

Matching with Identity I yields:

$$\nabla_X L = \delta Y \circ \mathbb{I}(X > 0) = \begin{cases} \delta Y_{ij} & \text{if } X_{ij} > 0 \\ 0 & \text{if } X_{ij} \le 0 \end{cases}$$

---

## 4. Softmax Cross-Entropy Loss Gradients

### 4.1 Mathematical Setup
Map unnormalized logits $Z \in \mathbb{R}^{N \times C}$ to class probabilities $P \in \mathbb{R}^{N \times C}$ evaluated against standard targets $Y \in \mathbb{R}^{N \times C}$:

$$P = \text{Softmax}(Z) = \left( \exp(Z) \mathbf{1}_C \right)^{-1}_{\mathrm{diag}} \exp(Z)$$

Where $\sum_{j=1}^C P_{ij} = 1$ and $\sum_{j=1}^C Y_{ij} = 1$. The averaged loss $L$ is:

$$L(P, Y) = -\frac{1}{N} \mathbf{1}_N^T \left( Y \circ \ln(P) \right) \mathbf{1}_C$$

### 4.2 Row-Vector Differential Identity
For a single sample row $z_i \in \mathbb{R}^{1 \times C}$ and probability row $p_i = \text{Softmax}(z_i)$:

$$\mathrm{d}p_i = \mathrm{d}z_i \left( \mathrm{diag}\,(p_i) - p_i^T p_i \right)$$

### 4.3 Total Differential Expansion
Applying differential $\mathrm{d}(\cdot)$ directly to $L(P, Y)$:

$$\mathrm{d}L = -\frac{1}{N} \text{Tr}\left( \mathbf{1}_C \mathbf{1}_N^T \mathrm{d}(Y \circ \ln(P)) \right) = -\frac{1}{N} \text{Tr}\left( \mathbf{1}_N \mathbf{1}_C^T (Y \circ P^{\circ -1} \circ \mathrm{d}P)^T \right)$$

By Identity III and row normalization ($\mathbf{1}_N \mathbf{1}_C^T$ being all ones):

$$\mathrm{d}L = -\frac{1}{N} \text{Tr}\left( (Y \circ P^{\circ -1})^T \mathrm{d}P \right)$$

### 4.4 Row Expansion & Matrix Re-consolidation
Expanding row-wise for sample $i \in \{1, \dots, N\}$ and substituting row differential $\mathrm{d}p_i$:

$$\mathrm{d}L = -\frac{1}{N} \sum_{i=1}^N \mathrm{d}p_i (y_i \circ p_i^{\circ -1})^T = -\frac{1}{N} \sum_{i=1}^N \mathrm{d}z_i \left( \mathrm{diag}\,(p_i) - p_i^T p_i \right) (y_i \circ p_i^{\circ -1})^T$$

Evaluating individual components:
1. $\mathrm{diag}\,(p_i) (y_i \circ p_i^{\circ -1})^T = y_i^T$
2. $p_i^T p_i (y_i \circ p_i^{\circ -1})^T = p_i^T \left( \sum_{j=1}^C y_{ij} \right) = p_i^T (1) = p_i^T$

Substituting back gives:

$$\mathrm{d}L = -\frac{1}{N} \sum_{i=1}^N \mathrm{d}z_i (y_i^T - p_i^T) = \frac{1}{N} \sum_{i=1}^N \mathrm{d}z_i (p_i - y_i)^T$$

Re-assembling row vectors into matrix trace form:

$$\mathrm{d}L = \frac{1}{N} \text{Tr}\left( (P - Y)^T \mathrm{d}Z \right)$$

Matching with Identity I yields the final average gradient:

$$\nabla_Z L = \frac{1}{N} (P - Y) \in \mathbb{R}^{N \times C}$$

---

## 5. Summary Matrix Gradient Table

| Module / Layer | Forward Mapping | Computed Gradient |
| :--- | :--- | :--- |
| **Linear (Input)** | $Y = XW + \mathbf{1}_N b^T$ | $\nabla_X L = \delta Y W^T$ |
| **Linear (Weights)** | $Y = XW + \mathbf{1}_N b^T$ | $\nabla_W L = X^T \delta Y$ |
| **Linear (Bias)** | $Y = XW + \mathbf{1}_N b^T$ | $\nabla_b L = \delta Y^T \mathbf{1}_N$ |
| **GELU** | $Y = X \circ \Phi(X)$ | $\nabla_X L = \delta Y \circ \left( \Phi(X) + \frac{X}{\sqrt{2\pi}} e^{-\frac{X^{\circ 2}}{2}} \right)$ |
| **ReLU** | $Y = \max(\mathbf{0}, X)$ | $\nabla_X L = \delta Y \circ \mathbb{I}(X > 0)$ |
| **Softmax Cross-Entropy** | $P = \text{Softmax}(Z)$ | $\nabla_Z L = \frac{1}{N}(P - Y)$ |