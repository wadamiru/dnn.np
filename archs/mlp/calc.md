# Calc. mlp

## Matrix Calculus Derivation: Linear Layer Gradients

### Mathematical Setup & Definitions

Consider a mini-batched linear transformation layer $f: \mathbb{R}^{N \times d_{\text{in}}} \to \mathbb{R}^{N \times d_{\text{out}}}$ defined by the pre-activation mapping:

$$Y = XW + \mathbf{1}_N b^T$$

Where:

* $X \in \mathbb{R}^{N \times d_{\text{in}}}$ : Batch Input Matrix
* $W \in \mathbb{R}^{d_{\text{in}} \times d_{\text{out}}}$ : Weight Parameter Matrix
* $b \in \mathbb{R}^{d_{\text{out}}}$ : Bias Parameter Vector
* $\mathbf{1}_N \in \mathbb{R}^{N \times 1}$ : Column Vector of Ones $(1, 1, \dots, 1)^T$
* $Y \in \mathbb{R}^{N \times d_{\text{out}}}$ : Pre-activation Output Matrix
* $L \in \mathbb{R}$ : Scalar Loss Value

Let the upstream loss gradient with respect to $Y$ be defined as:

$$\delta Y \equiv \nabla_Y L = \frac{\partial L}{\partial Y} \in \mathbb{R}^{N \times d_{\text{out}}}$$

---

### Axioms & Matrix Calculus Identities

> **Identity I (Frobenius Inner Product & Gradient Identification):**
> 
> $$\mathrm{d}L = \langle \nabla_A L, \mathrm{d}A \rangle = \text{Tr}\left( (\nabla_A L)^T \mathrm{d}A \right)$$
> 
> 
> 
> *Proof:* By definition of total differential, $\mathrm{d}L = \sum_{i,j} \frac{\partial L}{\partial A_{ij}} \mathrm{d}A_{ij} = \text{Tr}\left( \left(\frac{\partial L}{\partial A}\right)^T \mathrm{d}A \right)$.

> **Identity II (Matrix Differential Linearity):**
> 
> $$\mathrm{d}(AB) = (\mathrm{d}A)B + A(\mathrm{d}B)$$
> 
> 

> **Identity III (Cyclic Invariance & Transposition of Trace):**
> 
> $$\text{Tr}(ABC) = \text{Tr}(CAB) = \text{Tr}(BCA) \quad \text{and} \quad \text{Tr}(A) = \text{Tr}(A^T)$$
> 
> 

---

### Total Differential Expansion

Applying the total differential $\mathrm{d}(\cdot)$ to $Y = XW + \mathbf{1}_N b^T$:

$$\mathrm{d}Y = (\mathrm{d}X)W + X(\mathrm{d}W) + \mathbf{1}_N (\mathrm{d}b)^T$$

Expressing the scalar loss differential $\mathrm{d}L$ using Identity I:

$$\mathrm{d}L = \text{Tr}\left( \delta Y^T \mathrm{d}Y \right)$$

Substitute $\mathrm{d}Y$ into $\mathrm{d}L$ and expand by trace linearity:

$$\mathrm{d}L = \text{Tr}\left( \delta Y^T (\mathrm{d}X) W \right) + \text{Tr}\left( \delta Y^T X (\mathrm{d}W) \right) + \text{Tr}\left( \delta Y^T \mathbf{1}_N (\mathrm{d}b)^T \right)$$

Define the three terms as:

$$\mathrm{d}L = \mathcal{T}_X + \mathcal{T}_W + \mathcal{T}_b$$

---

### Derivation of Gradients

#### Input Gradient ($\nabla_X L$)

Isolate $\mathcal{T}_X$ and apply trace identities to isolate $\mathrm{d}X$:

$$\mathcal{T}_X = \text{Tr}\left( \delta Y^T (\mathrm{d}X) W \right)$$

$$\mathcal{T}_X = \text{Tr}\left( W \delta Y^T \mathrm{d}X \right) \quad \text{(Cyclic shift)}$$

$$\mathcal{T}_X = \text{Tr}\left( (W \delta Y^T \mathrm{d}X)^T \right) \quad \text{(Trace transpose)}$$

$$\mathcal{T}_X = \text{Tr}\left( (\delta Y W^T)^T \mathrm{d}X \right) \quad \text{(Transpose expansion)}$$

By matching with $\mathrm{d}L = \text{Tr}\left( (\nabla_X L)^T \mathrm{d}X \right)$:

$$\nabla_X L = \delta Y W^T \in \mathbb{R}^{N \times d_{\text{in}}}$$

---

#### Weight Gradient ($\nabla_W L$)

Isolate $\mathcal{T}_W$ and apply trace identities to isolate $\mathrm{d}W$:

$$\mathcal{T}_W = \text{Tr}\left( \delta Y^T X (\mathrm{d}W) \right)$$

$$\mathcal{T}_W = \text{Tr}\left( (X^T \delta Y)^T \mathrm{d}W \right) \quad \text{(Group and transpose)}$$

By matching with $\mathrm{d}L = \text{Tr}\left( (\nabla_W L)^T \mathrm{d}W \right)$:

$$\nabla_W L = X^T \delta Y \in \mathbb{R}^{d_{\text{in}} \times d_{\text{out}}}$$

---

#### Bias Gradient ($\nabla_b L$)

Isolate $\mathcal{T}_b$ and apply trace identities to isolate $\mathrm{d}b$:

$$\mathcal{T}_b = \text{Tr}\left( \delta Y^T \mathbf{1}_N (\mathrm{d}b)^T \right)$$

$$\mathcal{T}_b = \text{Tr}\left( (\mathrm{d}b)^T \delta Y^T \mathbf{1}_N \right) \quad \text{(Cyclic shift)}$$

$$\mathcal{T}_b = \text{Tr}\left( (\mathbf{1}_N^T \delta Y) \mathrm{d}b \right) \quad \text{(Transpose invariance)}$$

By matching with $\mathrm{d}L = \text{Tr}\left( (\nabla_b L)^T \mathrm{d}b \right)$:

$$\nabla_b L = \delta Y^T \mathbf{1}_N = \sum_{i=1}^{N} \delta Y_{i, \cdot} \in \mathbb{R}^{d_{\text{out}}}$$

---

### final: Linear

$$\nabla_X L = \delta Y W^T$$
$$\nabla_W L = X^T \delta Y$$
$$\nabla_b L = \delta Y^T \mathbf{1}_N$$

---
