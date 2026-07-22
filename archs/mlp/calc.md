# Calc. mlp

## Matrix Calc. Derivation: Linear Layer Gradients

### 1. Mathematical Setup & Definitions

Consider a mini-batched linear transformation layer $f: \mathbb{R}^{N \times d_{\text{in}}} \to \mathbb{R}^{N \times d_{\text{out}}}$ defined by the pre-activation mapping:

$$Y = XW + \mathbf{1}_N b^T$$

$$\begin{aligned} \text{\textbf{Where:}} \quad &X \in \mathbb{R}^{N \times d_{\text{in}}} && \text{Batch Input Matrix} \\ &W \in \mathbb{R}^{d_{\text{in}} \times d_{\text{out}}} && \text{Weight Parameter Matrix} \\ &b \in \mathbb{R}^{d_{\text{out}}} && \text{Bias Parameter Vector} \\ &\mathbf{1}_N \in \mathbb{R}^{N \times 1} && \text{Column Vector of Ones } (1, 1, \dots, 1)^T \\ &Y \in \mathbb{R}^{N \times d_{\text{out}}} && \text{Pre-activation Output Matrix} \\ &L \in \mathbb{R} && \text{Scalar Loss Value} \end{aligned}$$

Let the adjoint matrix (upstream loss gradient) w.r.t. the output $Y$ be defined as:

$$\delta Y \equiv \nabla_Y L = \frac{\partial L}{\partial Y} \in \mathbb{R}^{N \times d_{\text{out}}}$$

---

### 2. Axioms & Matrix Calculus Identities

The derivation relies on three fundamental identities in matrix calculus:

> **Identity I (Frobenius Inner Product & Gradient Identification):**
> 
> $$\mathrm{d}L = \langle \nabla_A L, \mathrm{d}A \rangle_{\mathrm{F}} = \text{Tr}\left( (\nabla_A L)^T \mathrm{d}A \right)$$
> 
> 
> 
> *Proof:* By definition of the total differential for a real-valued matrix function $L(A)$, $\mathrm{d}L = \sum_{i,j} \frac{\partial L}{\partial A_{ij}} \mathrm{d}A_{ij} = \text{Tr}\left( \left(\frac{\partial L}{\partial A}\right)^T \mathrm{d}A \right)$.

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

### 3. Total Differential Expansion

Applying the total differential operator $\mathrm{d}(\cdot)$ to the forward mapping $Y = XW + \mathbf{1}_N b^T$:

$$\mathrm{d}Y = (\mathrm{d}X)W + X(\mathrm{d}W) + \mathbf{1}_N (\mathrm{d}b)^T \quad \text{\small [By Identity II]}$$

Expressing the differential change in scalar loss $\mathrm{d}L$ via the upstream gradient $\delta Y$:

$$\mathrm{d}L = \text{Tr}\left( \delta Y^T \mathrm{d}Y \right) \quad \text{\small [By Identity I]}$$

Substituting $\mathrm{d}Y$ into $\mathrm{d}L$ and invoking the linearity of the trace operator:

$$\begin{aligned} \mathrm{d}L &= \text{Tr}\left( \delta Y^T \left( (\mathrm{d}X)W + X(\mathrm{d}W) + \mathbf{1}_N (\mathrm{d}b)^T \right) \right) \\ &= \underbrace{\text{Tr}\left( \delta Y^T (\mathrm{d}X) W \right)}_{\mathcal{T}_X} + \underbrace{\text{Tr}\left( \delta Y^T X (\mathrm{d}W) \right)}_{\mathcal{T}_W} + \underbrace{\text{Tr}\left( \delta Y^T \mathbf{1}_N (\mathrm{d}b)^T \right)}_{\mathcal{T}_b} \end{aligned}$$

---

### 4. Derivation of Adjoint Operators

#### 4.1 Input Gradient ($\nabla_X L$)

Isolating the differential term $\mathcal{T}_X$ and manipulating its structure to isolate $\mathrm{d}X$:

$$\begin{aligned} \mathcal{T}_X &= \text{Tr}\left( \delta Y^T (\mathrm{d}X) W \right) \\ &= \text{Tr}\left( W \delta Y^T \mathrm{d}X \right) && \text{\small [Cyclic shift: } \text{Tr}(ABC) \to \text{Tr}(CAB) \text{]} \\ &= \text{Tr}\left( (W \delta Y^T \mathrm{d}X)^T \right) && \text{\small [Trace transpose invariance: } \text{Tr}(M) = \text{Tr}(M^T) \text{]} \\ &= \text{Tr}\left( (\mathrm{d}X)^T \delta Y W^T \right) && \text{\small [Matrix transpose property: } (ABC)^T = C^T B^T A^T \text{]} \\ &= \text{Tr}\left( (\delta Y W^T)^T \mathrm{d}X \right) && \text{\small [Cyclic shift to canonical form } \text{Tr}(M^T \mathrm{d}X) \text{]} \end{aligned}$$

By identification against Identity I ($\mathrm{d}L = \text{Tr}\left( (\nabla_X L)^T \mathrm{d}X \right)$):

$${\nabla_X L = \delta Y W^T} \quad \in \mathbb{R}^{N \times d_{\text{in}}}$$

---

#### 4.2 Weight Gradient ($\nabla_W L$)

Isolating the differential term $\mathcal{T}_W$ and manipulating its structure to isolate $\mathrm{d}W$:

$$\begin{aligned} \mathcal{T}_W &= \text{Tr}\left( \delta Y^T X (\mathrm{d}W) \right) \\ &= \text{Tr}\left( (X^T \delta Y)^T \mathrm{d}W \right) && \text{\small [Group } (X^T \delta Y) \text{ and apply } (A^T B)^T = B^T A \text{]} \end{aligned}$$

By identification against Identity I ($\mathrm{d}L = \text{Tr}\left( (\nabla_W L)^T \mathrm{d}W \right)$):

$${\nabla_W L = X^T \delta Y} \quad \in \mathbb{R}^{d_{\text{in}} \times d_{\text{out}}}$$

---

#### 4.3 Bias Gradient ($\nabla_b L$)

Isolating the differential term $\mathcal{T}_b$ and manipulating its structure to isolate $\mathrm{d}b$:

$$\begin{aligned} \mathcal{T}_b &= \text{Tr}\left( \delta Y^T \mathbf{1}_N (\mathrm{d}b)^T \right) \\ &= \text{Tr}\left( (\mathrm{d}b)^T \delta Y^T \mathbf{1}_N \right) && \text{\small [Cyclic shift: } \text{Tr}(AB) \to \text{Tr}(BA) \text{]} \\ &= \text{Tr}\left( \left( \delta Y^T \mathbf{1}_N \right)^T \mathrm{d}b \right) && \text{\small [Apply } \text{Tr}(M) = \text{Tr}(M^T) \text{]} \\ &= \text{Tr}\left( (\mathbf{1}_N^T \delta Y) \mathrm{d}b \right) && \text{\small [Expand transpose: } (\delta Y^T \mathbf{1}_N)^T = \mathbf{1}_N^T \delta Y \text{]} \end{aligned}$$

By identification against Identity I ($\mathrm{d}L = \text{Tr}\left( (\nabla_b L)^T \mathrm{d}b \right)$):

$${\nabla_b L = \delta Y^T \mathbf{1}_N = \sum_{i=1}^{N} \delta Y_{i, \cdot}} \quad \in \mathbb{R}^{d_{\text{out}}}$$

---
