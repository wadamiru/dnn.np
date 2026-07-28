# Derivations

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

### Linear: final

$$\nabla_X L = \delta Y W^T$$
$$\nabla_W L = X^T \delta Y$$
$$\nabla_b L = \delta Y^T \mathbf{1}_N$$

---

---

## Matrix Calculus Derivation: GELU Layer Gradients

### Mathematical Setup & Definitions

Consider an element-wise Gaussian Error Linear Unit mapping $f: \mathbb{R}^{N \times d} \to \mathbb{R}^{N \times d}$ defined by:

$$Y = \text{GELU}(X) = X \circ \Phi(X)$$

Where:

* $X \in \mathbb{R}^{N \times d}$ : Batch Input Matrix
* $Y \in \mathbb{R}^{N \times d}$ : Activated Output Matrix
* $\circ$ : Hadamard (Element-wise) Product Operator
* $\Phi(X)$ : Standard Gaussian Cumulative Distribution Function (CDF) evaluated element-wise
* $\mathbf{1}_{N \times d} \in \mathbb{R}^{N \times d}$ : Matrix of Ones with dimension $N \times d$
* $L \in \mathbb{R}$ : Scalar Loss Value

The exact CDF $\Phi(X)$ is expressed using the Error Function $\text{erf}(\cdot)$:

$$\Phi(X) = \frac{1}{2} \left( \mathbf{1}_{N \times d} + \text{erf}\left( \frac{X}{\sqrt{2}} \right) \right)$$

Let the upstream loss gradient with respect to $Y$ be defined as:

$$\delta Y \equiv \nabla_Y L = \frac{\partial L}{\partial Y} \in \mathbb{R}^{N \times d}$$

---

### Axioms & Matrix Calculus Identities

> **Identity I (Hadamard Commutativity & Duality inside Trace):**
> $$\text{Tr}\left( A^T (B \circ C) \right) = \text{Tr}\left( (A \circ B)^T C \right)$$
> 
> 
> *Proof:* $\sum_{i,j} A_{ij} (B_{ij} C_{ij}) = \sum_{i,j} (A_{ij} B_{ij}) C_{ij}$.

> **Identity II (Standard Gaussian Density & Error Function Derivative):**
> $$\frac{\mathrm{d}}{\mathrm{d}u} \text{erf}(u) = \frac{2}{\sqrt{\pi}} e^{-u^2} \implies \Phi'(x) = \frac{\mathrm{d}\Phi(x)}{\mathrm{d}x} = \frac{1}{\sqrt{2\pi}} e^{-\frac{x^2}{2}}$$
> 
> 

> **Identity III (Total Differential of Hadamard Product):**
> $$\mathrm{d}(A \circ B) = (\mathrm{d}A) \circ B + A \circ (\mathrm{d}B)$$
> 
> 

---

### Total Differential Expansion

Applying the total differential $\mathrm{d}(\cdot)$ to $Y = X \circ \Phi(X)$ using Identity III:

$$\mathrm{d}Y = (\mathrm{d}X) \circ \Phi(X) + X \circ \mathrm{d}\Phi(X)$$

Expressing the scalar loss differential $\mathrm{d}L$ using the Frobenius Inner Product definition:

$$\mathrm{d}L = \text{Tr}\left( \delta Y^T \mathrm{d}Y \right)$$

Substitute $\mathrm{d}Y$ into $\mathrm{d}L$ and expand by trace linearity:

$$\mathrm{d}L = \text{Tr}\left( \delta Y^T \left( (\mathrm{d}X) \circ \Phi(X) \right) \right) + \text{Tr}\left( \delta Y^T \left( X \circ \mathrm{d}\Phi(X) \right) \right)$$

Define the two path terms as:

$$\mathrm{d}L = \mathcal{T}_{\text{direct}} + \mathcal{T}_{\text{rate}}$$

---

### Derivation of Gradients

#### Direct Activation Pathway ($\mathcal{T}_{\text{direct}}$)

Apply Identity I to isolate $\mathrm{d}X$:

$$\mathcal{T}_{\text{direct}} = \text{Tr}\left( \delta Y^T \left( (\mathrm{d}X) \circ \Phi(X) \right) \right)$$

$$\mathcal{T}_{\text{direct}} = \text{Tr}\left( \left( \delta Y \circ \Phi(X) \right)^T \mathrm{d}X \right) \quad \text{(Hadamard trace duality)}$$

---

#### Rate-of-Change Pathway ($\mathcal{T}_{\text{rate}}$)

Apply Identity I to isolate $\mathrm{d}\Phi(X)$:

$$\mathcal{T}_{\text{rate}} = \text{Tr}\left( \delta Y^T \left( X \circ \mathrm{d}\Phi(X) \right) \right)$$

$$\mathcal{T}_{\text{rate}} = \text{Tr}\left( \left( \delta Y \circ X \right)^T \mathrm{d}\Phi(X) \right) \quad \text{(Hadamard trace duality)}$$

Express $\mathrm{d}\Phi(X)$ in terms of $\mathrm{d}X$ using Identity II:

$$\mathrm{d}\Phi(X) = \Phi'(X) \circ \mathrm{d}X = \left( \frac{1}{\sqrt{2\pi}} \exp\left( -\frac{X^{\circ 2}}{2} \right) \right) \circ \mathrm{d}X$$

Substitute $\mathrm{d}\Phi(X)$ back into $\mathcal{T}_{\text{rate}}$:

$$\mathcal{T}_{\text{rate}} = \text{Tr}\left( (\delta Y \circ X)^T \left[ \left( \frac{1}{\sqrt{2\pi}} \exp\left( -\frac{X^{\circ 2}}{2} \right) \right) \circ \mathrm{d}X \right] \right)$$

Apply Identity I a second time:

$$\mathcal{T}_{\text{rate}} = \text{Tr}\left( \left( \delta Y \circ \frac{X}{\sqrt{2\pi}} \circ \exp\left( -\frac{X^{\circ 2}}{2} \right) \right)^T \mathrm{d}X \right)$$

---

#### Consolidation of Terms ($\nabla_X L$)

Recombine $T_{\mathrm{direct}}$ and $T_{\mathrm{rate}}$ into $\mathrm{d}L$:

$$\mathrm{d}L = \text{Tr}\left( \left( \delta Y \circ \Phi(X) \right)^T \mathrm{d}X \right) + \text{Tr}\left( \left( \delta Y \circ \frac{X}{\sqrt{2\pi}} \circ \exp\left( -\frac{X^{\circ 2}}{2} \right) \right)^T \mathrm{d}X \right)$$

Factor out $\delta Y$ and $\mathrm{d}X$:

$$\mathrm{d}L = \text{Tr}\left( \left[ \delta Y \circ \left( \Phi(X) + \frac{X}{\sqrt{2\pi}} \circ \exp\left( -\frac{X^{\circ 2}}{2} \right) \right) \right]^T \mathrm{d}X \right)$$

By matching with $\mathrm{d}L = \text{Tr}\left( (\nabla_X L)^T \mathrm{d}X \right)$:

$$\nabla_X L = \delta Y \circ \left( \Phi(X) + \frac{X}{\sqrt{2\pi}} \circ \exp\left( -\frac{X^{\circ 2}}{2} \right) \right) \in \mathbb{R}^{N \times d}$$

---

### GELU: final

#### Exact Formulation

$$\nabla_X L = \delta Y \circ \left( \frac{1}{2} \left[ \mathbf{1}_{N \times d} + \text{erf}\left( \frac{X}{\sqrt{2}} \right) \right] + \frac{X}{\sqrt{2\pi}} \circ \exp\left( -\frac{X^{\circ 2}}{2} \right) \right)$$

#### Fast $\tanh$ Approximation

For $U = \sqrt{\frac{2}{\pi}} \left( X + 0.044715 X^{\circ 3} \right)$:

$$\nabla_X L \approx \delta Y \circ \left( 0.5 \left( \mathbf{1}_{N \times d} + \tanh(U) \right) + 0.5 X \circ \left( \mathbf{1}_{N \times d} - \tanh^{\circ 2}(U) \right) \circ \sqrt{\frac{2}{\pi}} \left( \mathbf{1}_{N \times d} + 0.134145 X^{\circ 2} \right) \right)$$

---
---

## Matrix Calculus Derivation: ReLU Layer Gradients

### Mathematical Setup & Definitions

Consider an element-wise Rectified Linear Unit (ReLU) mapping $f: \mathbb{R}^{N \times d} \to \mathbb{R}^{N \times d}$ defined by:

$$Y = \text{ReLU}(X) = \max(\mathbf{0}_{N \times d}, X) = X \circ H(X)$$

Where:

* $X \in \mathbb{R}^{N \times d}$ : Batch Input Matrix
* $Y \in \mathbb{R}^{N \times d}$ : Activated Output Matrix
* $\circ$ : Hadamard (Element-wise) Product Operator
* $\mathbf{0}_{N \times d} \in \mathbb{R}^{N \times d}$ : Matrix of Zeros with dimension $N \times d$
* $H(X) \in \{0, 1\}^{N \times d}$ : Element-wise Heaviside Step Indicator Matrix
* $L \in \mathbb{R}$ : Scalar Loss Value

The element-wise Heaviside step function $H(x)$ is defined as:

$$H(x) = \begin{cases} 1 & \text{if } x > 0 \\ 0 & \text{if } x < 0 \end{cases}$$

*(Note: At non-differentiable boundary points $x = 0$, a subgradient value is conventionally assigned, typically $H(0) = 0$ or $H(0) = 0.5$).*

Let the upstream loss gradient with respect to $Y$ be defined as:

$$\delta Y \equiv \nabla_Y L = \frac{\partial L}{\partial Y} \in \mathbb{R}^{N \times d}$$

---

### Axioms & Matrix Calculus Identities

> **Identity I (Hadamard Commutativity & Duality inside Trace):**
> $$\text{Tr}\left( A^T (B \circ C) \right) = \text{Tr}\left( (A \circ B)^T C \right)$$
> 
> 
> *Proof:* $\sum_{i,j} A_{ij} (B_{ij} C_{ij}) = \sum_{i,j} (A_{ij} B_{ij}) C_{ij}$.

> **Identity II (Subderivative of the ReLU Mapping):**
> $$\frac{\mathrm{d}}{\mathrm{d}x} \text{ReLU}(x) = H(x) = \mathbb{I}(x > 0)$$
> 
> 
> Where $\mathbb{I}(\cdot)$ is the indicator function returning $1$ when true and $0$ otherwise.

> **Identity III (Total Differential of Hadamard Product):**
> $$\mathrm{d}(A \circ B) = (\mathrm{d}A) \circ B + A \circ (\mathrm{d}B)$$
> 
> 

---

### Total Differential Expansion

Applying the total differential $\mathrm{d}(\cdot)$ to $Y = X \circ H(X)$ using Identity III:

$$\mathrm{d}Y = (\mathrm{d}X) \circ H(X) + X \circ \mathrm{d}H(X)$$

Because $H(X)$ is piecewise constant everywhere except at $X = 0$ (a set of measure zero), its differential vanishes almost everywhere ($\mathrm{d}H(X) = \mathbf{0}_{N \times d}$). The second pathway simplifies directly:

$$\mathrm{d}Y = (\mathrm{d}X) \circ H(X)$$

Expressing the scalar loss differential $\mathrm{d}L$ using the Frobenius Inner Product definition:

$$\mathrm{d}L = \text{Tr}\left( \delta Y^T \mathrm{d}Y \right)$$

Substitute $\mathrm{d}Y$ into $\mathrm{d}L$:

$$\mathrm{d}L = \text{Tr}\left( \delta Y^T \left( (\mathrm{d}X) \circ H(X) \right) \right)$$

---

### Derivation of Gradients

Apply Identity I to isolate $\mathrm{d}X$:

$$\mathrm{d}L = \text{Tr}\left( \left( \delta Y \circ H(X) \right)^T \mathrm{d}X \right) \quad \text{(Hadamard trace duality)}$$

Substitute the definition of $H(X) = \mathbb{I}(X > 0)$:

$$\mathrm{d}L = \text{Tr}\left( \left( \delta Y \circ \mathbb{I}(X > 0) \right)^T \mathrm{d}X \right)$$

By matching with $\mathrm{d}L = \text{Tr}\left( (\nabla_X L)^T \mathrm{d}X \right)$:

$$\nabla_X L = \delta Y \circ \mathbb{I}(X > 0) \in \mathbb{R}^{N \times d}$$

---

### ReLU: final

$$\nabla_X L = \delta Y \circ \mathbb{I}(X > 0) = \begin{cases} \delta Y_{ij} & \text{if } X_{ij} > 0 \\ 0 & \text{if } X_{ij} \le 0 \end{cases}$$

---
---

## Matrix Calculus Derivation: Softmax Cross-Entropy Gradients

### Mathematical Setup & Definitions

Consider a classification layer mapping pre-activation logits $Z \in \mathbb{R}^{N \times C}$ to class probabilities $P \in \mathbb{R}^{N \times C}$, evaluated against a target distribution matrix $Y \in \mathbb{R}^{N \times C}$:

$$P = \text{Softmax}(Z) = \left( \exp(Z) \mathbf{1}_C \right)^{-1}_{\text{diag}} \exp(Z)$$

Where:

* $Z \in \mathbb{R}^{N \times C}$ : Unnormalized Logit Matrix
* $Y \in \mathbb{R}^{N \times C}$ : One-Hot Ground Truth Matrix ($\sum_{j=1}^C Y_{ij} = 1$)
* $P \in \mathbb{R}^{N \times C}$ : Softmax Class Probability Matrix ($\sum_{j=1}^C P_{ij} = 1$)
* $\mathbf{1}_C \in \mathbb{R}^{C \times 1}$ : Column Vector of Ones $(1, 1, \dots, 1)^T$
* $\mathbf{1}_N \in \mathbb{R}^{N \times 1}$ : Column Vector of Ones $(1, 1, \dots, 1)^T$
* $L \in \mathbb{R}$ : Average Cross-Entropy Loss Value across $N$ samples

The mini-batch Cross-Entropy Loss $L$ is defined as:

$$L(P, Y) = -\frac{1}{N} \mathbf{1}_N^T \left( Y \circ \ln(P) \right) \mathbf{1}_C$$

---

### Axioms & Matrix Calculus Identities

> **Identity I (Row Normalization & Softmax Invariance):**
> $$P \mathbf{1}_C = \mathbf{1}_N \quad \text{and} \quad Y \mathbf{1}_C = \mathbf{1}_N$$
> 
> 

> **Identity II (Hadamard Commutativity & Duality inside Trace):**
> $$\text{Tr}\left( A^T (B \circ C) \right) = \text{Tr}\left( (A \circ B)^T C \right)$$
> 
> 

> **Identity III (Differential of Softmax Row Vectors):**
> For a single row logit vector $z_i \in \mathbb{R}^{1 \times C}$ and probability vector $p_i = \text{Softmax}(z_i)$:
> $$\mathrm{d}p_i = \mathrm{d}z_i \left( \operatorname{diag}(p_i) - p_i^T p_i \right)$$
> 
> 

---

### Total Differential Expansion

Applying the total differential $\mathrm{d}(\cdot)$ directly to the scalar loss $L(P, Y)$:

$$\mathrm{d}L = -\frac{1}{N} \text{Tr}\left( \mathbf{1}_C \mathbf{1}_N^T \mathrm{d}\left( Y \circ \ln(P) \right) \right)$$

$$\mathrm{d}L = -\frac{1}{N} \text{Tr}\left( \mathbf{1}_N \mathbf{1}_C^T \left( Y \circ (P^{\circ -1} \circ \mathrm{d}P) \right)^T \right) \quad \text{(Trace transpose)}$$

Applying Identity II (Hadamard duality) to isolate $\mathrm{d}P$:

$$\mathrm{d}L = -\frac{1}{N} \text{Tr}\left( \left[ \left( \mathbf{1}_N \mathbf{1}_C^T \right) \circ Y \circ P^{\circ -1} \right]^T \mathrm{d}P \right)$$

Since $\mathbf{1}_N \mathbf{1}_C^T$ is a matrix of all ones, this simplifies to:

$$\mathrm{d}L = -\frac{1}{N} \text{Tr}\left( \left( Y \circ P^{\circ -1} \right)^T \mathrm{d}P \right)$$

---

### Derivation of Gradients

#### Step-by-Step Row Differential Evaluation

To evaluate $\mathrm{d}L$ in terms of $\mathrm{d}Z$, expand row-wise for sample $i \in \{1, \dots, N\}$:

$$\mathrm{d}L = -\frac{1}{N} \sum_{i=1}^N \mathrm{d}p_i \left( y_i \circ p_i^{\circ -1} \right)^T$$

Substitute the row softmax differential from Identity III ($\mathrm{d}p_i = \mathrm{d}z_i \left( \operatorname{diag}(p_i) - p_i^T p_i \right)$):

$$\mathrm{d}L = -\frac{1}{N} \sum_{i=1}^N \mathrm{d}z_i \left( \operatorname{diag}(p_i) - p_i^T p_i \right) \left( y_i \circ p_i^{\circ -1} \right)^T$$

Expand the matrix-vector multiplication:

$$\left( \operatorname{diag}(p_i) - p_i^T p_i \right) \left( y_i \circ p_i^{\circ -1} \right)^T = \operatorname{diag}(p_i) \left( y_i \circ p_i^{\circ -1} \right)^T - p_i^T p_i \left( y_i \circ p_i^{\circ -1} \right)^T$$

#### Term 1: $\operatorname{diag}(p_i) \left( y_i \circ p_i^{\circ -1} \right)^T$

$$\operatorname{diag}(p_i) \left( \frac{y_i}{p_i} \right)^T = p_i^T \circ \left( \frac{y_i}{p_i} \right)^T = y_i^T$$

#### Term 2: $p_i^T p_i \left( y_i \circ p_i^{\circ -1} \right)^T$

$$p_i \left( y_i \circ p_i^{\circ -1} \right)^T = \sum_{j=1}^C p_{ij} \frac{y_{ij}}{p_{ij}} = \sum_{j=1}^C y_{ij} = 1 \quad \text{(using Identity I)}$$

Thus, $p_i^T p_i \left( y_i \circ p_i^{\circ -1} \right)^T = p_i^T (1) = p_i^T$.

---

#### Matrix Re-consolidation ($\nabla_Z L$)

Substituting Term 1 and Term 2 back into the row expansion:

$$\mathrm{d}L = -\frac{1}{N} \sum_{i=1}^N \mathrm{d}z_i \left( y_i^T - p_i^T \right) = \frac{1}{N} \sum_{i=1}^N \mathrm{d}z_i \left( p_i - y_i \right)^T$$

Re-assembling the individual row vectors back into full mini-batch matrices $Z, P, Y \in \mathbb{R}^{N \times C}$ via trace form:

$$\mathrm{d}L = \frac{1}{N} \text{Tr}\left( (P - Y)^T \mathrm{d}Z \right)$$

By matching with $\mathrm{d}L = \text{Tr}\left( (\nabla_Z L)^T \mathrm{d}Z \right)$:

$$\nabla_Z L = \frac{1}{N} (P - Y) \in \mathbb{R}^{N \times C}$$

---

### Softmax Cross-Entropy: final

#### Unaveraged Loss Gradient

$$\nabla_Z L_{\text{sum}} = P - Y$$

#### Average Mini-Batch Loss Gradient

$$\nabla_Z L = \frac{1}{N} (P - Y)$$