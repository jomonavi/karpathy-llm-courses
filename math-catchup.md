# Fast-Track Math Guide: Precalculus to Backpropagation
*A streamlined curriculum designed for engineering professionals refreshing math for Machine Learning and LLMs.*

---

## Phase 1: Targeted Precalculus Prep
*Do not take a full course. Navigate to Khan Academy's main **Precalculus** course and cherry-pick these exact units:*

### 1. Unit 1: Composite and Inverse Functions
* **Why it matters:** A neural network is a massive chain of nested functions: $f(g(h(x)))$. 
* **Core Concepts:** Focus on how the output of one layer (like a matrix multiplication) becomes the direct input for the next layer (like an activation function).

### 2. Unit 6: Vectors
* **Why it matters:** Features, embeddings, and biases are vectors. 
* **Core Concepts:** Learn vector addition, scalar multiplication, and especially the **dot product**, which handles how weights are combined with inputs.

### 3. Unit 7: Matrices
* **Why it matters:** Weights in neural networks are stored as large grids (matrices). Training a model means multiplying these grids by data vectors.
* **Core Concepts:** Focus heavily on **matrix multiplication**. 
* **What to skip:** You can completely ignore inverse matrices, determinants, and matrices as systems of equations for now.

### 4. Unit 10: Limits and Continuity
* **Why it matters:** Derivatives are mathematically built using limits. You must understand how a function behaves as it gets infinitely close to a specific point.
* **Core Concepts:** Understand limits graphically and practice basic algebraic evaluation.

---

## Phase 2: Core Calculus (The Machinery of Learning)
*Move immediately to **AP® Calculus AB** or **Calculus 1** once Phase 1 is done. You only need the first half of the course.*

### 1. Unit 2: Defining Derivatives
* **Why it matters:** The derivative is the instantaneous rate of change (the slope). In ML, the derivative tells you how changing a single weight affects the model's total error.
* **Core Concepts:** Tangent lines, secant lines, and formalizing a derivative using limits.

### 2. Unit 3: Differentiation: Composite, Implicit, and Inverse Functions
* **Why it matters:** This contains the **Chain Rule**, which is the exact mathematical engine of backpropagation. 
* **Core Concepts:** Mastering how to take the derivative of composite functions: 
  $$\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}$$
* **ML Application:** Backpropagation uses the Chain Rule to pass the error rate backward from the final output layer all the way to the very first weight in the network.

---

## Phase 3: Moving to Multivariable Space
*Because neural networks have millions of parameters, you must eventually jump to Khan Academy's **Multivariable Calculus** course.*

### 1. Unit 2: Multivariable Derivatives
* **Why it matters:** Standard calculus deals with one input ($x$). Neural networks deal with millions of weights. A partial derivative isolates just *one* weight, treating all other weights as frozen constants.
* **Core Concepts:** 
  * **Partial Derivatives:** Taking derivatives with respect to one variable at a time.
  * **The Gradient:** A vector pack of all your partial derivatives. It points in the direction of the steepest increase in error.
  * **Multivariable Chain Rule:** Combining the chain rule with partial derivatives to calculate network error over multiple paths simultaneously.


# ML/LLM Full Math Study Path: Limits, Derivatives & Beyond

A focused Khan Academy curriculum to understand backpropagation, then
probability-based ML (loss functions, KL divergence, generative models).


---

## Table of Contents

- [The Big Picture](#the-big-picture)
- [Phase 1 — Backpropagation Foundation](#phase-1--backpropagation-foundation)
  - [Course 1: Calculus 1 (or AP Calculus BC)](#course-1-calculus-1-or-ap-calculus-bc)
  - [Course 2: Multivariable Calculus](#course-2-multivariable-calculus)
  - [Course 3: Linear Algebra](#course-3-linear-algebra)
- [Phase 2 — Probability & Holistic ML](#phase-2--probability--holistic-ml)
  - [Course 4: Calculus 2](#course-4-calculus-2)
  - [Course 5: Statistics and Probability](#course-5-statistics-and-probability)
- [Master Checklist](#master-checklist)
- [Time Estimates](#time-estimates)
- [The Key Mental Model](#the-key-mental-model)
- [What Each Phase Unlocks](#what-each-phase-unlocks)

---

## The Big Picture

```
┌─────────────────────────────────────────────────────┐
│  PHASE 1: Backpropagation Foundation (derivative)   │
├─────────────────────────────────────────────────────┤
│  Calculus 1 / AP Calc BC                            │
│    Unit 1: Limits and continuity                    │
│    Unit 2: Derivatives: definition & basic rules    │
│    Unit 3: Derivatives: chain rule                  │
│                                                     │
│  Multivariable Calculus                             │
│    Unit 2: Partial derivatives, gradients,          │
│            multivariable chain rule                 │
│                                                     │
│  Linear Algebra                                     │
│    Unit 1: Vectors and spaces                       │
│    Unit 2: Matrix transformations                   │
│                                                     │
│  ✅ At this point: You understand backpropagation   │
└─────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│  PHASE 2: Probability & Holistic ML (integrals)     │
├─────────────────────────────────────────────────────┤
│  Calculus 2                                         │
│    Unit 1: Integrals review ← THE KEY UNIT          │
│    Unit 2: Integration techniques (skim)            │
│    Unit 4: Applications of integrals                │
│      [skip Units 3, 5, 6]                           │
│                                                     │
│  Statistics and Probability                         │
│    Probability + Continuous random variables        │
│    Normal distribution                              │
│    Sampling distributions                           │
│                                                     │
│  ✅ At this point: You understand loss functions    │
│     probabilistically, KL divergence, entropy,      │
│     and the foundations of generative models        │
└─────────────────────────────────────────────────────┘
```


---

## Phase 1 — Backpropagation Foundation

**Goal:** Master limits (intuitively), derivatives, the chain rule,
partial derivatives, gradients, and the matrix/vector language of
neural networks.

**Outcome:** You can derive and explain how gradients flow backward
through a neural network.

---

### Course 1: Calculus 1 (or AP Calculus BC)

> **Pick ONE course shell** — they contain the same videos.
> - [Calculus 1](https://www.khanacademy.org/math/calculus-1) — more focused
> - [AP Calculus BC](https://www.khanacademy.org/math/ap-calculus-bc) — most comprehensive & up-to-date
>
> Recommended: **Calculus 1**, since you only need derivatives, not integrals.

| Done | Unit | Topic | Why It Matters |
|:----:|------|-------|----------------|
| [ ] | Unit 1 | **Limits and continuity** | Need the *intuitive* idea of a limit to understand the derivative definition. Focus on: what limits mean, estimating from graphs, limit properties. **Skim epsilon-delta and continuity proofs.** |
| [ ] | Unit 2 | **Derivatives: definition and basic rules** | The core. Derivative as instantaneous rate of change & slope of tangent line. Power rule, product rule, quotient rule. |
| [ ] | Unit 3 | **Derivatives: chain rule & advanced topics** | ⭐ **MOST CRITICAL UNIT.** The chain rule is the engine of backpropagation. Also need derivatives of $e^x$, $\ln(x)$, and the sigmoid function. |

**⏭️ Skip:**
- Units 4+ (applications of derivatives: related rates, optimization)
- All integral calculus (that's Phase 2)

---

### Course 2: Multivariable Calculus

> [Multivariable Calculus](https://www.khanacademy.org/math/multivariable-calculus)

| Done | Unit | Topic | Why It Matters |
|:----:|------|-------|----------------|
| [ ] | Unit 2 | **Derivatives of multivariable functions** | The heart of what you need. Contains partial derivatives, the gradient vector, directional derivatives, and the **multivariable chain rule**. |

**Focus on these lessons within Unit 2:**

- [ ] **Partial derivatives** — how a function changes with respect to one
      variable while holding others constant
- [ ] **The gradient vector** — all partial derivatives collected into a
      vector, pointing in the direction of steepest ascent
- [ ] **Multivariable chain rule** — generalizes the chain rule to multiple
      variables; this is *literally* what backpropagation computes

**⏭️ Skip:**
- Unit 1 (vectors & multivariable functions intro — covered by Linear Algebra)
- Units 3+ (multiple integrals, surface integrals, Stokes' theorem)

---

### Course 3: Linear Algebra

> [Linear Algebra](https://www.khanacademy.org/math/linear-algebra)

| Done | Unit | Topic | Why It Matters |
|:----:|------|-------|----------------|
| [ ] | Unit 1 | **Vectors and spaces** | Vectors, dot products, linear combinations, span. NN inputs/outputs/activations are all vectors. |
| [ ] | Unit 2 | **Matrix transformations** | Matrix multiplication, matrix-vector products, linear transformations. Every NN layer is a matrix multiply + nonlinear activation. |

**⏭️ Skip:**
- Unit 3 (alternate coordinate systems: eigenvalues, eigenvectors) —
  useful for PCA and advanced ML, not needed for backprop.

---

## Phase 2 — Probability & Holistic ML

**Goal:** Understand *why* ML loss functions work, what probability
distributions are doing, and how generative models are built.

**Outcome:** You can read about cross-entropy, KL divergence, entropy,
and VAEs/diffusion models and understand the math.

---

### Course 4: Calculus 2

> [Calculus 2](https://www.khanacademy.org/math/calculus-2)

#### Tier 1 — Study These (Directly Relevant to ML)

| Done | Unit | Topic | Why It Matters |
|:----:|------|-------|----------------|
| [ ] | Unit 1 | **Integrals review** | ⭐ **THE KEY UNIT.** What an integral is, definite vs. indefinite, the Fundamental Theorem of Calculus, accumulation functions. |
| [ ] | Unit 2 | **Integration techniques** | Substitution (reverse chain rule) & integration by parts. Focus on substitution; skim the rest. |

#### Tier 2 — Useful Context (Study If You Have Time)

| Done | Unit | Topic | Why It Matters |
|:----:|------|-------|----------------|
| [ ] | Unit 3 | **Differential equations** | Systems evolving over time. Relevant to neural ODEs & optimization dynamics. |
| [ ] | Unit 4 | **Applications of integrals** | Area under curves, accumulated change. Relevant to ROC/AUC and probability as area under a density. |

#### Tier 3 — Skip for ML

| Done | Unit | Topic | Why You Can Skip |
|:----:|------|-------|------------------|
| [ ] | Unit 5 | **Parametric, polar & vector-valued functions** | Mostly physics/geometry; not directly relevant to ML. |
| [ ] | Unit 6 | **Series** (convergence, Taylor, Maclaurin) | Beautiful & appears in some theory, but not needed for practical ML. |

---

### Course 5: Statistics and Probability

> [Statistics and Probability](https://www.khanacademy.org/math/statistics-probability)
>
> **This is the real payoff of integrals for ML.** Calculus 2 alone
> won't fill the gap — you need probability.

| Done | Topic | Why It Matters |
|:----:|-------|----------------|
| [ ] | **Probability** — random variables, probability distributions | Foundation for all probabilistic ML |
| [ ] | **Continuous random variables** | ⭐ Where integrals & probability merge: $P(a \leq X \leq b) = \int_a^b f(x)\,dx$ |
| [ ] | **Normal distribution** | The most important distribution in ML (weight init, VAEs, diffusion, noise models) |
| [ ] | **Sampling distributions** | How models generalize; basis for confidence intervals & hypothesis testing |

---

## Master Checklist

### Phase 1 — Backpropagation Foundation

- [x] **Calculus 1** — Unit 1: Limits and continuity (intuitive only)
- [x] **Calculus 1** — Unit 2: Derivatives: definition and basic rules
- [x] **Calculus 1** — Unit 3: Chain rule & advanced topics ⭐
- [x] **Multivariable Calculus** — Unit 2: Partial derivatives
- [x] **Multivariable Calculus** — Unit 2: Gradient vector
- [x] **Multivariable Calculus** — Unit 2: Multivariable chain rule ⭐
- [x] **Linear Algebra** — Unit 1: Vectors and spaces
- [ ] **Linear Algebra** — Unit 2: Matrix transformations

> ✅ **Checkpoint:** You understand backpropagation.

### Phase 2 — Probability & Holistic ML

- [x] **Calculus 2** — Unit 1: Integrals review ⭐
- [x] **Calculus 2** — Unit 2: Integration techniques (substitution focus)
- [ ] **Calculus 2** — Unit 4: Applications of integrals
- [ ] **Statistics** — Probability & random variables
- [ ] **Statistics** — Continuous random variables ⭐
- [ ] **Statistics** — Normal distribution
- [ ] **Statistics** — Sampling distributions

> ✅ **Checkpoint:** You understand loss functions probabilistically,
> KL divergence, entropy, and the foundations of generative models.

### Optional / Advanced

- [x] **Calculus 2** — Unit 3: Differential equations (neural ODEs)
- [ ] **Linear Algebra** — Unit 3: Eigenvalues & eigenvectors (PCA)
- [ ] **Calculus 2** — Unit 6: Taylor series (optimization theory)

---

## Time Estimates

| Phase | Component | Estimated Hours |
|-------|-----------|:---------------:|
| **1** | Calculus 1 (Units 1–3) | 15–25 hrs |
| **1** | Multivariable Calculus (Unit 2) | 10–15 hrs |
| **1** | Linear Algebra (Units 1–2) | 15–20 hrs |
| **1** | **Phase 1 Subtotal** | **~40–60 hrs** |
| **2** | Calculus 2 (Units 1, 2, 4) | 15–20 hrs |
| **2** | Statistics and Probability (focus units) | 15–20 hrs |
| **2** | **Phase 2 Subtotal** | **~30–40 hrs** |
| | **Grand Total** | **~70–100 hrs** |

**Pacing:**
- At 10 hrs/week → Phase 1 in ~4–6 weeks, Phase 2 in ~3–4 weeks
- At 5 hrs/week → Phase 1 in ~8–12 weeks, Phase 2 in ~6–8 weeks

---

## The Key Mental Model

Keep this framing in mind throughout — it connects everything:

> A neural network is a **giant composite function**:
>
> $$\text{output} = f_L(f_{L-1}(\cdots f_2(f_1(\text{input}))\cdots))$$
>
> where each $f_i$ is a layer (matrix multiply + activation). The loss $C$
> measures how wrong the output is.
>
> **Backpropagation uses the multivariable chain rule** to compute
> $\frac{\partial C}{\partial w}$ for every weight $w$ in every layer,
> working backward from output to input.
>
> **Gradient descent** then uses those partial derivatives (collected into
> a **gradient vector**) to update all weights in the direction that
> reduces the loss.

Every piece of Phase 1 maps directly to one part of that paragraph.
Once you can read it and understand every mathematical term, you have
what you need.

---

## What Each Phase Unlocks

| If your goal is... | You need... | Integrals? |
|--------------------|-------------|:----------:|
| Understand how backpropagation works mechanically | Phase 1 only | ❌ No |
| Understand *why* cross-entropy is the right loss for LLMs | Phase 1 + basic probability + integral concept | ✅ Yes |
| Understand generative models (VAEs, diffusion) | Phase 1 + Phase 2 (full) | ✅ Yes |
| Read and understand ML research papers | Phase 1 + Phase 2 (full) | ✅ Yes |
| Use existing frameworks (PyTorch, TensorFlow) practically | Phase 1 is sufficient | ❌ No |

---

## Why the "Skipped" Content Doesn't Hurt You

- **Applications of derivatives (Phase 1, Unit 4):** related rates and
  optimization are great practice, but backprop needs the *mechanics* of
  differentiation, not physics-style word problems.
- **Integral calculus in Phase 1:** backprop is pure differentiation. No
  integrals appear in the gradient computation itself.
- **Series (Phase 2, Unit 6):** Taylor series show up in convergence
  analysis of optimizers, but you can use Adam/SGD perfectly well without
  ever expanding a series.

Come back to any skipped unit later *if and when* a specific topic demands it.

---

*Last updated: 2026-10-12*
