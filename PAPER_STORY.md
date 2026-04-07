# Token-Guard: Algorithm Specification

This document outlines the core mathematical concepts, variables, and algorithms for the Token-Guard framework. The implementation must strictly follow these specifications to ensure algorithmic fidelity.

## Stage 1: Token-Level Hallucination Self-Checking
- **Core Component:** `LatentEnvironment`
  - Purpose: Manages semantic representations ($s_j$) and contextual states ($h_j$).
  - Initialization: Implement `initialize_initial_anchor` to compute $h_x$ (the mean hidden state of the input context).
- **Candidate Token Set:** $\mathcal{A}_t = \{a_t^{(1)}, ..., a_t^{(M)}\}$.
- **Scoring Logic:** Implement `hybrid_token_level_scoring`.
  - Hyperparameter: $\lambda = 0.6$ (Weights the semantic consistency vs. token probability).
- **Selection Threshold:** Token propagation requires passing $\tau_{\text{token}} = 0.4$.
- **Memory Management:** Implement temporary buffering bounded by $\mathcal{O}(L_{\text{max}} \cdot d)$.

## Stage 2: Candidate Segment Representation
- **Core Component:** `CandidateSegment` (Representing sequence $C_k$ with aggregated vector $H_k$).
- **Metrics Implementation:**
  1. `calculate_weighted_token_score`: Aggregates token reliability using softmax weights $w_i$.
  2. `calculate_local_consistency`: Computes semantic transition smoothness.
  3. `calculate_global_alignment`: Computes cosine similarity against the input context.
- **Segment Scoring:** Implement `segment_level_hallucination_score`.
  - Hyperparameters: $\alpha = 0.5$, $\beta = 0.3$, $\gamma = 0.2$.
- **Local Refinement Mechanism:**
  - Logic: Identify the lowest-scoring token, form a local window $W_k^{(l)}$, and refine.
  - Thresholds: $\tau_{\text{seg}}^{\text{low}} = 0.55$, $\tau_{\text{seg}}^{\text{high}} = 0.75$.
  - Iteration Limit: Max refinement steps $N_{\text{max}} = 3$.

## Stage 3: Global Iteration and Correction
- **Core Component:** Global Reasoning Chains ($R$).
- **Clustering:** Implement `cluster_and_select_chains` utilizing TF-IDF and KMeans.
  - Parameter: $K = 5$ for large datasets, $K = 3$ for small datasets.
- **Global Metrics:**
  1. `calculate_factual_consistency`: Computes $F_{\text{fact}}(R)$.
  2. `calculate_logical_coherence`: Computes $F_{\text{logic}}(R)$, integrating contextual factor $\lambda_k$ and $\text{sim}_{\text{ctx}}$.
- **Global Scoring:** Implement `global_iteration_correction` using a soft minimum combination of factual and logical scores.
- **Dynamic Thresholding:**
  - Base Threshold: $\tau_{\text{global}} = 0.7$.
  - Adjustment Margin: $\Delta\tau = 0.1$ (Dynamically adjust segment thresholds if Fact/Logic scores diverge).
  - Iteration Limit: Max global iterations $M_{\text{max}} = 2$.
  - Fallback: Output "cannot answer" if convergence fails.