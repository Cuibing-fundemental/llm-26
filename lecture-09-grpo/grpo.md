# Group Relative Policy Optimization (GRPO)

> **Source:** DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (Shao et al., 2024)

---

## 1. Why Not Just Use PPO?

Proximal Policy Optimization (PPO) requires training **two models simultaneously**:

- **Actor** $\pi_\theta$ — the policy we are optimizing
- **Critic** $V_\phi$ — a value network that estimates the expected return from each state

The critic adds substantial memory and compute cost, especially for large LLMs. It must match the size of the actor to be accurate, effectively doubling the training footprint.

**GRPO removes the critic entirely.** Instead it estimates advantages by comparing outputs within a sampled group.

---

## 2. Setup: Group Sampling

For each prompt $q$ drawn from the training distribution $P(Q)$, GRPO samples a **group** of $G$ independent outputs from the current (old) policy:

$$\{o_1, o_2, \ldots, o_G\} \sim \pi_{\theta_{\text{old}}}(\cdot \mid q)$$

Each output $o_i$ is a complete response (a sequence of tokens). A reward function $\mathcal{R}$ then scores each one:

$$r_i = \mathcal{R}(q, o_i), \quad i = 1, \ldots, G$$

So we get a group of reward signals $\mathbf{r} = (r_1, r_2, \ldots, r_G)$ — all from the **same prompt**, all from the **same old policy**. This is the key observation that makes GRPO work.

---

## 3. Group-Relative Advantage

In PPO, the advantage $\hat{A}_t$ at token $t$ is estimated by the critic:

$$\hat{A}_t^{\text{PPO}} = r_t + \gamma V(s_{t+1}) - V(s_t)$$

GRPO has no critic. Instead it uses the **group mean and standard deviation** of rewards as the baseline:

$$\boxed{\hat{A}_i = \frac{r_i - \mu_{\mathbf{r}}}{\sigma_{\mathbf{r}}}}$$

where

$$\mu_{\mathbf{r}} = \frac{1}{G}\sum_{i=1}^G r_i, \qquad \sigma_{\mathbf{r}} = \sqrt{\frac{1}{G}\sum_{i=1}^G (r_i - \mu_{\mathbf{r}})^2}$$

**Intuition:**
- If output $o_i$ got a reward **above** the group average → $\hat{A}_i > 0$ → encourage this output
- If output $o_i$ got a reward **below** the group average → $\hat{A}_i < 0$ → suppress this output
- The std normalization keeps the advantage signal on a consistent scale regardless of the reward magnitude

Note: $\hat{A}_i$ is a **scalar per output** — it is broadcast to every token $t$ in $o_i$, so $\hat{A}_{i,t} = \hat{A}_i$ for all $t$.

---

## 4. Importance Ratio

Because GRPO is an **off-policy** update (outputs were sampled from $\pi_{\theta_{\text{old}}}$ but we are updating $\pi_\theta$), we correct for the distribution shift with an importance ratio at each token $t$:

$$\boxed{\rho_{i,t}(\theta) = \frac{\pi_\theta(o_{i,t} \mid q,\, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t} \mid q,\, o_{i,<t})}}$$

- When $\pi_\theta \approx \pi_{\theta_{\text{old}}}$, the ratio $\rho \approx 1$ and there is no correction
- When $\pi_\theta$ has moved far from $\pi_{\theta_{\text{old}}}$, the ratio can be large or small, which is dangerous

---

## 5. Clipped Surrogate Objective

To prevent large policy updates, GRPO inherits PPO's **clipping** trick. The per-token surrogate loss is:

$$\boxed{L_{\text{clip}}(i, t, \theta) = \min\!\Big(\rho_{i,t}\,\hat{A}_i,\quad \text{clip}(\rho_{i,t},\; 1-\varepsilon,\; 1+\varepsilon)\,\hat{A}_i\Big)}$$

where $\varepsilon$ is a small hyperparameter (typically $0.1$ or $0.2$).

**Why does clipping work?**

Consider the two cases:

**Case 1 — $\hat{A}_i > 0$ (good output, want to increase probability):**

$$\min(\rho_{i,t}\,\hat{A}_i,\; (1+\varepsilon)\,\hat{A}_i)$$

If $\rho_{i,t} > 1+\varepsilon$ (policy has already increased this token's probability too much), the clip kicks in and stops further increase. The gradient becomes zero.

**Case 2 — $\hat{A}_i < 0$ (bad output, want to decrease probability):**

$$\min(\rho_{i,t}\,\hat{A}_i,\; (1-\varepsilon)\,\hat{A}_i)$$

If $\rho_{i,t} < 1-\varepsilon$ (policy has already decreased this token's probability too much), the clip stops further decrease.

In both cases, the objective is **pessimistic** — it takes the lower bound, acting as a conservative constraint on how much the policy can change in one step.

---

## 6. KL Divergence Penalty

To further prevent the updated policy $\pi_\theta$ from drifting too far from the **reference policy** $\pi_{\text{ref}}$ (usually the SFT model), GRPO adds a KL penalty. The KL divergence is:

$$D_{\text{KL}}[\pi_\theta \| \pi_{\text{ref}}] = \mathbb{E}\left[\log \frac{\pi_\theta(o)}{\pi_{\text{ref}}(o)}\right]$$

In practice this is estimated per token using the following **unbiased estimator** (Schulman, 2020):

$$\boxed{D_{\text{KL}}[\pi_\theta \| \pi_{\text{ref}}]_{i,t} \approx \frac{\pi_{\text{ref}}(o_{i,t} \mid q, o_{i,<t})}{\pi_\theta(o_{i,t} \mid q, o_{i,<t})} - \log\frac{\pi_{\text{ref}}(o_{i,t} \mid q, o_{i,<t})}{\pi_\theta(o_{i,t} \mid q, o_{i,<t})} - 1}$$

This estimator is always $\geq 0$ and equals $0$ only when $\pi_\theta = \pi_{\text{ref}}$.

The KL term is weighted by $\beta > 0$, a hyperparameter controlling how tightly the policy is anchored to the reference model.

---

## 7. The Full GRPO Objective

Putting it all together, the GRPO objective to **maximize** is:

$$\boxed{J_{\text{GRPO}}(\theta) = \mathbb{E}_{\substack{q \sim P(Q) \\ \{o_i\}_{i=1}^G \sim \pi_{\theta_{\text{old}}}(\cdot \mid q)}} \left[ \frac{1}{G} \sum_{i=1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \left( L_{\text{clip}}(i,t,\theta) - \beta\, D_{\text{KL}}[\pi_\theta \| \pi_{\text{ref}}]_{i,t} \right) \right]}$$

Breaking it down:

| Term | Meaning |
|---|---|
| $\frac{1}{G}\sum_{i=1}^G$ | average over the group of $G$ sampled outputs |
| $\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}$ | average over all tokens in output $o_i$ |
| $L_{\text{clip}}(i,t,\theta)$ | clipped policy gradient term (maximize) |
| $\beta\, D_{\text{KL}}[\pi_\theta \| \pi_{\text{ref}}]_{i,t}$ | KL penalty anchoring to reference model (minimize) |

---

## 8. The GRPO Algorithm

```
Algorithm: GRPO

Input:  initial policy π_θ (= SFT model), reference policy π_ref,
        reward function R, group size G, iterations T

for iteration = 1 to T:
    1. Sample a batch of prompts {q} from P(Q)
    
    2. For each prompt q:
       - Sample G outputs {o_1, ..., o_G} from π_θ_old(· | q)
       - Compute rewards r_i = R(q, o_i) for i = 1..G
       - Compute group-relative advantages:
             Â_i = (r_i - mean(r)) / std(r)
    
    3. Update θ by maximizing J_GRPO(θ):
       - Compute ρ_{i,t} = π_θ(o_{i,t}|q,o_{i,<t}) / π_θ_old(o_{i,t}|q,o_{i,<t})
       - Compute clipped surrogate L_clip
       - Compute KL penalty w.r.t. π_ref
       - Gradient step on J_GRPO

    4. Update θ_old ← θ

Output: optimized policy π_θ
```

---

## 9. GRPO vs PPO: Key Differences

| | PPO | GRPO |
|---|---|---|
| Advantage estimation | Critic network $V_\phi$ | Group mean/std normalization |
| Extra model required | Yes (critic, same size as actor) | No |
| Memory cost | ~2× actor size | ~1× actor size |
| Baseline | State value $V(s_t)$ | Group mean reward $\mu_\mathbf{r}$ |
| Token-level signal | Yes (via GAE) | Uniform within output (same $\hat{A}_i$ for all tokens) |
| Clipping | ✅ | ✅ |
| KL penalty | Optional | Standard |

The main **limitation** of GRPO is that every token in the same output gets the same advantage signal $\hat{A}_i$ — there is no credit assignment at the token level within a single response. PPO with a good critic can assign different advantages to different tokens, giving a finer-grained learning signal.

---

## 10. Why It Works for Math / Reasoning

GRPO is especially effective when the reward signal is **binary or sparse** (e.g., correct/incorrect on a math problem):

- With $G$ samples per prompt, even if most outputs are wrong, a correct one will have $r_i \gg \mu_\mathbf{r}$ and get a large positive advantage
- The model is pushed toward the reasoning pattern that produced the correct answer
- No credit assignment problem within the chain-of-thought — getting the right final answer reinforces the entire reasoning trace

This matches perfectly with the **outcome reward model** (ORM) used in DeepSeekMath: reward 1 if the final answer is correct, 0 otherwise.

---

## References

1. Shao et al. *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models.* 2024. [`arXiv:2402.03300`](https://arxiv.org/abs/2402.03300)
2. Schulman et al. *Proximal Policy Optimization Algorithms.* 2017. [`arXiv:1707.06347`](https://arxiv.org/abs/1707.06347)
3. DeepSeek-AI. *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning.* 2025. [`arXiv:2501.12948`](https://arxiv.org/abs/2501.12948)
