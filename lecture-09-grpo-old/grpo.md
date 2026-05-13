# Group Relative Policy Optimization (GRPO)

> **Primary reference:** [GRPO-Zero](https://github.com/policy-gradient/GRPO-Zero) — a minimal from-scratch implementation (only `torch` + `tokenizers`, no HuggingFace Transformers, no vLLM).  
> **Original paper:** DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (Shao et al., 2024)  
> **Variant:** R1-Zero — pure RL from a pretrained model, **no SFT warm-up, no KL penalty, no reference model.**

---

## 1. The Big Picture

Standard PPO fine-tunes an LLM with RL by training **two models simultaneously**:

- **Actor** $\pi_\theta$ — the policy being optimized  
- **Critic** $V_\phi$ — a value network that estimates expected return, roughly the same size as the actor

The critic doubles memory cost. **GRPO removes it entirely** by using the spread of rewards within a sampled group as the baseline.

The full training loop is three steps repeated indefinitely:

```
Loop:
  1. ROLLOUT  — sample G responses per prompt, score each with a reward function
  2. NORMALIZE — compute group-relative advantage for each response
  3. UPDATE   — take one gradient step on the policy using the clipped PG objective
```

---

## 2. Step 1 — Rollout

### What it does

For each prompt $q$, generate $G$ independent responses using the current policy $\pi_\theta$:

$$\{o_1, o_2, \ldots, o_G\} \sim \pi_\theta(\cdot \mid q)$$

Then score each response with a reward function $\mathcal{R}$:

$$r_i = \mathcal{R}(q, o_i), \quad i = 1, \ldots, G$$

### The code (from `grpo.py`)

```python
@torch.no_grad()
def rollout(model, batch, tokenizer, max_gen_len,
            num_answer_per_question, reward_function, device, dtype):
```

Key implementation details:

**KV-cache for efficiency.** Generation is token-by-token. GRPO-Zero pre-allocates a KV cache for all $G$ sequences at once so each forward pass only processes one new token (not the full sequence):

```python
model.init_kv_cache(max_batch_size=bsz, max_seq_len=total_len, ...)

for cur_pos in range(min_prompt_len, total_len):
    logits = model.inference(tokens[:, prev_pos:cur_pos], prev_pos)
    probs  = torch.softmax(logits[:, -1], dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    ...
```

**Early stopping.** Once all sequences have emitted an `<eos>` token, generation stops:

```python
is_finished = is_finished | (is_end_token & is_generated_token)
if is_finished.all():
    break
```

**Reward assignment.** After generation, each completed response is scored:

```python
rewards = reward_function(response=generated_text, numbers=..., target=...)
episode = Episode(..., reward=rewards["reward"], ...)
```

Each `Episode` stores the prompt, generated tokens, whether the sequence finished, and the reward — everything needed for the policy update.

---

## 3. Step 2 — Group-Relative Advantage

### The math

This is the core idea of GRPO. Within each group of $G$ responses to the same prompt, compute the **mean** and **standard deviation** of rewards, then normalize:

$$\boxed{\hat{A}_i = \frac{r_i - \mu_{\mathbf{r}}}{\sigma_{\mathbf{r}} + \varepsilon}}$$

where $\mu_{\mathbf{r}} = \frac{1}{G}\sum_{i=1}^G r_i$ and $\sigma_{\mathbf{r}} = \sqrt{\frac{1}{G}\sum_{i=1}^G (r_i - \mu_{\mathbf{r}})^2}$.

**Intuition:**
- $\hat{A}_i > 0$ → this response was better than average → increase its probability  
- $\hat{A}_i < 0$ → this response was worse than average → decrease its probability  
- The normalization keeps gradients on a consistent scale regardless of reward magnitude

This replaces the critic: instead of $\hat{A}_t = r_t + \gamma V(s_{t+1}) - V(s_t)$, we use the group as a self-referential baseline.

### The code

```python
def normalize_rewards_per_group(episodes: List[Episode]) -> List[Episode]:
    groups = defaultdict(list)
    for episode in episodes:
        groups[tuple(episode.prefix)].append(episode)   # group by prompt

    output = []
    for group in groups.values():
        group_rewards = [item.reward for item in group]
        mean_reward = np.mean(group_rewards)
        std_reward  = np.std(group_rewards)
        for episode in group:
            normalized = (episode.reward - mean_reward) / (std_reward + 1e-4)
            episode = dataclasses.replace(episode, reward=normalized)
            output.append(episode)
    return output
```

After this function, each `episode.reward` field holds $\hat{A}_i$ — the group-relative advantage, not the raw reward.

Note: $\hat{A}_i$ is a **scalar per response**. Every token in response $o_i$ gets the same advantage value. There is no token-level credit assignment within a single response (unlike PPO with a good critic).

---

## 4. Step 3 — Policy Update

### The objective

GRPO maximizes (R1-Zero variant — **no KL term**):

$$\boxed{J(\theta) = \frac{1}{N_{\text{tokens}}} \sum_{i=1}^{G} \sum_{t=1}^{|o_i|} \log \pi_\theta(o_{i,t} \mid q, o_{i,<t}) \cdot \hat{A}_i}$$

Compared to the full GRPO paper, R1-Zero **drops the KL divergence penalty** and **drops the clipping**. The update is a straightforward policy gradient weighted by the group-relative advantage.

> **Why no clipping here?** GRPO-Zero does only **one gradient update per rollout batch** (not multiple epochs over the same data like PPO). With a single update, the policy doesn't move far, so clipping is less critical.

### Token log-probability via cross-entropy

Computing $\log \pi_\theta(o_{i,t} \mid q, o_{i,<t})$ is equivalent to the **negative cross-entropy loss** at each token position. GRPO-Zero uses exactly this:

```python
log_probs = -torch.nn.functional.cross_entropy(
    logits.reshape(-1, logits.size(-1)),   # [B*T, vocab]
    target_token_ids.reshape(-1),           # [B*T]
    ignore_index=pad_token_id,
    reduction="none",
).reshape(B, T)                             # [B, T]  — one log-prob per token
```

### The policy gradient

The objective is then:

```python
obj  = log_probs * batch_advantages[:, None]   # broadcast advantage over tokens
obj  = (obj * target_masks).sum() / num_target_tokens  # mean over generated tokens
loss = -obj                                    # gradient ascent → minimize negative
loss.backward()
```

`target_masks` is 1 for generated tokens and 0 for prompt tokens — we only train on what the model produced, not on the input prompt.

`num_target_tokens` normalizes over the **total number of generated tokens** in the batch (token-level, not sequence-level), so longer responses don't dominate.

### The full `update_policy` function

```python
def update_policy(model, optimizer, episodes, micro_batch_size,
                  pad_token_id, max_grad_norm, device, dtype):

    episodes = normalize_rewards_per_group(episodes)       # Step 2
    episodes.sort(key=lambda x: len(x.prefix_token_ids)
                               + len(x.generated_token_ids))  # sort for efficient batching

    num_target_tokens = sum(len(e.generated_token_ids) for e in episodes)

    for i in range(0, len(episodes), micro_batch_size):    # gradient accumulation
        batch_episodes = episodes[i : i + micro_batch_size]

        # --- build tensors ---
        batch_token_ids  = [prefix + generated + padding]  # [B, T]
        batch_masks      = [0*len(prefix) + 1*len(gen) + 0*padding]
        batch_advantages = [episode.reward for episode in batch_episodes]

        # --- forward pass ---
        logits = model.forward(input_token_ids)             # [B, T, vocab]

        # --- log-probs ---
        log_probs = -cross_entropy(logits, targets, reduction="none")

        # --- weighted objective ---
        obj  = (log_probs * advantages[:, None] * masks).sum() / num_target_tokens
        loss = -obj
        loss.backward()

    # --- one optimizer step for the whole batch ---
    clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    optimizer.zero_grad()
```

---

## 5. Reward Function (CountDown Task)

GRPO-Zero uses the **CountDown** task: given numbers like `[1, 2, 3, 4]` and a target `11`, the model must produce an arithmetic expression like `1 + (2 * 3) + 4`.

The reward is the sum of two components:

| Component | Reward | Condition |
|---|---|---|
| Format reward | 0.0 – 1.0 | Does the output contain `<think>...</think><answer>...</answer>`? |
| Answer reward | 1.0 | Does the expression use each number exactly once **and** evaluate to the target? |
| Combined | `0.1 × format + answer` | |

```python
def format_reward_function(response, end_token=None):
    # Full correct format → 1.0
    # Has <think> tag → +0.1
    # Has <answer> tag → +0.5
    ...

def answer_reward_function(response, numbers, target):
    # Extract expression from <answer>...</answer>
    # Check all numbers used exactly once
    # Check eval(expression) ≈ target → 1.0
    ...
```

**Why this reward works well for GRPO:**  
The reward is binary and verifiable — no learned reward model needed. Within each group of $G$ responses, some will get `answer_reward=1` and some `answer_reward=0`, creating a clear spread that makes the group-relative advantage informative.

---

## 6. The Training Loop

```python
# train.py (simplified)
for step, batch in enumerate(train_dataloader):

    # Step 1: Generate G responses per prompt, score each
    episodes = rollout(model, batch,
                       num_answer_per_question=NUM_ANSWERS_PER_QUESTION, ...)

    # Step 2 + 3: Normalize advantages and update policy
    results = update_policy(model, optimizer, episodes, ...)
```

With `batch_size=256` and `num_questions_per_batch=32`, each step:
- Takes 32 prompts
- Generates 256/32 = **8 responses per prompt**
- Scores all 256 responses
- Normalizes within each group of 8
- Takes one gradient step

---

## 7. GRPO vs PPO — Key Differences

| | PPO | GRPO (R1-Zero) |
|---|---|---|
| Baseline | Critic network $V_\phi$ | Group mean reward $\mu_{\mathbf{r}}$ |
| Extra model | Critic (~same size as actor) | None |
| KL penalty | Optional | **Removed** (R1-Zero) |
| Clipping | ✅ | **Removed** (1 update/rollout) |
| Token-level advantage | Yes (via GAE) | No (same $\hat{A}_i$ for all tokens in $o_i$) |
| Dependencies | HuggingFace, DeepSpeed, … | Just `torch` + `tokenizers` |

---

## 8. Running GRPO-Zero

```bash
# Install
pip install uv
uv sync

# Download data and model
git clone https://huggingface.co/datasets/Jiayi-Pan/Countdown-Tasks-3to4
git clone https://huggingface.co/Qwen/Qwen2.5-3B-Instruct

# Train (48GB GPU)
uv run train.py

# Train (24GB GPU — offloads optimizer states to CPU)
uv run train.py --config config_24GB.yaml
```

---

## References

1. Shao et al. *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models.* 2024. [`arXiv:2402.03300`](https://arxiv.org/abs/2402.03300)
2. DeepSeek-AI. *DeepSeek-R1.* 2025. [`arXiv:2501.12948`](https://arxiv.org/abs/2501.12948)
3. Yu et al. *DAPO: An Open-Source LLM Reinforcement Learning System at Scale.* 2025. [`arXiv:2503.14476`](https://arxiv.org/abs/2503.14476)
4. GRPO-Zero implementation: [`github.com/policy-gradient/GRPO-Zero`](https://github.com/policy-gradient/GRPO-Zero)
