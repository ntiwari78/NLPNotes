

# 📑 VIP Cheatsheet: Transformers & Large Language Models

*By Afshine Amidi & Shervine Amidi — March 23, 2025*

This cheatsheet distills content from the *Super Study Guide: Transformers & Large Language Models* (\~600 illustrations, 250+ pages). More at [superstudy.guide](https://superstudy.guide).

---

## 1. Foundations

### 1.1 Tokens

* **Definition**: A *token* is the indivisible unit of text (word, subword, character, byte).
* **Tokenizer (T)**: Divides text into tokens of varying granularity.

#### Types of Tokenizers

| Type          | Pros                              | Cons                                     | Example            |
| ------------- | --------------------------------- | ---------------------------------------- | ------------------ |
| **Word**      | Easy to interpret, short sequence | Large vocab, poor handling of variations | `teddy bear`       |
| **Subword**   | Word roots leveraged, intuitive   | Longer sequences, more complex           | `ted##dy`, `bear`  |
| **Character** | No OOV issues                     | Much longer sequences                    | `t`, `e`, `d`, ... |
| **Byte**      | Very small vocab                  | Patterns hard to interpret               | Byte codes         |

> **Note**: Byte-Pair Encoding (BPE) and Unigram are widely used subword tokenizers.
> **Special tokens**: `[UNK]` (unknown), `[PAD]` (padding).

---

### 1.2 Embeddings

* **Definition**: An embedding is a numerical vector representation of a token, sentence, etc.
* **Similarity**: Measured via **cosine similarity**

$$
\text{similarity}(t_1, t_2) = \frac{t_1 \cdot t_2}{\|t_1\|\|t_2\|} = \cos(\theta), \quad \in [-1,1]
$$

* Examples:

  * *Similar*: `cute` ↔ `teddy_bear`
  * *Dissimilar*: `unpleasant` ↔ `teddy_bear`
  * *Independent*: `airplane` ↔ `teddy_bear`

---

## 2. Transformers

### 2.1 Attention

* **Formula**: Given query $q$, determine how much attention it should pay to each key $k$, w\.r.t. associated value $v$.

$$
\text{attention} = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

* **Multi-Head Attention (MHA)**

  * Splits queries, keys, values into multiple heads.
  * Each head projects inputs via matrices $W^Q, W^K, W^V$.
  * Combined via projection $W^O$.

> Variants: **Grouped-Query Attention (GQA)** and **Multi-Query Attention (MQA)** share keys/values across heads for efficiency.

---

### 2.2 Architecture

* **Overview**:

  * Transformer = *encoders + decoders*.
  * Encoders: contextual embeddings of input.
  * Decoders: predict next tokens using encoder outputs.

---

## 🔗 References

* [Super Study Guide: Transformers & LLMs](https://superstudy.guide)
* [Attention Is All You Need (Transformer paper, Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
* [Tokenization (Wikipedia)](https://en.wikipedia.org/wiki/Tokenization_%28lexical_analysis%29)
* [Cosine Similarity (Wikipedia)](https://en.wikipedia.org/wiki/Cosine_similarity)
* [Multi-Head Attention Explained (Lil’Log Blog)](https://lilianweng.github.io/posts/2018-06-24-attention/)



# 📑 VIP Cheatsheet (Part 2): Transformers & Large Language Models


## 2.2 Architecture

### 🔹 Encoder–Decoder Framework

* **Encoders**: produce embeddings capturing the meaning of input tokens.
* **Decoders**: generate output embeddings, conditioned on both the input and what’s been generated so far.

#### Components

* **Encoder**:

  * Self-Attention
  * Feed-Forward Neural Network
* **Decoder**:

  * Masked Self-Attention (to prevent looking ahead)
  * Cross-Attention (attends to encoder outputs)
  * Feed-Forward Neural Network

📝 **Note**: Transformers were first proposed for **machine translation** but are now used in nearly all LLM applications.

---

### 🔹 Position Embeddings

* Inform the model about the **token’s position** in a sequence.
* Same dimensionality as token embeddings.
* Can be fixed, learned, or rotational.

💡 **Remark**: *Rotary Position Embeddings (RoPE)* efficiently rotate query & key vectors to capture **relative** position information.

---

## 2.3 Variants

* **Encoder-only (BERT)**

  * Stack of encoders.
  * Produces contextual embeddings → downstream tasks like classification.
  * Example: `[CLS]` token captures sentence meaning → sentiment analysis.

* **Decoder-only (GPT)**

  * Stack of decoders.
  * Autoregressive: predicts next token step by step.
  * Example: input prompt + generated continuation.
  * Used in GPT, LLaMA, Mistral, Gemma, DeepSeek, etc.

> 📝 *Encoder–decoder models like T5 are also autoregressive and share many features with decoder-only architectures.*

---

## 2.4 Optimizations

### 🔹 Attention Approximation

* **Problem**: Standard self-attention is $O(n^2)$, expensive for long sequences.
* **Solutions**:

  * **Sparsity**: attend only to the most relevant tokens.
  * **Low-rank**: approximate attention via low-rank matrix factorization.

### 🔹 Flash Attention

* Exact attention optimization leveraging **GPU hardware**.
* Uses **fast SRAM** before writing to **slower HBM**.
* Reduces memory usage + speeds computation significantly.

---

## ✅ Summary Table

| Concept             | Key Idea                                                                             |
| ------------------- | ------------------------------------------------------------------------------------ |
| **Encoder**         | Encodes contextual meaning from input tokens.                                        |
| **Decoder**         | Generates output token by token using masked + cross-attention.                      |
| **Position Embeds** | Encode token order (fixed, learned, or relative like RoPE).                          |
| **Encoder-only**    | BERT → contextual embeddings for tasks like classification.                          |
| **Decoder-only**    | GPT → autoregressive generation, widely used in today’s LLMs.                        |
| **Optimizations**   | Sparsity & low-rank approximations reduce compute, FlashAttention boosts efficiency. |

---

## 📚 References

* [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762)
* [Rotary Position Embedding (RoPE) paper](https://arxiv.org/abs/2104.09864)
* [BERT: Pre-training of Deep Bidirectional Transformers (Devlin et al., 2018)](https://arxiv.org/abs/1810.04805)
* [GPT (Radford et al., 2018–2020)](https://openai.com/research/overview)
* [FlashAttention: Faster Attention with IO-Awareness (Dao et al., 2022)](https://arxiv.org/abs/2205.14135)



# 📑 VIP Cheatsheet (Part 3): Large Language Models

---

## 3.1 Overview

* **Definition**:
  A **Large Language Model (LLM)** is a Transformer-based neural model with billions of parameters, trained to handle complex NLP tasks.

* **Lifecycle**:
  LLM training involves **three steps**:

  1. **Pretraining** → learn general patterns of language
  2. **Finetuning** → adapt to specific tasks
  3. **Preference tuning** → align model with human expectations

---

## 3.2 Prompting

* **Context length**: The maximum number of tokens an LLM can process at once (ranges from thousands to millions).

* **Decoding sampling**:
  Token predictions $p_i$ sampled from softmax distribution controlled by **temperature $T$**:

$$
p_i = \frac{\exp(x_i/T)}{\sum_{j=1}^n \exp(x_j/T)}
$$

* $T \ll 1$: deterministic, conservative outputs

* $T \gg 1$: creative, diverse outputs

* **Chain-of-Thought (CoT)**:
  Reasoning technique where models generate intermediate steps before the final answer.

  * **Tree of Thoughts (ToT)**: advanced form exploring multiple reasoning paths.
  * **Self-consistency**: aggregates multiple reasoning paths for better reliability.

---

## 3.3 Finetuning

* **SFT (Supervised Fine-Tuning)**:
  Post-training alignment using high-quality input-output pairs.

  * Special case: *instruction tuning* when the data is task instructions.

* **PEFT (Parameter-Efficient Fine-Tuning)**:
  Adaptation methods that fine-tune models without retraining all parameters.

  * Example: **LoRA (Low-Rank Adaptation)** approximates weight matrices with low-rank updates:

$$
W \approx W_0 + A B
$$

---

## 3.4 Preference Tuning

* **Reward model (RM)**: Predicts how well an output aligns with desired behavior.

  * *Best-of-N sampling*: choose best response among N candidates using RM.

* **Reinforcement learning (RL)**: Updates model $f$ based on reward signals.

  * **RLHF (Reinforcement Learning from Human Feedback)**: RM is trained on human preferences.
  * **PPO (Proximal Policy Optimization)**: popular RL algorithm ensuring stable updates.

> 📝 Other approaches: **Direct Preference Optimization (DPO)** and supervised hybrids combining RM + RL.

---

## 3.5 Optimizations

* **Mixture of Experts (MoE)**:
  Activates only a subset of model neurons per input via a gating network $G$.

  * Improves inference efficiency.
  * Used in models like **Switch Transformers**, though challenging to train.

$$
\hat{y} = \sum_{i=1}^n G(x)_i E_i(x)
$$

* **Distillation**:
  Train a smaller **student model** $S$ using predictions from a larger **teacher model** $T$.

  * Uses **KL divergence loss**:

$$
KL(y_T \| y_S) = \sum_i y_T(i) \log \frac{y_T(i)}{y_S(i)}
$$

* Training labels treated as *soft* probabilities.

---

## ✅ Key Takeaways

| Concept               | Purpose                                                            |
| --------------------- | ------------------------------------------------------------------ |
| **Lifecycle**         | Pretraining → Finetuning → Preference Tuning                       |
| **Prompting**         | Control creativity with temperature; use CoT for reasoning         |
| **SFT**               | Direct supervised alignment to tasks                               |
| **PEFT (LoRA)**       | Efficient tuning with low-rank updates                             |
| **Preference tuning** | Align model with human values via RM, RL, or hybrids               |
| **MoE**               | Efficient gating activates only relevant experts                   |
| **Distillation**      | Compress big models into smaller students with soft-label training |

---

## 📚 References

* [LoRA: Low-Rank Adaptation (Hu et al., 2021)](https://arxiv.org/abs/2106.09685)
* [RLHF (Christiano et al., 2017)](https://arxiv.org/abs/1706.03741)
* [PPO: Proximal Policy Optimization (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347)
* [Mixture of Experts (Shazeer et al., 2017)](https://arxiv.org/abs/1701.06538)
* [Knowledge Distillation (Hinton et al., 2015)](https://arxiv.org/abs/1503.02531)



# 📑 VIP Cheatsheet (Part 4): Applications, Agents & Reasoning



## 🔹 4.1 LLM-as-a-Judge (LaaJ)

**Definition**:
LLM-as-a-Judge is a method where an LLM scores outputs given criteria, producing both a **score** and a **rationale**.

Example:

* **Input**: “Teddy bear” (criterion: cuteness)
* **Output**: *“Teddy bears are the cutest”* → Score: 10/10

### ✅ Pros

* No need for reference text (unlike ROUGE).
* Correlates well with human ratings when large models are used.

### ⚠️ Common Biases

| Bias                      | Problem                              | Mitigation                           |
| ------------------------- | ------------------------------------ | ------------------------------------ |
| **Position bias**         | Favors first in pairwise comparisons | Randomize positions, average scores  |
| **Verbosity bias**        | Favors more verbose content          | Penalize long outputs                |
| **Self-enhancement bias** | Favors outputs generated by itself   | Use different base model for judging |

---

## 🔹 4.2 Retrieval-Augmented Generation (RAG)

**Definition**:
RAG augments prompts with **external knowledge retrieval**, helping overcome the LLM’s cutoff date or missing facts.

* **Retriever**: fetches relevant docs → encodes into vectors → attaches to prompt.
* Commonly uses **encoder-only embeddings** for retrieval.
* **Hyperparameters**: doc chunk size $n_c$, embedding dimension $d$.

---

## 🔹 4.3 Agents

**Definition**:
An **agent** autonomously completes tasks via sequences of LLM calls.

### ReAct (Reason + Act) framework

Steps:

1. **Observe**: summarize observations and current state
2. **Plan**: decide next actions/tools
3. **Act**: call an API or query knowledge base

> ⚠️ Evaluation of agents is difficult, must be done at both component-level (inputs/outputs) and system-level (task chains).

---

## 🔹 4.4 Reasoning Models

**Definition**:
Models that explicitly output **Chain-of-Thought (CoT) traces** to solve multi-step tasks.

* Examples: **OpenAI o-series, DeepSeek-R1, Google Gemini Flash Thinking**
* *DeepSeek-R1* outputs reasoning traces within `<think>` tags.

### Scaling Methods for Reasoning

| Type                   | Description                                                               | Benefit                      |
| ---------------------- | ------------------------------------------------------------------------- | ---------------------------- |
| **Train-time scaling** | Train longer so the model learns CoT-style reasoning before final answers | Boosts reasoning reliability |
| **Test-time scaling**  | Extend inference time by forcing multiple reasoning steps (e.g., “wait”)  | Improves reasoning depth     |

---

## 🔹 4.5 Quantization

* **Definition**: Reduces precision of model weights to shrink memory & speed inference.
* **Remark**: *QLoRA* is a widely used quantized variant of LoRA.

---

## ✅ Key Takeaways

| Concept              | Purpose                                                            |
| -------------------- | ------------------------------------------------------------------ |
| **LLM-as-a-Judge**   | Automates evaluation of outputs with rationale; must monitor bias  |
| **RAG**              | Adds external knowledge retrieval to overcome LLM knowledge cutoff |
| **Agents (ReAct)**   | Chain LLM calls with Observe–Plan–Act loops for complex tasks      |
| **Reasoning models** | Explicit CoT traces, enhanced via scaling (train or test time)     |
| **Quantization**     | Speeds inference & reduces memory with minimal performance loss    |

---

## 📚 References

* [RAG: Retrieval-Augmented Generation (Lewis et al., 2020)](https://arxiv.org/abs/2005.11401)
* [ReAct: Reason + Act (Yao et al., 2022)](https://arxiv.org/abs/2210.03629)
* [DeepSeek-R1 announcement](https://arxiv.org/abs/2405.04434)
* [QLoRA: Efficient Finetuning (Dettmers et al., 2023)](https://arxiv.org/abs/2305.14314)

