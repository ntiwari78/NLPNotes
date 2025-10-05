
# 📘 Lecture Summary: Word Vectors, Optimization, & Neural Classifiers

## 🧩 Plan for Today

1. Finish reviewing **optimization basics** (gradient descent, stochastic gradient descent)
2. Delve into **word2vec** and **word vector** techniques (skip-gram, negative sampling)
3. Discuss alternatives based purely on **co-occurrence counts & matrix factorization**
4. Talk about **evaluation** of word vectors (intrinsic vs extrinsic)
5. Introduce **neural network classifiers** and begin the intuition behind what a neural network is

---

## 🔧 Optimization Review

* We have a loss function $(J(\theta))$ . To minimize it, we compute the **gradient** $(\nabla J)$ , pick a small step size (learning rate $(\alpha))$ , and update:

$$
  [
  \theta \leftarrow \theta - \alpha \nabla J(\theta)
  ]
$$

* **Gradient descent** (batch) uses the full training set to compute (\nabla J) at every update — expensive if dataset is large.

* **Stochastic gradient descent** (or mini‑batch gradient descent) picks a small random subset (e.g. 16, 32, or 64 examples), estimates the gradient using that subset, and updates.

  * It’s noisier, but much faster.
  * In practice, the noise can help escape shallow local minima or plateaus in the landscape of neural networks.

---

## 🧠 Word Vectors & Word2Vec

### Basic idea

* Each word $( w )$ is associated with two vectors (for “center word” and “outside word”) — typically initialized randomly (small values).
* Given a center word, the model tries to predict its context words (skip-gram) by dot products between word vectors, passing them through a softmax to get predicted probabilities.
* The discrepancy between predictions and actual context words gives a loss; gradients of that loss are used to update the vectors.

### Why two vectors per word?

* Using separate vectors for "center" and "outside" simplifies the mathematics (avoids quadratic terms when a word appears both as center and outside in the same context).
* In practice, people sometimes average the two vectors after training to get a single vector representation.

### Negative Sampling

* Softmax requires summing over the entire vocabulary (which is huge), making computation expensive.
* **Negative sampling** is a simplification: for a given positive (center, context) pair, sample a small number (e.g. 5 or 10) of “negative” words (not in the context).
* Use the logistic (sigmoid) function $(\sigma(x) = 1 / (1 + e^{-x}))$ .
* Encourage high (\sigma(\mathbf{v}*{center} \cdot \mathbf{v}*{outside})) for true pairs, and low (\sigma(\cdot)) for negative pairs.
* The negative words are sampled from a modified unigram distribution (often raising frequencies to the 3/4 power).

---

## 📊 Count-Based & Factorization Alternatives

* A straightforward approach: build a **co-occurrence matrix** ( C ) of size $( V \times V )$ , where ( C_{ij} ) is how often word ( j ) appears in the context of word ( i ).

* But $( V \times V )$ is huge (e.g. 400,000 × 400,000), so this is impractical as is.

* Use **dimensionality reduction** (e.g. PCA or Singular Value Decomposition, SVD) to reduce the matrix to lower-dimensional representations.

* Earlier work (e.g. LSA) used this approach.

* GloVe (Global Vectors) is an approach that blends the strengths of count-based and predictive models: it models (\log) of co-occurrence counts or ratios of co-occurrence probabilities using word vectors and biases, aiming for linear relationships in the vector space.

---

## ✅ Evaluation of Word Vectors

### Intrinsic vs. Extrinsic

* **Intrinsic evaluation**: directly evaluate the embeddings on small linguistic tasks, e.g. analogies, word similarity datasets.
* **Extrinsic evaluation**: embed words and then test performance on a downstream task (e.g. Named Entity Recognition, part-of-speech tagging, machine translation) — more realistic, but more work.

### Intrinsic methods used

* **Analogies**: “man : king :: woman : ?” → should return “queen”
* **Word similarity judgments**: human annotators score pairs of words (e.g. “car” and “automobile”) on how similar they think they are. Compare model cosine similarity to human scores (correlation).

---

## 🧬 Word Senses and Ambiguity

* Many words are **polysemous** — multiple senses (e.g. “bank” = financial institution vs riverbank).
* One approach: represent each sense separately — cluster word occurrences by context and learn separate vector for each sense.
* But the more common modern approach: learn a **single vector per word**, which is a weighted average (superposition) over senses.
* Interestingly, via **sparse coding** or related techniques, one can sometimes *recover* sense-level vectors from the single vector in high-dimensional spaces.

---

## 🧮 Neural Classifiers & Basic Neural Network Intuition

### Motivation

* Traditional classifiers (logistic regression, SVM, etc.) are **linear** in the input features.
* A neural classifier can learn **representations** of the input (via hidden layers) that make classification easier (i.e. nonlinear decision boundaries in the original space).

### Setup for a small neural network classifier

* Input: a window of words (e.g. 5 words), convert each word to its vector, and **concatenate** those vectors into one long input vector.
* Hidden layer: apply an affine transformation (weight matrix + bias) + nonlinearity (e.g. logistic, ReLU).
* Output layer: linear score → apply logistic (for binary classification) or softmax (for multiple classes).
* When trained end-to-end, both the word embeddings and the classifier weights get adjusted, so the whole model “learns” meaningful features.

### Connections to logistic regression

* A single neuron + logistic = logistic regression.
* Neural networks stack many such “neurons” (units) in layers.
* The hidden layers allow the model to transform input into representations better suited for the final linear classification.

---

## 🔍 Cross-Entropy Loss (Preview for PyTorch Assignment)

* **Cross-entropy** is a loss function from information theory. If (p) is the true distribution and (q) is the predicted distribution, the cross-entropy is:

$$
  [
  H(p, q) = - \sum_i p_i \log q_i
  ]
$$

* In classification with “one-hot” true labels (y), only the term corresponding to the correct class remains, giving:

$$
  [

  * \log q_{y}
    ]
$$

* In PyTorch, you'll typically use **`CrossEntropyLoss`**, which combines the log-softmax + negative log likelihood internally.

---

## 🧠 Biological Inspiration (Brief)

* The concept of a **neuron**: many inputs converge, processed in the cell body; if activation is strong enough, output is generated (axon).
* Artificial neural networks loosely mimic this: weighted sum of inputs → nonlinearity → output.
* The networks build complexity by stacking layers of such “neurons.”

---
---



# 📘 Study Guide: Word Vectors and Introduction to Neural Networks

This guide reviews core concepts in word embeddings, optimization, evaluation strategies, and neural network classifiers.

---

## 🧪 Short-Answer Quiz

> **Instructions**: Answer each question in 2–3 sentences using concepts from the lecture.

1. **What is the core difference between standard Gradient Descent and Stochastic Gradient Descent (SGD), and why is SGD preferred for training neural networks?**
   Standard Gradient Descent computes gradients using the full dataset, which is computationally expensive. SGD uses mini-batches, allowing faster updates and introducing noise that can help escape local minima.

2. **Why are two sets of vectors used for each word in Word2Vec?**
   Using separate center (`v`) and outside (`u`) vectors avoids quadratic terms during optimization and simplifies calculations. These vectors are often averaged post-training.

3. **Describe "analogies" in Word2Vec.**
   Word2Vec captures semantic relationships as vector operations, such as:
   `vector(King) - vector(man) + vector(woman) ≈ vector(Queen)`.

4. **What is the computational challenge of softmax, and how does negative sampling help?**
   Softmax requires computing a sum over the full vocabulary. Negative sampling simplifies this by training the model to distinguish real context words from a few negative samples.

5. **How do count-based methods differ from predictive models like Word2Vec?**
   Count-based models build a [co-occurrence matrix](https://en.wikipedia.org/wiki/Co-occurrence_matrix) and reduce its dimensionality (e.g., via [SVD](https://en.wikipedia.org/wiki/Singular_value_decomposition)). Predictive models like Word2Vec learn vectors directly by predicting context words.

6. **What is GloVe’s central insight?**
   GloVe models ratios of co-occurrence probabilities using word vectors. It maps the dot product of word vectors to the logarithm of their co-occurrence count.

7. **Define intrinsic and extrinsic evaluation with examples.**
   *Intrinsic:* Evaluate vectors on analogy tasks.
   *Extrinsic:* Use vectors in downstream tasks like [Named Entity Recognition (NER)](https://en.wikipedia.org/wiki/Named-entity_recognition).

8. **How do standard models handle polysemy?**
   They assign a single vector per word, representing a weighted average of all senses.

9. **What is a key advantage of neural classifiers over logistic regression?**
   Neural classifiers learn both the representation (e.g., word vectors) and classifier, allowing nonlinear boundaries and improved performance.

10. **Relationship between cross-entropy loss and negative log-likelihood?**
    In supervised learning, [cross-entropy](https://en.wikipedia.org/wiki/Cross_entropy) simplifies to the [negative log-likelihood](https://en.wikipedia.org/wiki/Likelihood_function) of the correct class.

---

## 🧠 Essay Questions

> Formulate comprehensive responses integrating concepts from the lecture.

1. **Compare Word2Vec and GloVe:**
   Discuss skip-gram's prediction-based training vs. GloVe's count-based matrix factorization. Compare their objective functions and how both capture analogies.

2. **Deconstruct the Word2Vec training process:**
   Explain initialization, prediction via dot products, error measurement, and how [SGD](https://en.wikipedia.org/wiki/Stochastic_gradient_descent) updates vectors to reflect semantics.

3. **Handling polysemy with a single vector:**
   Describe the rationale for superposition, the drawbacks of sense-specific vectors, and how [sparse coding](https://en.wikipedia.org/wiki/Sparse_coding) can uncover individual senses.

4. **Intrinsic vs. Extrinsic evaluation:**
   Analyze their roles in validating word embeddings, benefits of each, and why both are necessary for reliable assessment.

5. **From neurons to neural classifiers:**
   Trace the biological inspiration, show how a [logistic unit](https://en.wikipedia.org/wiki/Logistic_function) becomes a neuron, and how stacking layers enables deep [neural networks](https://en.wikipedia.org/wiki/Artificial_neural_network) to learn complex patterns.

---

## 📚 Glossary of Key Terms

| Term                                   | Definition                                                                                                         |
| -------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| **Analogies**                          | Linear semantic relationships captured in vector space (e.g., King - man + woman ≈ Queen).                         |
| **Bag of Words**                       | Text represented as unordered word collections; Word2Vec treats context this way.                                  |
| **Co-occurrence Matrix**               | Counts how often words appear near others in a context window.                                                     |
| **Cross-Entropy Loss**                 | Measures divergence between true and predicted distributions; reduces to negative log-likelihood for labeled data. |
| **Extrinsic Evaluation**               | Tests word vectors via downstream tasks like NER or sentiment analysis.                                            |
| **GloVe**                              | A count-based vector learning model that captures word relationships using co-occurrence probabilities.            |
| **Gradient Descent**                   | Optimization method using full dataset to compute gradients.                                                       |
| **Intrinsic Evaluation**               | Evaluates model on subtasks like word similarity or analogies.                                                     |
| **Learning Rate**                      | Step size in optimization algorithms (α) controlling update magnitude.                                             |
| **Linear Classifier**                  | Makes decisions using linear combinations of input features (e.g., logistic regression).                           |
| **Logistic Function (Sigmoid)**        | Maps real values into probabilities ((0, 1)); used in classification.                                              |
| **Mini-batch**                         | A subset of data used in SGD for faster, approximate gradient estimation.                                          |
| **Named Entity Recognition (NER)**     | NLP task: identify entities (people, places, etc.) in text.                                                        |
| **Negative Sampling**                  | Trains Word2Vec to distinguish true from false word-context pairs.                                                 |
| **Neural Network**                     | Layered model inspired by neurons, used to learn non-linear functions.                                             |
| **Polysemy**                           | Words with multiple meanings (e.g., “bat” as animal vs. sports equipment).                                         |
| **SVD (Singular Value Decomposition)** | Reduces matrix size to produce dense embeddings.                                                                   |
| **Skip-gram**                          | Word2Vec model predicting surrounding words given a center word.                                                   |
| **Softmax**                            | Converts raw scores to probabilities over multiple classes; expensive for large vocabularies.                      |
| **Sparse Coding**                      | Represents a vector using few basis components, helpful in disambiguating senses.                                  |
| **Stochastic Gradient Descent (SGD)**  | Efficient optimization using data subsets per update.                                                              |
| **Superposition (Word Senses)**        | A single vector averages multiple senses, weighted by usage.                                                       |
| **Supervised Learning**                | Learning from labeled input-output pairs.                                                                          |
| **Unigram Distribution**               | Probability distribution over individual word frequencies in corpus.                                               |
| **Word Vectors / Word Embeddings**     | Dense vector representations capturing semantic meaning.                                                           |
| **Word2Vec**                           | Predictive model producing word embeddings via skip-gram or CBOW architectures.                                    |

---

## 🔗 References

* [Gradient Descent](https://en.wikipedia.org/wiki/Gradient_descent)
* [Stochastic Gradient Descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent)
* [Word2Vec](https://en.wikipedia.org/wiki/Word2vec)
* [GloVe](https://nlp.stanford.edu/projects/glove/)
* [Co-occurrence Matrix](https://en.wikipedia.org/wiki/Co-occurrence_matrix)
* [Singular Value Decomposition](https://en.wikipedia.org/wiki/Singular_value_decomposition)
* [Sparse Coding](https://en.wikipedia.org/wiki/Sparse_coding)
* [Cross Entropy](https://en.wikipedia.org/wiki/Cross_entropy)
* [Neural Network](https://en.wikipedia.org/wiki/Artificial_neural_network)
* [Named Entity Recognition](https://en.wikipedia.org/wiki/Named-entity_recognition)
* [Sigmoid Function](https://en.wikipedia.org/wiki/Sigmoid_function)
* [Supervised Learning](https://en.wikipedia.org/wiki/Supervised_learning)

