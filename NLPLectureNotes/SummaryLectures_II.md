
# Natural Language Processing – Lecture 22

## 🔠 GloVe: Global Vectors for Word Representation

This lecture introduces **GloVe (Global Vectors)**, an alternative to Word2Vec for learning word embeddings, developed at **Stanford (2014)**.

---

## 📖 What is GloVe?

* A **log-bilinear model** that learns word embeddings from **global word–context co-occurrence statistics**.
* Unlike **Word2Vec** (predictive), GloVe is a **count-based** model.

📖 [GloVe (Wikipedia)](https://en.wikipedia.org/wiki/GloVe_%28machine_learning%29)

---

## 🧠 Motivation

* Word meaning depends not just on **local context** (Word2Vec) but also **global co-occurrence patterns**.
* Example:

  * “ice” and “steam” both co-occur with “water”, but with different probabilities.
  * GloVe captures this **distributional difference**.

---

## ⚙️ Co-occurrence Matrix

* Build a large **matrix $X$** where:

  * $X_{ij}$ = number of times word $j$ appears in context of word $i$.

* Example:

  * For “ice”, context words = *solid, cold, melt* (high values).
  * For “steam”, context words = *gas, boil, hot* (high values).

---

## 🔢 GloVe Objective Function

* Goal: Learn word vectors $w_i$ such that:

  $$
  w_i^T \tilde{w}_j + b_i + \tilde{b}_j = \log(X_{ij})
  $$

* Where:

  * $w_i$ = word vector
  * $\tilde{w}_j$ = context word vector
  * $X_{ij}$ = co-occurrence count

* Uses a **weighted least-squares loss**:

  $$
  J = \sum_{i,j=1}^{V} f(X_{ij}) (w_i^T \tilde{w}_j + b_i + \tilde{b}_j - \log(X_{ij}))^2
  $$

📖 [GloVe Paper](https://nlp.stanford.edu/pubs/glove.pdf)

---

## ⚡ Key Properties of GloVe

1. **Combines local + global information**

   * Local context (like Word2Vec)
   * Global co-occurrence statistics

2. **Linear Substructure**

   * Captures analogies:

     $$
     vector(king) - vector(man) + vector(woman) \approx vector(queen)
     $$

3. **Efficient Training**

   * Works on very large corpora (Wikipedia, Common Crawl).

---

## 📊 GloVe vs Word2Vec

| Aspect        | Word2Vec (Predictive)        | GloVe (Count-Based)                   |
| ------------- | ---------------------------- | ------------------------------------- |
| Approach      | Predict word from context    | Matrix factorization of co-occurrence |
| Strength      | Captures local patterns well | Captures global statistics            |
| Analogy Tasks | Good                         | Good (sometimes better)               |
| Efficiency    | Fast training                | More memory-intensive                 |

📖 [Comparison of Word Embeddings](https://en.wikipedia.org/wiki/Word_embedding)

---

## 🔍 Applications of GloVe

* **Semantic Similarity**
* **Information Retrieval**
* **Question Answering**
* **Machine Translation**
* **Clustering & Classification**

---

## 📌 Summary

* **GloVe** = word embedding model based on **matrix factorization** of word–context co-occurrence.
* Captures both **local context** and **global distributional information**.
* Performs well on **semantic similarity and analogy tasks**.

---

## References

* [GloVe (Wikipedia)](https://en.wikipedia.org/wiki/GloVe_%28machine_learning%29)
* [GloVe Paper (Stanford)](https://nlp.stanford.edu/pubs/glove.pdf)
* [Word Embeddings](https://en.wikipedia.org/wiki/Word_embedding)


# Natural Language Processing – Lecture 23

## 🔠 Subword and Character-Level Embeddings

This lecture covers **subword** and **character-level embeddings**, approaches that address the **out-of-vocabulary (OOV)** problem and improve word representation.

---

## ⚠️ Limitations of Word-Level Embeddings

* Word2Vec, GloVe, FastText use **fixed vocabulary**.
* Issues:

  1. **OOV Words** → New words not in training vocabulary get `<UNK>`.
  2. **Morphological Variants** → "play", "plays", "playing" → different vectors.
  3. **Rare Words** → Poor embeddings due to low frequency.

📖 [Word Embedding Limitations](https://en.wikipedia.org/wiki/Word_embedding#Limitations)

---

## 🧠 Subword Embeddings

### Idea

* Break words into **subword units** (morphemes, character n-grams).
* Represent a word as **sum/average of subword embeddings**.

### Example

* Word: *playing* → \[play] + \[ing]
* Word: *unbelievable* → \[un] + \[believe] + \[able]

---

### 🔹 FastText (Facebook, 2016)

* Extends Word2Vec with **character n-grams**.
* Word embedding = sum of embeddings of all its n-grams.
* Handles **rare and OOV words** gracefully.

📖 [FastText](https://en.wikipedia.org/wiki/FastText)

---

## 🧩 Character-Level Embeddings

### Idea

* Represent words at **character level** instead of whole word.
* Learn embeddings for characters → build word representation using neural models.

### Approaches

1. **Character CNNs** → Convolutional layers over characters.
2. **Character RNNs** → LSTM/GRU sequence over characters.

📖 [Character Embeddings](https://en.wikipedia.org/wiki/Word_embedding#Character-level_embeddings)

---

## ⚡ Advantages

* **OOV Handling** → Can construct embeddings for unseen words.
* **Morphology Awareness** → Better at inflected/derived forms.
* **Compact Vocabulary** → Works well for morphologically rich languages.

---

## ⚠️ Limitations

* **Computationally expensive** → character models slower than word-level.
* **Semantic meaning** sometimes harder to capture at character level alone.

---

## 🔍 Applications

* **Morphologically rich languages** (Finnish, Turkish).
* **Speech recognition & OCR** → frequent misspellings.
* **Social media text** → handles slang, creative spellings.
* **Machine translation** → improves robustness.

---

## 📌 Summary

* Word-level embeddings struggle with **OOV, morphology, rare words**.
* **Subword embeddings** (e.g., FastText) use n-grams to improve representation.
* **Character-level embeddings** use CNNs/RNNs over characters.
* Both approaches improve robustness in **real-world NLP tasks**.

---

## References

* [Word Embeddings](https://en.wikipedia.org/wiki/Word_embedding)
* [FastText](https://en.wikipedia.org/wiki/FastText)
* [Character-Level Embeddings](https://en.wikipedia.org/wiki/Word_embedding#Character-level_embeddings)



# Natural Language Processing – Lecture 24

## 🧩 Contextual Word Embeddings

This lecture introduces **contextual embeddings**, which represent words differently depending on their **context**, unlike static embeddings (Word2Vec, GloVe, FastText).

---

## ⚠️ Limitations of Static Embeddings

* **Word2Vec / GloVe** assign a **single vector per word**.
* Fails with:

  * **Polysemy** → “bank” (riverbank vs financial institution).
  * **Contextual nuances** → “light” (not heavy vs illumination).

📖 [Word Embedding Limitations](https://en.wikipedia.org/wiki/Word_embedding#Limitations)

---

## 🧠 Contextual Embeddings

* Words are represented as **functions of their context**.
* Embedding of *“bank”* in:

  * “He deposited money in the bank” ≠ “He sat by the river bank”.
* Achieved via **deep neural language models**.

📖 [Contextual Embeddings](https://en.wikipedia.org/wiki/Word_embedding#Contextual_embeddings)

---

## 🔹 Early Contextual Models

### 1. **ELMo (Embeddings from Language Models, 2018)**

* Based on **bidirectional LSTMs**.
* Produces embeddings as **functions of entire sentence**.
* Context-sensitive and task-specific.

📖 [ELMo](https://en.wikipedia.org/wiki/ELMo)

---

### 2. **ULMFiT (2018)**

* **Universal Language Model Fine-Tuning**.
* Transfer learning for NLP.
* Pretrained language model fine-tuned for downstream tasks.

📖 [ULMFiT](https://arxiv.org/abs/1801.06146)

---

## 🔹 Transformer-Based Models

### 1. **BERT (2018, Google)**

* **Bidirectional Transformer**.
* Pretraining tasks:

  * **Masked Language Modeling (MLM)** → predict missing word.
  * **Next Sentence Prediction (NSP)** → predict if one sentence follows another.
* Produces **contextual embeddings for each token**.

📖 [BERT](https://en.wikipedia.org/wiki/BERT_%28language_model%29)

---

### 2. **GPT (Generative Pretrained Transformer, OpenAI)**

* Autoregressive transformer (predict next word).
* Learns powerful contextual representations.

📖 [GPT](https://en.wikipedia.org/wiki/GPT-3)

---

## ⚡ Advantages of Contextual Embeddings

* Handle **polysemy & context**.
* Improve performance across NLP tasks:

  * **Named Entity Recognition (NER)**
  * **Question Answering (QA)**
  * **Sentiment Analysis**
  * **Machine Translation**

---

## ⚠️ Challenges

* **Resource-intensive** (training BERT/GPT requires massive data + compute).
* **Bias in training data** can propagate into embeddings.

📖 [Bias in AI](https://en.wikipedia.org/wiki/Algorithmic_bias)

---

## 📌 Summary

* Static embeddings = single vector per word (context-independent).
* Contextual embeddings (ELMo, ULMFiT, BERT, GPT) = dynamic vectors based on context.
* Transformers revolutionized contextual representation.
* Widely used in **modern NLP pipelines**.

---

## References

* [Word Embedding](https://en.wikipedia.org/wiki/Word_embedding)
* [ELMo](https://en.wikipedia.org/wiki/ELMo)
* [ULMFiT](https://arxiv.org/abs/1801.06146)
* [BERT](https://en.wikipedia.org/wiki/BERT_%28language_model%29)
* [GPT](https://en.wikipedia.org/wiki/GPT-3)
* [Algorithmic Bias](https://en.wikipedia.org/wiki/Algorithmic_bias)


# Natural Language Processing – Lecture 25

## 🧩 Topic Modeling – Introduction

This lecture introduces **topic modeling**, an **unsupervised learning** technique used to discover hidden **topics** in a collection of documents.

---

## 📖 What is Topic Modeling?

* Task: Identify **latent themes (topics)** in a text corpus.
* Each **document** is represented as a mixture of topics.
* Each **topic** is a distribution over words.

📖 [Topic Model (Wikipedia)](https://en.wikipedia.org/wiki/Topic_model)

---

## 🧠 Motivation

* Large corpora (news, research papers, social media) are **unstructured**.
* Topic modeling helps:

  * **Summarize** collections of text.
  * **Cluster** documents by themes.
  * **Support search & recommendation systems**.

---

## 🔹 Approaches to Topic Modeling

### 1. **Latent Semantic Analysis (LSA)**

* Based on **Singular Value Decomposition (SVD)** of term-document matrix.
* Reduces dimensionality → uncovers hidden semantic structure.
* Limitation: purely algebraic, not probabilistic.

📖 [Latent Semantic Analysis](https://en.wikipedia.org/wiki/Latent_semantic_analysis)

---

### 2. **Probabilistic Latent Semantic Analysis (pLSA)**

* Models documents as a mixture of topics.
* Each topic is a probability distribution over words.
* Limitation: Number of parameters grows with documents → poor generalization.

📖 [Probabilistic Latent Semantic Analysis](https://en.wikipedia.org/wiki/Probabilistic_latent_semantic_analysis)

---

### 3. **Latent Dirichlet Allocation (LDA, 2003)**

* Most popular probabilistic topic model.
* Assumes:

  * Each document is a **mixture of topics**.
  * Each topic is a **distribution over words**.
* Uses **Dirichlet priors** to ensure generalization.

📖 [Latent Dirichlet Allocation](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation)

---

## ⚙️ How LDA Works (Simplified)

1. For each document:

   * Choose a distribution of topics.
2. For each word:

   * Choose a topic (according to document’s topic distribution).
   * Choose a word from that topic’s word distribution.

* Output:

  * Topics → lists of high-probability words.
  * Documents → mixtures of topics.

---

## 📊 Example (News Corpus)

* Topic 1: *government, election, policy, parliament*
* Topic 2: *game, team, player, match*
* Topic 3: *market, stock, company, profit*

A news article may be **70% politics, 20% sports, 10% finance**.

---

## 🔍 Applications of Topic Modeling

* **Document classification & clustering**
* **Recommender systems**
* **Trend detection** (e.g., social media)
* **Information retrieval** (semantic search)
* **Summarization**

---

## ⚠️ Limitations

* Number of topics must be chosen in advance.
* Topics may not always be **coherent** or human-interpretable.
* Sensitive to **preprocessing** (stop words, stemming, lemmatization).

---

## 📌 Summary

* Topic modeling = uncovering hidden themes in text corpora.
* Approaches: **LSA, pLSA, LDA**.
* **LDA** = most widely used probabilistic model.
* Applications in **classification, recommendation, IR, trend analysis**.

---

## References

* [Topic Model](https://en.wikipedia.org/wiki/Topic_model)
* [Latent Semantic Analysis](https://en.wikipedia.org/wiki/Latent_semantic_analysis)
* [Probabilistic Latent Semantic Analysis](https://en.wikipedia.org/wiki/Probabilistic_latent_semantic_analysis)
* [Latent Dirichlet Allocation](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation)



# Natural Language Processing – Lecture 26

## 📊 Latent Dirichlet Allocation (LDA) – Details

This lecture provides a deeper dive into **Latent Dirichlet Allocation (LDA)**, the most widely used **probabilistic topic modeling** method.

---

## 📖 Recap: What is LDA?

* A **generative probabilistic model** for documents.
* Each document is a **mixture of topics**.
* Each topic is a **distribution over words**.

📖 [Latent Dirichlet Allocation (Wikipedia)](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation)

---

## 🧠 LDA Generative Process

For each document $d$:

1. Choose topic proportions:

   * $\theta_d \sim Dirichlet(\alpha)$
   * (distribution over topics)

2. For each word $w$ in the document:

   * Choose a topic $z \sim Multinomial(\theta_d)$.
   * Choose word $w$ from topic’s distribution:

     * $w \sim Multinomial(\beta_z)$.

---

## ⚙️ Dirichlet Priors

* **$\alpha$** → controls topic distribution per document.

  * Small $\alpha$: documents focus on few topics.
  * Large $\alpha$: documents spread across many topics.

* **$\beta$** → controls word distribution per topic.

  * Small $\beta$: topics use fewer dominant words.
  * Large $\beta$: topics use a wide variety of words.

📖 [Dirichlet Distribution](https://en.wikipedia.org/wiki/Dirichlet_distribution)

---

## 🧮 Inference in LDA

* Goal: Given documents, infer:

  * Topic distribution for each document ($\theta$).
  * Word distribution for each topic ($\beta$).
  * Topic assignment for each word ($z$).

### Methods:

1. **Variational Inference** (approximation using optimization)
2. **Gibbs Sampling** (Markov Chain Monte Carlo)

📖 [Gibbs Sampling](https://en.wikipedia.org/wiki/Gibbs_sampling)

---

## 📊 Example Topics (LDA Output)

Corpus: News articles

* **Topic 1 (Politics):** government, election, policy, law
* **Topic 2 (Sports):** game, team, player, match
* **Topic 3 (Finance):** stock, market, profit, investment

Each document is represented as a **mixture of topics**.

---

## ⚡ Strengths of LDA

* Captures hidden **semantic structure**.
* Works in **unsupervised settings** (no labels needed).
* Interpretable: topics represented by **top words**.

---

## ⚠️ Limitations of LDA

1. **Number of topics (K) must be predefined**.
2. Sensitive to **preprocessing** (stop words, stemming).
3. Assumes **bag-of-words** model → ignores word order.
4. Struggles with **short texts** (Twitter, reviews).

---

## 🔍 Applications of LDA

* **Document classification & clustering**
* **Recommender systems** (content-based filtering)
* **Trend analysis** (e.g., social media, research topics)
* **Information retrieval** (semantic search)

---

## 📌 Summary

* LDA = **probabilistic topic model** with Dirichlet priors.
* Documents → mixtures of topics.
* Topics → distributions over words.
* Inference via **variational methods** or **Gibbs sampling**.
* Limitations: predefined K, bag-of-words assumption, preprocessing sensitivity.

---

## References

* [Latent Dirichlet Allocation](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation)
* [Dirichlet Distribution](https://en.wikipedia.org/wiki/Dirichlet_distribution)
* [Gibbs Sampling](https://en.wikipedia.org/wiki/Gibbs_sampling)
* [Topic Model](https://en.wikipedia.org/wiki/Topic_model)



# Natural Language Processing – Lecture 27

## 📰 Information Extraction – Introduction

This lecture introduces **Information Extraction (IE)**, a key NLP task that aims to automatically extract **structured information** (entities, relations, events) from **unstructured text**.

---

## 📖 What is Information Extraction?

* **Definition:** Converting unstructured text into **structured data**.
* Example:

Text: *“Barack Obama was born in Hawaii and served as the 44th President of the United States.”*

Extracted Information:

* Entity: Barack Obama
* Birthplace: Hawaii
* Position: 44th President of the US

📖 [Information Extraction (Wikipedia)](https://en.wikipedia.org/wiki/Information_extraction)

---

## 🔑 Core Tasks in Information Extraction

### 1. **Named Entity Recognition (NER)**

* Identify proper nouns and classify them into categories.
* Example:

  * “Apple” → Organization
  * “Paris” → Location
  * “Barack Obama” → Person

📖 [NER](https://en.wikipedia.org/wiki/Named-entity_recognition)

---

### 2. **Relation Extraction**

* Identify **relations** between entities.
* Example:

  * (Barack Obama, born-in, Hawaii)
  * (Barack Obama, president-of, USA)

📖 [Relation Extraction](https://en.wikipedia.org/wiki/Relationship_extraction)

---

### 3. **Event Extraction**

* Detect events and participants.
* Example:

  * Sentence: *“Google acquired YouTube in 2006.”*
  * Event: Acquisition
  * Entities: Google (acquirer), YouTube (acquired), 2006 (time)

📖 [Event Extraction](https://en.wikipedia.org/wiki/Information_extraction#Event_extraction)

---

## ⚙️ Approaches to Information Extraction

1. **Rule-Based Systems**

   * Use patterns & linguistic rules (regex, dependency parsing).
   * Example: “X was born in Y” → birth relation.
   * Limitation: brittle, hard to scale.

2. **Statistical / Machine Learning**

   * Train classifiers for NER, relation extraction.
   * Features: words, POS tags, dependency paths.

3. **Neural Approaches**

   * Use deep learning (RNNs, LSTMs, Transformers).
   * Pretrained models (e.g., [BERT](https://en.wikipedia.org/wiki/BERT_%28language_model%29)) achieve state-of-the-art.

---

## 📊 Example Pipeline

Sentence: *“Elon Musk founded SpaceX in 2002.”*

1. **NER** → Elon Musk (Person), SpaceX (Organization), 2002 (Date)
2. **Relation Extraction** → (Elon Musk, founded, SpaceX)
3. **Event Extraction** → Founding Event

---

## ⚡ Applications of Information Extraction

* **Knowledge Graph Construction** (e.g., Google Knowledge Graph, Wikidata)
* **Question Answering Systems**
* **Business Intelligence** (extracting company news, financial data)
* **Biomedical Text Mining** (drug–disease relationships)

---

## 📌 Summary

* **IE = structured data extraction** from unstructured text.
* Subtasks: **NER, relation extraction, event extraction**.
* Approaches: rule-based, ML-based, neural (transformers).
* Widely applied in **knowledge graphs, QA, biomedicine, business intelligence**.

---

## References

* [Information Extraction](https://en.wikipedia.org/wiki/Information_extraction)
* [Named Entity Recognition](https://en.wikipedia.org/wiki/Named-entity_recognition)
* [Relation Extraction](https://en.wikipedia.org/wiki/Relationship_extraction)
* [Event Extraction](https://en.wikipedia.org/wiki/Information_extraction#Event_extraction)
* [BERT](https://en.wikipedia.org/wiki/BERT_%28language_model%29)


# Natural Language Processing – Lecture 28

## 📰 Named Entity Recognition (NER)

This lecture focuses on **Named Entity Recognition (NER)**, a fundamental task in **Information Extraction (IE)** that identifies and classifies entities in text.

---

## 📖 What is NER?

* **Definition:** Detecting and classifying **named entities** in text into predefined categories.
* Common categories:

  * **Person** (e.g., “Barack Obama”)
  * **Organization** (e.g., “Google”)
  * **Location** (e.g., “Paris”)
  * **Date/Time** (e.g., “January 2020”)
  * **Miscellaneous** (e.g., events, nationalities)

📖 [NER (Wikipedia)](https://en.wikipedia.org/wiki/Named-entity_recognition)

---

## 🔑 Example

Sentence: *“Apple acquired Beats in 2014 for \$3 billion.”*

NER Output:

* Apple → Organization
* Beats → Organization
* 2014 → Date
* \$3 billion → Money

---

## ⚙️ Approaches to NER

### 1. **Rule-Based Approaches**

* Handcrafted rules, regex, gazetteers (lists of names).
* Example: Capitalized words after “Mr.” → Person.
* Limitation: brittle, domain-specific.

---

### 2. **Machine Learning Approaches**

* Train classifiers using features (POS tags, capitalization, context words).
* Algorithms: **Hidden Markov Models (HMMs)**, **Conditional Random Fields (CRFs)**.

📖 [Conditional Random Field](https://en.wikipedia.org/wiki/Conditional_random_field)

---

### 3. **Neural Approaches**

* Use deep learning for sequence labeling:

  * **RNNs / LSTMs**
  * **BiLSTM + CRF** models
  * **Transformers (BERT, RoBERTa, etc.)**
* Achieve **state-of-the-art performance**.

📖 [BERT](https://en.wikipedia.org/wiki/BERT_%28language_model%29)

---

## ⚠️ Challenges in NER

1. **Ambiguity**

   * “Apple” → company or fruit?
   * “Jordan” → country or person?

2. **Domain Adaptation**

   * News-trained NER may fail on biomedical text.

3. **Multilingual NER**

   * Languages with rich morphology (e.g., Turkish, Finnish).

4. **Emerging Entities**

   * New names (e.g., startups, social media hashtags).

---

## 📊 Applications of NER

* **Information Extraction** (structured knowledge bases).
* **Search Engines** (better indexing & ranking).
* **Question Answering** (entity-focused queries).
* **Business Intelligence** (tracking companies, products, people).
* **Biomedical NLP** (disease, drug, gene extraction).

---

## 📌 Summary

* NER identifies and classifies **entities** (persons, locations, organizations, dates, etc.).
* Approaches: **rule-based, statistical (CRFs, HMMs), neural (LSTMs, Transformers)**.
* Challenges: **ambiguity, domain adaptation, multilinguality, emerging entities**.
* Applications: **search, QA, knowledge graphs, biomedical IE**.

---

## References

* [Named Entity Recognition](https://en.wikipedia.org/wiki/Named-entity_recognition)
* [Conditional Random Field](https://en.wikipedia.org/wiki/Conditional_random_field)
* [BERT](https://en.wikipedia.org/wiki/BERT_%28language_model%29)
* [Information Extraction](https://en.wikipedia.org/wiki/Information_extraction)



# Natural Language Processing – Lecture 29

## 🔗 Relation Extraction

This lecture focuses on **Relation Extraction (RE)**, a key step in **Information Extraction (IE)** that identifies **relationships between entities** in text.

---

## 📖 What is Relation Extraction?

* **Definition:** Detecting and classifying **semantic relations** between entities in a sentence or document.
* Example:

  * Sentence: *“Barack Obama was born in Hawaii.”*
  * Extracted relation: *(Barack Obama, born-in, Hawaii)*

📖 [Relation Extraction (Wikipedia)](https://en.wikipedia.org/wiki/Relationship_extraction)

---

## 🔑 Types of Relations

1. **Binary Relations**

   * Involve two entities.
   * Example: *(Google, acquired, YouTube)*

2. **N-ary Relations**

   * Involve multiple entities.
   * Example: *(Google, acquired, YouTube, in 2006, for \$1.65B)*

3. **Common Relation Types**

   * Birthplace
   * Employment / Position
   * Organization–Location
   * Acquisition / Merger

---

## ⚙️ Approaches to Relation Extraction

### 1. **Rule-Based Approaches**

* Use **linguistic patterns** and **dependency parsing**.
* Example: Pattern “X was born in Y” → *born-in relation*.
* Limitation: brittle, domain-dependent.

---

### 2. **Supervised Learning**

* Train classifiers on labeled data (features: words, POS tags, dependency paths).
* Algorithms: **SVMs, CRFs, Decision Trees**.
* Requires **annotated corpora** (e.g., ACE dataset).

📖 [ACE Program](https://en.wikipedia.org/wiki/Automatic_Content_Extraction)

---

### 3. **Distant Supervision**

* Use existing **knowledge bases** (e.g., Freebase, Wikidata) to automatically generate training data.
* Assumption: if a KB says *(Obama, born-in, Hawaii)*, any sentence mentioning both is a training example.
* Issue: introduces **noise** (not all mentions express the relation).

📖 [Distant Supervision](https://en.wikipedia.org/wiki/Distant_supervision)

---

### 4. **Neural Approaches**

* Use **deep learning** to model relation extraction:

  * CNNs on sentences
  * RNNs / LSTMs for sequence modeling
  * Transformers (BERT, RoBERTa, T5) for **state-of-the-art RE**

📖 [BERT](https://en.wikipedia.org/wiki/BERT_%28language_model%29)

---

## ⚠️ Challenges in Relation Extraction

1. **Ambiguity**

   * "Paris" → city or person?
2. **Cross-Sentence Relations**

   * Relations may span multiple sentences.
3. **Noisy Data**

   * Especially in distant supervision.
4. **Domain Adaptation**

   * Models trained on news may fail in biomedical text.

---

## 📊 Applications of Relation Extraction

* **Knowledge Graph Construction** (Google Knowledge Graph, Wikidata).
* **Question Answering** (e.g., “Where was Obama born?” → Hawaii).
* **Business Intelligence** (mergers, partnerships).
* **Biomedical IE** (drug–disease, gene–protein relations).

---

## 📌 Summary

* Relation Extraction = detecting **semantic relations between entities**.
* Approaches: **rule-based, supervised, distant supervision, neural models**.
* Challenges: ambiguity, cross-sentence relations, noisy data, domain shift.
* Applications: **knowledge graphs, QA, biomedicine, business intelligence**.

---

## References

* [Relation Extraction](https://en.wikipedia.org/wiki/Relationship_extraction)
* [ACE Program](https://en.wikipedia.org/wiki/Automatic_Content_Extraction)
* [Distant Supervision](https://en.wikipedia.org/wiki/Distant_supervision)
* [BERT](https://en.wikipedia.org/wiki/BERT_%28language_model%29)



# Natural Language Processing – Lecture 30

## 🗓 Event Extraction

This lecture focuses on **Event Extraction (EE)**, the task of identifying **events and their participants** from unstructured text.

---

## 📖 What is Event Extraction?

* **Definition:** Detecting events (things that happen) and extracting their **arguments (participants, locations, time, etc.)**.
* Example:

  * Sentence: *“Google acquired YouTube in 2006 for \$1.65 billion.”*
  * Event: **Acquisition**
  * Arguments:

    * Acquirer → Google
    * Acquired → YouTube
    * Time → 2006
    * Price → \$1.65 billion

📖 [Event Extraction (Wikipedia)](https://en.wikipedia.org/wiki/Information_extraction#Event_extraction)

---

## 🔑 Components of Event Extraction

1. **Event Trigger**

   * Word/phrase that indicates an event.
   * Example: “acquired”, “married”, “born”.

2. **Event Arguments**

   * Entities associated with the event.
   * Example: *(Elon Musk, founded, SpaceX, 2002)*

3. **Event Types**

   * Birth, death, acquisition, merger, attack, election, etc.

---

## ⚙️ Approaches to Event Extraction

### 1. **Rule-Based**

* Use **lexical and syntactic patterns**.
* Example: Pattern “X was born in Y” → Birth Event.
* Limitation: domain-specific, brittle.

---

### 2. **Supervised Learning**

* Frame as a **classification problem**:

  * Trigger detection → classify word as event trigger or not.
  * Argument identification → classify entity roles.
* Algorithms: **SVMs, CRFs** with features like POS tags, dependency paths.

---

### 3. **Neural Approaches**

* Use **deep learning** to jointly model triggers and arguments.
* Architectures:

  * CNNs over dependency paths.
  * BiLSTMs for sequence modeling.
  * Transformers (e.g., **BERT**) for state-of-the-art performance.

📖 [BERT](https://en.wikipedia.org/wiki/BERT_%28language_model%29)

---

### 4. **Distant Supervision & Knowledge Bases**

* Use structured sources (e.g., **Wikidata, DBpedia**) to generate training data.
* Reduces need for manual annotation.

📖 [Distant Supervision](https://en.wikipedia.org/wiki/Distant_supervision)

---

## ⚠️ Challenges in Event Extraction

1. **Ambiguity**

   * “launched” → product launch, rocket launch, campaign launch.

2. **Complex Arguments**

   * Events may involve multiple roles (e.g., merger = two companies, date, location).

3. **Cross-Sentence Events**

   * Some events described across multiple sentences.

4. **Domain Adaptation**

   * Biomedical events (e.g., drug–gene interaction) differ from news events.

---

## 📊 Applications of Event Extraction

* **Knowledge Graph Construction** (event-based knowledge graphs).
* **News Monitoring** (tracking political, business, or crisis events).
* **Question Answering** (Who married whom? When?).
* **Biomedical IE** (protein interactions, clinical events).

---

## 📌 Summary

* Event Extraction = identify **event triggers** + **event arguments**.
* Approaches: **rule-based, supervised ML, neural models, distant supervision**.
* Challenges: ambiguity, complex roles, cross-sentence events, domain shift.
* Applications: **knowledge graphs, QA, biomedicine, news monitoring**.

---

## References

* [Event Extraction](https://en.wikipedia.org/wiki/Information_extraction#Event_extraction)
* [Distant Supervision](https://en.wikipedia.org/wiki/Distant_supervision)
* [BERT](https://en.wikipedia.org/wiki/BERT_%28language_model%29)
* [Knowledge Graph](https://en.wikipedia.org/wiki/Knowledge_Graph)



# Natural Language Processing – Lecture 31

## 📄 Text Classification – Introduction

This lecture introduces **Text Classification**, one of the most widely used applications of NLP, where the goal is to assign **predefined labels** to text.

---

## 📖 What is Text Classification?

* **Definition:** Task of categorizing text into one or more classes.
* Examples:

  * Spam Detection → *Spam / Not Spam*
  * Sentiment Analysis → *Positive / Negative / Neutral*
  * Topic Classification → *Sports, Politics, Technology*

📖 [Text Classification (Wikipedia)](https://en.wikipedia.org/wiki/Text_classification)

---

## 🔑 Applications

1. **Spam Filtering** (e.g., Gmail spam detection)
2. **Sentiment Analysis** (reviews, social media monitoring)
3. **Topic Categorization** (news, blogs)
4. **Intent Detection** (chatbots, virtual assistants)
5. **Toxicity Detection** (moderation in online platforms)

📖 [Sentiment Analysis](https://en.wikipedia.org/wiki/Sentiment_analysis)

---

## ⚙️ Approaches to Text Classification

### 1. **Rule-Based Approaches**

* Use manually crafted rules and keyword matching.
* Example: If email contains “lottery” and “prize” → Spam.
* Limitation: brittle, doesn’t generalize well.

---

### 2. **Traditional Machine Learning**

* Represent text as **features** (bag-of-words, TF-IDF, n-grams).
* Train classifiers such as:

  * **Naive Bayes**
  * **Logistic Regression**
  * **SVMs**
  * **Decision Trees / Random Forests**

📖 [Naive Bayes Text Classification](https://en.wikipedia.org/wiki/Naive_Bayes_text_classification)

---

### 3. **Neural Network Approaches**

* Learn dense **word embeddings** (Word2Vec, GloVe, FastText).
* Architectures:

  * **CNNs** → capture local patterns (n-grams).
  * **RNNs / LSTMs** → capture sequential context.
  * **Transformers (BERT, RoBERTa, etc.)** → capture contextual semantics.

📖 [BERT](https://en.wikipedia.org/wiki/BERT_%28language_model%29)

---

## 🧮 Feature Representations

1. **Bag-of-Words (BoW)**

   * Represent document as word counts.
   * Ignores order of words.

2. **TF-IDF (Term Frequency–Inverse Document Frequency)**

   * Weighs important words higher than frequent but uninformative ones.

3. **Word Embeddings**

   * Dense, low-dimensional vectors (semantic meaning).

📖 [TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)

---

## ⚠️ Challenges in Text Classification

1. **High Dimensionality** → Vocabulary can be huge.
2. **Data Sparsity** → Many rare words.
3. **Ambiguity** → “Apple” = fruit or company.
4. **Domain Adaptation** → Classifier trained on news may fail on tweets.
5. **Imbalanced Classes** → Some categories underrepresented.

---

## 📊 Evaluation Metrics

* **Accuracy**
* **Precision, Recall, F1-score**
* **Confusion Matrix**
* **ROC-AUC** (for imbalanced datasets)

📖 [Evaluation Metrics in ML](https://en.wikipedia.org/wiki/Precision_and_recall)

---

## 📌 Summary

* Text classification = assign labels to text.
* Approaches: **rule-based, ML (Naive Bayes, SVM), neural (CNNs, RNNs, Transformers)**.
* Features: **BoW, TF-IDF, embeddings**.
* Applications: spam detection, sentiment analysis, intent recognition.

---

## References

* [Text Classification](https://en.wikipedia.org/wiki/Text_classification)
* [Sentiment Analysis](https://en.wikipedia.org/wiki/Sentiment_analysis)
* [Naive Bayes Text Classification](https://en.wikipedia.org/wiki/Naive_Bayes_text_classification)
* [TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)
* [Precision and Recall](https://en.wikipedia.org/wiki/Precision_and_recall)
* [BERT](https://en.wikipedia.org/wiki/BERT_%28language_model%29)



# Natural Language Processing – Lecture 32

## 📄 Text Classification – Naive Bayes Classifier

This lecture explains **Naive Bayes Classifiers**, one of the simplest yet effective methods for **text classification**.

---

## 📖 Naive Bayes Classifier

* **Probabilistic model** based on **Bayes’ theorem** with the assumption of **feature independence**.
* Works well for **text classification** despite simplifying assumptions.

📖 [Naive Bayes Classifier (Wikipedia)](https://en.wikipedia.org/wiki/Naive_Bayes_classifier)

---

## 🔢 Bayes’ Theorem

For a document $d$ and class $c$:

$$
P(c|d) = \frac{P(d|c) \cdot P(c)}{P(d)}
$$

* $P(c|d)$: Posterior probability of class given document
* $P(d|c)$: Likelihood of document given class
* $P(c)$: Prior probability of class
* $P(d)$: Normalizing constant

---

## ⚙️ Naive Assumption

* Assume **conditional independence** of words given the class:

$$
P(d|c) = \prod_{i=1}^{n} P(w_i|c)
$$

* Works surprisingly well for text, even though independence is not strictly true.

---

## 🧮 Training Naive Bayes for Text

1. Compute **priors**:

$$
   P(c) = \frac{\text{# documents in class c}}{\text{Total documents}}
$$

2. Compute **likelihoods**:

$$
   P(w|c) = \frac{\text{Count}(w,c) + \alpha}{\sum_{w'} \text{Count}(w',c) + \alpha |V|}
$$

   * Uses **Laplace smoothing** ($\alpha$) to avoid zero probabilities.

📖 [Additive Smoothing](https://en.wikipedia.org/wiki/Additive_smoothing)

---

## 📊 Example: Spam Classification

Email: *“Win money now”*

* Classes: {Spam, Not Spam}
* Compute:

  * $P(\text{Spam}|\text{doc})$
  * $P(\text{Not Spam}|\text{doc})$
* Choose class with **maximum posterior probability**.

---

## ⚡ Advantages

* Simple and fast.
* Performs well with **small datasets**.
* Robust to **irrelevant features**.
* Widely used in **spam filtering, sentiment analysis**.

---

## ⚠️ Limitations

* Independence assumption not realistic.
* Poor performance when features are highly correlated.
* Struggles with **rare words** (needs smoothing).

---

## 📌 Summary

* Naive Bayes = **probabilistic classifier** based on Bayes’ theorem + independence assumption.
* Efficient for **text classification**.
* Uses **priors, likelihoods, smoothing**.
* Pros: fast, effective for spam/sentiment classification.
* Cons: ignores feature correlations.

---

## References

* [Naive Bayes Classifier](https://en.wikipedia.org/wiki/Naive_Bayes_classifier)
* [Additive Smoothing](https://en.wikipedia.org/wiki/Additive_smoothing)
* [Text Classification](https://en.wikipedia.org/wiki/Text_classification)




# Natural Language Processing – Lecture 33

## 📄 Text Classification – Beyond Naive Bayes

This lecture explores **alternative methods** for text classification beyond Naive Bayes, including **Logistic Regression, SVMs, and Neural Networks**.

---

## ⚖️ Logistic Regression for Text Classification

### Idea

* A **discriminative model** that directly estimates $P(c|d)$.
* Uses a **linear decision boundary** with a **sigmoid function**.

### Formula

$$
P(c|d) = \sigma(w \cdot x + b) = \frac{1}{1 + e^{-(w \cdot x + b)}}
$$

* $x$ = feature vector (e.g., TF-IDF, embeddings)
* $w$ = learned weights

📖 [Logistic Regression](https://en.wikipedia.org/wiki/Logistic_regression)

---

## 📊 Support Vector Machines (SVMs)

### Idea

* Find a **maximum-margin hyperplane** separating classes.
* Works well for **high-dimensional sparse data** (like text).

### Kernel Trick

* Allows nonlinear classification using kernels (e.g., polynomial, RBF).

📖 [Support Vector Machine](https://en.wikipedia.org/wiki/Support_vector_machine)

---

## 🧠 Neural Networks for Text Classification

### 1. **Feedforward Neural Networks**

* Simple classifiers using embeddings or BoW vectors.

### 2. **Convolutional Neural Networks (CNNs)**

* Capture **local patterns** (n-grams).
* Example: used in **sentence classification** tasks.

📖 [CNNs for NLP](https://en.wikipedia.org/wiki/Convolutional_neural_network#Applications)

### 3. **Recurrent Neural Networks (RNNs, LSTMs, GRUs)**

* Capture **sequential context** in text.
* Useful for sentiment, intent classification.

📖 [Recurrent Neural Network](https://en.wikipedia.org/wiki/Recurrent_neural_network)

### 4. **Transformers**

* Contextual embeddings (BERT, RoBERTa, DistilBERT).
* State-of-the-art for classification tasks.

📖 [BERT](https://en.wikipedia.org/wiki/BERT_%28language_model%29)

---

## 🧮 Feature Representations

* **Bag-of-Words (BoW)**
* **TF-IDF**
* **Word embeddings** (Word2Vec, GloVe, FastText)
* **Contextual embeddings** (BERT, GPT)

📖 [Word Embedding](https://en.wikipedia.org/wiki/Word_embedding)

---

## ⚡ Comparison of Approaches

| Method              | Strengths                  | Weaknesses                          |
| ------------------- | -------------------------- | ----------------------------------- |
| Naive Bayes         | Fast, simple               | Independence assumption unrealistic |
| Logistic Regression | Better decision boundaries | Still linear                        |
| SVMs                | Works well on sparse data  | Expensive on large datasets         |
| Neural Networks     | Capture deep features      | Require lots of data & compute      |
| Transformers (BERT) | State-of-the-art           | Heavy compute, large memory         |

---

## 📌 Summary

* Beyond Naive Bayes, **logistic regression and SVMs** provide strong baselines.
* **Neural networks** (CNNs, RNNs, Transformers) achieve **state-of-the-art**.
* Choice depends on **data size, resources, and task complexity**.

---

## References

* [Logistic Regression](https://en.wikipedia.org/wiki/Logistic_regression)
* [Support Vector Machine](https://en.wikipedia.org/wiki/Support_vector_machine)
* [Convolutional Neural Network](https://en.wikipedia.org/wiki/Convolutional_neural_network)
* [Recurrent Neural Network](https://en.wikipedia.org/wiki/Recurrent_neural_network)
* [BERT](https://en.wikipedia.org/wiki/BERT_%28language_model%29)
* [Word Embedding](https://en.wikipedia.org/wiki/Word_embedding)



# Natural Language Processing – Lecture 34

## 🗣 Sentiment Analysis – Introduction

This lecture introduces **Sentiment Analysis (SA)**, a popular NLP application that determines the **emotional tone or polarity** of text.

---

## 📖 What is Sentiment Analysis?

* **Definition:** Task of classifying text into **sentiment categories**.

* Typical categories:

  * **Positive**
  * **Negative**
  * **Neutral**

* Example:

  * *“I love this phone!”* → Positive
  * *“This service is terrible.”* → Negative

📖 [Sentiment Analysis (Wikipedia)](https://en.wikipedia.org/wiki/Sentiment_analysis)

---

## 🔑 Applications

1. **Product Reviews** → customer feedback monitoring.
2. **Social Media Analysis** → opinion mining on Twitter, Facebook.
3. **Political Sentiment** → election campaign analysis.
4. **Finance** → stock market sentiment analysis.
5. **Customer Support** → detecting frustration or satisfaction.

---

## ⚙️ Approaches to Sentiment Analysis

### 1. **Rule-Based Approaches**

* Use dictionaries of positive/negative words (lexicons).
* Example: "good, excellent" → positive; "bad, horrible" → negative.
* Limitation: ignores context, sarcasm.

📖 [Opinion Lexicon](https://en.wikipedia.org/wiki/Sentiment_analysis#Lexicon-based_approaches)

---

### 2. **Machine Learning Approaches**

* Treat sentiment classification as a **text classification** problem.
* Features: **bag-of-words, TF-IDF, n-grams**.
* Algorithms: Naive Bayes, Logistic Regression, SVMs.

📖 [Naive Bayes Text Classification](https://en.wikipedia.org/wiki/Naive_Bayes_text_classification)

---

### 3. **Neural Approaches**

* Use word embeddings + neural networks.
* CNNs and RNNs for sentiment detection.
* **Transformers (BERT, RoBERTa, DistilBERT)** → state-of-the-art.

📖 [BERT](https://en.wikipedia.org/wiki/BERT_%28language_model%29)

---

## ⚠️ Challenges in Sentiment Analysis

1. **Sarcasm & Irony**

   * *“Great, another delay in my flight!”* → Negative, not Positive.

2. **Domain Dependence**

   * "Unpredictable" → positive in movie review, negative in car review.

3. **Aspect-Based Sentiment Analysis (ABSA)**

   * Detect sentiment towards specific aspects.
   * Example: *“The camera is great, but the battery is awful.”*

     * Camera → Positive
     * Battery → Negative

📖 [Aspect-Based Sentiment Analysis](https://en.wikipedia.org/wiki/Sentiment_analysis#Aspect-based_sentiment_analysis)

---

## 📊 Evaluation Metrics

* **Accuracy**
* **Precision, Recall, F1-score**
* **Confusion Matrix**

---

## 📌 Summary

* Sentiment analysis = **detecting polarity (positive/negative/neutral)** in text.
* Approaches: **rule-based, ML, neural (transformers)**.
* Challenges: sarcasm, domain adaptation, aspect-level sentiment.
* Applications: **reviews, social media, politics, finance**.

---

## References

* [Sentiment Analysis](https://en.wikipedia.org/wiki/Sentiment_analysis)
* [Naive Bayes Text Classification](https://en.wikipedia.org/wiki/Naive_Bayes_text_classification)
* [BERT](https://en.wikipedia.org/wiki/BERT_%28language_model%29)
* [Aspect-Based Sentiment Analysis](https://en.wikipedia.org/wiki/Sentiment_analysis#Aspect-based_sentiment_analysis)



# Natural Language Processing – Lecture 35

## 🗣 Sentiment Analysis – Advanced Methods

This lecture expands on sentiment analysis, covering **fine-grained sentiment**, **aspect-based sentiment analysis (ABSA)**, and **state-of-the-art deep learning approaches**.

---

## 📖 Beyond Basic Sentiment Classification

* Traditional SA: **positive / negative / neutral**.
* Advanced tasks require **finer granularity**:

  * **5-star ratings** (Amazon, Yelp)
  * **Aspect-level sentiment** (camera quality vs battery life in a phone review)
  * **Emotion detection** (joy, anger, sadness, fear, surprise, disgust)

📖 [Emotion Recognition](https://en.wikipedia.org/wiki/Emotion_recognition)

---

## 🔑 Aspect-Based Sentiment Analysis (ABSA)

### Definition

* Detecting **sentiment towards specific aspects** of an entity.

### Example

Sentence: *“The camera is great, but the battery is awful.”*

* Aspect 1: Camera → Positive
* Aspect 2: Battery → Negative

📖 [Aspect-Based Sentiment Analysis](https://en.wikipedia.org/wiki/Sentiment_analysis#Aspect-based_sentiment_analysis)

---

## ⚙️ Approaches

### 1. **Lexicon + Rules**

* Use aspect lexicons (e.g., “camera”, “battery”) + sentiment words.
* Limitation: brittle, poor generalization.

---

### 2. **Machine Learning**

* Classify **(aspect, sentiment)** pairs.
* Features: n-grams, POS tags, dependency relations.

---

### 3. **Deep Learning**

* **CNNs** → capture local context.
* **RNNs / LSTMs** → capture long dependencies.
* **Attention Mechanisms** → focus on relevant aspect terms.

---

### 4. **Transformers**

* Models like **BERT, RoBERTa, DistilBERT** fine-tuned for ABSA.
* Capture both **aspect term** and **context sentiment**.
* Achieve **state-of-the-art results**.

📖 [BERT](https://en.wikipedia.org/wiki/BERT_%28language_model%29)

---

## ⚠️ Challenges

1. **Sarcasm & Irony**

   * *“The battery lasts forever… if forever means 2 hours.”*

2. **Implicit Aspects**

   * *“Too heavy to carry around”* → implicit aspect = weight.

3. **Domain Dependence**

   * “Unpredictable” → good for a movie, bad for a car.

4. **Multilingual Sentiment**

   * Requires handling multiple languages, cultural context.

---

## 📊 Applications

* **Product Reviews** → e-commerce (Amazon, Flipkart).
* **Social Media Monitoring** → brand reputation management.
* **Customer Feedback Analysis** → airlines, hotels, services.
* **Political Analysis** → sentiment on policies, leaders.

---

## 📌 Summary

* Advanced SA goes beyond polarity → includes **ratings, aspect-level, emotions**.
* ABSA: identifies sentiment towards **specific aspects**.
* Deep learning & transformers achieve **state-of-the-art performance**.
* Challenges: sarcasm, implicit aspects, domain & language variation.

---

## References

* [Sentiment Analysis](https://en.wikipedia.org/wiki/Sentiment_analysis)
* [Aspect-Based Sentiment Analysis](https://en.wikipedia.org/wiki/Sentiment_analysis#Aspect-based_sentiment_analysis)
* [Emotion Recognition](https://en.wikipedia.org/wiki/Emotion_recognition)
* [BERT](https://en.wikipedia.org/wiki/BERT_%28language_model%29)



# Natural Language Processing – Lecture 36

## 🤖 Question Answering (QA) – Introduction

This lecture introduces **Question Answering (QA)**, a key NLP application that enables systems to answer **natural language questions**.

---

## 📖 What is Question Answering?

* **Definition:** Task of automatically answering questions posed in natural language.
* Input: Question (e.g., *“Who founded Microsoft?”*)
* Output: Answer (e.g., *“Bill Gates and Paul Allen”*)

📖 [Question Answering (Wikipedia)](https://en.wikipedia.org/wiki/Question_answering)

---

## 🔑 Types of QA Systems

### 1. **Closed-Domain QA**

* Restricted to a specific domain.
* Example: Medical QA system (answers only health-related questions).

### 2. **Open-Domain QA**

* Works on any topic.
* Requires **large knowledge bases** or the **web** as a resource.

---

## 📊 QA System Architectures

### 1. **Information Retrieval (IR)-Based QA**

* Pipeline:

  1. Retrieve relevant documents (search engine).
  2. Extract sentences containing possible answers.
  3. Rank and return best candidate.

📖 [Information Retrieval](https://en.wikipedia.org/wiki/Information_retrieval)

---

### 2. **Knowledge-Based QA**

* Uses structured sources:

  * [Knowledge Graphs](https://en.wikipedia.org/wiki/Knowledge_Graph)
  * [DBpedia](https://en.wikipedia.org/wiki/DBpedia), [Wikidata](https://en.wikipedia.org/wiki/Wikidata)
* Example:

  * Q: *“Who is the CEO of Tesla?”*
  * A: *“Elon Musk”* (looked up in KB).

---

### 3. **Neural QA (Machine Reading Comprehension)**

* Uses deep learning to extract answers directly from text.
* Example: **SQuAD (Stanford Question Answering Dataset)** → extractive QA.
* Models: **BiDAF, BERT, RoBERTa, T5**.

📖 [SQuAD Dataset](https://en.wikipedia.org/wiki/Stanford_Question_Answering_Dataset)

---

## 🧩 Types of Questions

1. **Factoid Questions** → “When was Google founded?”
2. **List Questions** → “Which countries are in the EU?”
3. **Definition Questions** → “What is photosynthesis?”
4. **Why/How Questions** → require reasoning.

---

## ⚠️ Challenges in QA

1. **Ambiguity in Questions**

   * “Where is Apple headquartered?” (company vs fruit).

2. **Context Understanding**

   * Multi-sentence reasoning required.

3. **Answer Variability**

   * “Barack Obama” vs “Obama” vs “President Obama”.

4. **Commonsense & World Knowledge**

   * Many questions require knowledge beyond text.

📖 [Commonsense Reasoning](https://en.wikipedia.org/wiki/Commonsense_reasoning)

---

## 📌 Summary

* QA = answering natural language questions.
* Approaches: **IR-based, KB-based, neural (reading comprehension)**.
* Types: closed-domain vs open-domain QA.
* Challenges: ambiguity, reasoning, variability, commonsense knowledge.

---

## References

* [Question Answering](https://en.wikipedia.org/wiki/Question_answering)
* [Information Retrieval](https://en.wikipedia.org/wiki/Information_retrieval)
* [Knowledge Graph](https://en.wikipedia.org/wiki/Knowledge_Graph)
* [DBpedia](https://en.wikipedia.org/wiki/DBpedia)
* [Wikidata](https://en.wikipedia.org/wiki/Wikidata)
* [SQuAD Dataset](https://en.wikipedia.org/wiki/Stanford_Question_Answering_Dataset)
* [Commonsense Reasoning](https://en.wikipedia.org/wiki/Commonsense_reasoning)



# Natural Language Processing – Lecture 37

## 🤖 Question Answering (QA) – Advanced Approaches

This lecture expands on QA, focusing on **deep learning methods**, **transformers**, and **reasoning challenges** in modern QA systems.

---

## 🧠 Neural Question Answering

### 1. **Extractive QA**

* Answer is a **span of text** in a passage.
* Example:

  * Q: *“Who discovered penicillin?”*
  * Passage: *“… Alexander Fleming discovered penicillin in 1928 …”*
  * Answer: *“Alexander Fleming”*
* Models: BiDAF, BERT, RoBERTa, ALBERT.

📖 [Extractive QA](https://en.wikipedia.org/wiki/Question_answering#Extractive_question_answering)

---

### 2. **Abstractive QA**

* Generates answers in **natural language** rather than extracting spans.
* Example:

  * Q: *“What is penicillin used for?”*
  * Answer: *“Penicillin is used to treat bacterial infections.”*
* Models: Seq2Seq, T5, GPT.

📖 [Abstractive QA](https://en.wikipedia.org/wiki/Question_answering#Abstractive_question_answering)

---

## 🔹 Transformer-Based QA

* **BERT (2018)** → fine-tuned for extractive QA (SQuAD benchmark).
* **RoBERTa, ALBERT, DistilBERT** → improved variations.
* **T5, GPT models** → used for generative QA.
* These models revolutionized QA with **contextual embeddings**.

📖 [BERT](https://en.wikipedia.org/wiki/BERT_%28language_model%29)
📖 [GPT](https://en.wikipedia.org/wiki/GPT-3)

---

## 🧩 Multi-Hop Question Answering

* Some questions require **reasoning across multiple sentences/documents**.
* Example:

  * Q: *“Where was the wife of Barack Obama born?”*
  * Requires:

    1. Identify wife of Obama → Michelle Obama.
    2. Find birthplace of Michelle Obama → Chicago.

📖 [Multi-hop QA](https://en.wikipedia.org/wiki/Question_answering#Multi-hop_question_answering)

---

## ⚙️ Commonsense and Knowledge-Augmented QA

* Many questions require **commonsense reasoning** not present in text.
* Example:

  * Q: *“Can a penguin fly?”*
  * Needs world knowledge: Penguins are birds but cannot fly.
* Solutions:

  * Integrating **knowledge graphs** (Wikidata, ConceptNet).
  * Hybrid models combining **neural + symbolic reasoning**.

📖 [Commonsense Knowledge](https://en.wikipedia.org/wiki/Commonsense_reasoning)

---

## 📊 Challenges in Modern QA

1. **Ambiguity**

   * "Where is Washington?" → state, city, or person?
2. **Answer Variability**

   * “Obama”, “Barack Obama”, “President Obama”.
3. **Reasoning Requirements**

   * Temporal reasoning: *“Who was president of the US in 2008?”*
4. **Domain Adaptation**

   * Web-trained models may fail on biomedical or legal QA.

---

## 📌 Summary

* Modern QA uses **deep learning & transformers**.
* Two types: **Extractive QA** (span prediction) and **Abstractive QA** (answer generation).
* Advanced challenges: **multi-hop reasoning, commonsense, domain adaptation**.
* Hybrid approaches integrate **neural models with knowledge bases**.

---

## References

* [Question Answering](https://en.wikipedia.org/wiki/Question_answering)
* [Extractive QA](https://en.wikipedia.org/wiki/Question_answering#Extractive_question_answering)
* [Abstractive QA](https://en.wikipedia.org/wiki/Question_answering#Abstractive_question_answering)
* [BERT](https://en.wikipedia.org/wiki/BERT_%28language_model%29)
* [GPT-3](https://en.wikipedia.org/wiki/GPT-3)
* [Multi-hop QA](https://en.wikipedia.org/wiki/Question_answering#Multi-hop_question_answering)
* [Commonsense Reasoning](https://en.wikipedia.org/wiki/Commonsense_reasoning)



# Natural Language Processing – Lecture 38

## 🗣 Dialogue Systems – Introduction

This lecture introduces **Dialogue Systems (Conversational Agents)**, which interact with humans using **natural language**.

---

## 📖 What are Dialogue Systems?

* **Definition:** Computer systems designed to converse with humans in natural language.
* Examples:

  * Virtual Assistants → Siri, Alexa, Google Assistant.
  * Chatbots → Customer support bots, FAQ assistants.

📖 [Dialogue System (Wikipedia)](https://en.wikipedia.org/wiki/Dialogue_system)

---

## 🔑 Types of Dialogue Systems

### 1. **Task-Oriented Systems**

* Goal: Complete a **specific task**.
* Examples:

  * Book a flight.
  * Order food.
  * Schedule a meeting.

📖 [Task-Oriented Dialogue](https://en.wikipedia.org/wiki/Dialogue_system#Task-oriented_dialogue)

---

### 2. **Open-Domain (Chatbots)**

* General conversation, no fixed task.
* Examples:

  * ChatGPT, Replika.
* More challenging → requires commonsense + world knowledge.

📖 [Chatbot](https://en.wikipedia.org/wiki/Chatbot)

---

## ⚙️ Architecture of Dialogue Systems

### 1. **Traditional Pipeline**

* **Automatic Speech Recognition (ASR)** → converts speech → text.
* **Natural Language Understanding (NLU)** → extracts meaning.
* **Dialogue Manager (DM)** → decides system response.
* **Natural Language Generation (NLG)** → generates response text.
* **Text-to-Speech (TTS)** → converts text → speech.

📖 [Speech Recognition](https://en.wikipedia.org/wiki/Speech_recognition)
📖 [Natural Language Generation](https://en.wikipedia.org/wiki/Natural_language_generation)

---

### 2. **End-to-End Neural Dialogue Systems**

* Use deep learning to **map input directly to output**.
* Sequence-to-sequence models, Transformers (e.g., GPT).
* Advantages: Flexible, less manual design.
* Challenges: Lack of controllability, factual errors.

📖 [Transformer Model](https://en.wikipedia.org/wiki/Transformer_%28machine_learning_model%29)

---

## 🧩 Dialogue Manager (DM)

* Central component in pipeline-based systems.
* Handles:

  * **Dialogue State Tracking (DST)** → tracks context, slot values.
  * **Policy Learning** → decides next action.
* Approaches:

  * Rule-based.
  * Reinforcement learning (train policy to maximize success).

📖 [Dialogue State Tracking](https://en.wikipedia.org/wiki/Dialogue_state_tracking)

---

## ⚠️ Challenges in Dialogue Systems

1. **Ambiguity & Context**

   * "Book a ticket to Paris" → which Paris (France, Texas)?

2. **Long-Term Context**

   * Maintaining memory over multiple turns.

3. **Personalization**

   * Adapting to user preferences.

4. **Safety & Bias**

   * Avoiding toxic, biased, or harmful responses.

📖 [Bias in AI](https://en.wikipedia.org/wiki/Algorithmic_bias)

---

## 📊 Applications

* Virtual Assistants (Alexa, Siri, Google Assistant).
* Customer Service (banking, e-commerce, travel).
* Healthcare (symptom checkers, medical chatbots).
* Education (tutoring systems).

---

## 📌 Summary

* Dialogue systems enable **human–machine conversation**.
* Two main types: **task-oriented** and **open-domain**.
* Architectures: **pipeline-based** vs **end-to-end neural**.
* Challenges: ambiguity, context, personalization, safety.
* Widely used in **assistants, customer support, healthcare, education**.

---

## References

* [Dialogue System](https://en.wikipedia.org/wiki/Dialogue_system)
* [Task-Oriented Dialogue](https://en.wikipedia.org/wiki/Dialogue_system#Task-oriented_dialogue)
* [Chatbot](https://en.wikipedia.org/wiki/Chatbot)
* [Speech Recognition](https://en.wikipedia.org/wiki/Speech_recognition)
* [Natural Language Generation](https://en.wikipedia.org/wiki/Natural_language_generation)
* [Transformer Model](https://en.wikipedia.org/wiki/Transformer_%28machine_learning_model%29)
* [Dialogue State Tracking](https://en.wikipedia.org/wiki/Dialogue_state_tracking)
* [Algorithmic Bias](https://en.wikipedia.org/wiki/Algorithmic_bias)




# Natural Language Processing – Lecture 39

## 🗣 Dialogue Systems – Advanced Approaches

This lecture explores **advanced methods for dialogue systems**, including **neural models, reinforcement learning, and evaluation challenges**.

---

## 🧠 Neural Dialogue Systems

### 1. **Sequence-to-Sequence Models**

* Encode input utterance → generate response.
* Based on **RNNs, LSTMs, GRUs**.
* Limitation: often produce **generic responses** (*“I don’t know”*).

📖 [Seq2Seq Models](https://en.wikipedia.org/wiki/Sequence-to-sequence_model)

---

### 2. **Transformer-Based Models**

* **GPT (OpenAI)** → autoregressive transformer for open-domain dialogue.
* **BERT-like models** → used for dialogue understanding (NLU tasks).
* **DialogPT, BlenderBot, LaMDA, ChatGPT** → large-scale conversational models.

📖 [GPT](https://en.wikipedia.org/wiki/GPT-3)
📖 [LaMDA](https://en.wikipedia.org/wiki/LaMDA)

---

## ⚙️ Reinforcement Learning in Dialogue

* Dialogue framed as a **sequential decision-making process**.
* **Dialogue Policy Learning**: select next system action to maximize long-term success.
* **Reward Signals**:

  * Task success (e.g., booking completed).
  * User satisfaction.
* Algorithms: Q-learning, Policy Gradients, Deep RL.

📖 [Reinforcement Learning](https://en.wikipedia.org/wiki/Reinforcement_learning)

---

## 🧩 Dialogue State Tracking (DST)

* Maintain context across turns.
* Example:

  * User: “Book me a flight to Paris.”
  * System: (fills slot: destination=Paris).
  * User: “Actually, make that London.” → system updates slot.

📖 [Dialogue State Tracking](https://en.wikipedia.org/wiki/Dialogue_state_tracking)

---

## 📊 Evaluation of Dialogue Systems

### 1. **Automatic Metrics**

* BLEU, ROUGE, METEOR → borrowed from MT and summarization.
* Limitation: correlate poorly with human judgment.

### 2. **Human Evaluation**

* Fluency, coherence, relevance, task success.
* Expensive but more reliable.

📖 [BLEU](https://en.wikipedia.org/wiki/BLEU)

---

## ⚠️ Challenges

1. **Generic Responses**

   * Seq2Seq models produce safe but boring replies.

2. **Long-Term Coherence**

   * Hard to maintain consistent persona/context.

3. **Commonsense & Knowledge Integration**

   * Requires external knowledge bases (e.g., Wikidata, ConceptNet).

4. **Ethics & Safety**

   * Avoid toxic, biased, or misleading answers.

📖 [Commonsense Reasoning](https://en.wikipedia.org/wiki/Commonsense_reasoning)

---

## 📌 Summary

* Advanced dialogue systems use **seq2seq, transformers, and RL**.
* **DST** ensures tracking of user goals across dialogue.
* Evaluation remains hard → human judgment is gold standard.
* Challenges: **generic responses, coherence, knowledge, ethics**.

---

## References

* [Seq2Seq Models](https://en.wikipedia.org/wiki/Sequence-to-sequence_model)
* [GPT](https://en.wikipedia.org/wiki/GPT-3)
* [LaMDA](https://en.wikipedia.org/wiki/LaMDA)
* [Reinforcement Learning](https://en.wikipedia.org/wiki/Reinforcement_learning)
* [Dialogue State Tracking](https://en.wikipedia.org/wiki/Dialogue_state_tracking)
* [BLEU](https://en.wikipedia.org/wiki/BLEU)
* [Commonsense Reasoning](https://en.wikipedia.org/wiki/Commonsense_reasoning)




# Natural Language Processing – Lecture 40

## 📖 Machine Translation – Introduction

This lecture introduces **Machine Translation (MT)**, the task of automatically translating text from one language to another.

---

## 🌍 What is Machine Translation?

* **Definition:** Converting text (or speech) in a **source language** into a **target language** using computational methods.
* Example:

  * Input: *“Bonjour le monde”* (French)
  * Output: *“Hello world”* (English)

📖 [Machine Translation (Wikipedia)](https://en.wikipedia.org/wiki/Machine_translation)

---

## 🔑 Types of Machine Translation

### 1. **Rule-Based MT (RBMT)**

* Uses **linguistic rules and dictionaries**.
* Components:

  * Morphological analysis
  * Syntax transfer rules
  * Generation rules
* Example: **Systran** (used by early Google Translate).

📖 [Rule-Based MT](https://en.wikipedia.org/wiki/Rule-based_machine_translation)

---

### 2. **Statistical MT (SMT)**

* Learns **translation probabilities** from parallel corpora.
* Example:

  * *IBM Models, Phrase-Based SMT (Moses system)*.
* Limitation: struggles with long-distance dependencies.

📖 [Statistical Machine Translation](https://en.wikipedia.org/wiki/Statistical_machine_translation)

---

### 3. **Neural MT (NMT)**

* Uses deep learning models (RNNs, LSTMs, Transformers).
* End-to-end learning: directly maps source → target.
* Example: Google Translate now uses NMT.

📖 [Neural Machine Translation](https://en.wikipedia.org/wiki/Neural_machine_translation)

---

## ⚙️ Challenges in MT

1. **Ambiguity**

   * Word “bank” → financial institution or riverbank.

2. **Word Order**

   * Different across languages (English vs Japanese).

3. **Idioms**

   * *“Kick the bucket”* → cannot be translated literally.

4. **Low-Resource Languages**

   * Lack of parallel corpora (e.g., African languages).

5. **Morphologically Rich Languages**

   * Turkish, Finnish → complex word forms.

📖 [Linguistic Typology](https://en.wikipedia.org/wiki/Linguistic_typology)

---

## 📊 Evaluation of MT

### 1. **Automatic Metrics**

* **BLEU (Bilingual Evaluation Understudy)**
* **METEOR, ROUGE, TER**
* Pros: fast, scalable.
* Cons: don’t always align with human judgment.

📖 [BLEU](https://en.wikipedia.org/wiki/BLEU)

### 2. **Human Evaluation**

* Fluency, adequacy, fidelity.
* More reliable, but costly.

---

## ⚡ Applications of MT

* **Cross-lingual communication** (Google Translate, DeepL).
* **Multilingual information retrieval**.
* **International business & diplomacy**.
* **Assistive technology** (helping low-resource language speakers).

---

## 📌 Summary

* MT = automatic translation between languages.
* Approaches: **RBMT, SMT, NMT**.
* Modern systems dominated by **neural MT (Transformers)**.
* Challenges: ambiguity, word order, idioms, low-resource languages.
* Evaluation: automatic metrics (BLEU, ROUGE) + human judgment.

---

## References

* [Machine Translation](https://en.wikipedia.org/wiki/Machine_translation)
* [Rule-Based MT](https://en.wikipedia.org/wiki/Rule-based_machine_translation)
* [Statistical MT](https://en.wikipedia.org/wiki/Statistical_machine_translation)
* [Neural MT](https://en.wikipedia.org/wiki/Neural_machine_translation)
* [BLEU](https://en.wikipedia.org/wiki/BLEU)
* [Linguistic Typology](https://en.wikipedia.org/wiki/Linguistic_typology)



# Natural Language Processing – Lecture 41

## 🌍 Machine Translation – Statistical Approaches

This lecture dives deeper into **Statistical Machine Translation (SMT)**, which was dominant before the rise of neural MT.

---

## 📖 What is Statistical Machine Translation?

* **Definition:** Translation based on **probabilistic models** learned from large **parallel corpora** (sentence-aligned bilingual data).
* Idea: Choose the target sentence $e$ that maximizes:

$$
\hat{e} = \arg\max_e P(e|f) = \arg\max_e P(f|e) \cdot P(e)
$$

* $f$ = source sentence
* $e$ = target sentence
* $P(f|e)$ = translation model
* $P(e)$ = language model

📖 [Statistical Machine Translation (Wikipedia)](https://en.wikipedia.org/wiki/Statistical_machine_translation)

---

## 🔑 Components of SMT

### 1. **Translation Model**

* Learns word/phrase alignments from parallel corpora.
* Example: IBM Models (word-based), Phrase-based SMT.

📖 [IBM Alignment Models](https://en.wikipedia.org/wiki/IBM_alignment_models)

---

### 2. **Language Model**

* Ensures fluency of target text.
* Typically n-gram models with smoothing.

📖 [Language Model](https://en.wikipedia.org/wiki/Language_model)

---

### 3. **Decoder**

* Searches for the most probable target sentence.
* Uses beam search or heuristic algorithms.

📖 [Machine Translation Decoding](https://en.wikipedia.org/wiki/Machine_translation#Decoding)

---

## 🧩 Phrase-Based SMT

* Extension of word-based models.
* Uses **phrases (sequences of words)** as translation units.
* Example:

  * French: “à la maison”
  * English: “at home”

📖 [Phrase-Based SMT](https://en.wikipedia.org/wiki/Statistical_machine_translation#Phrase-based_translation)

---

## ⚡ Strengths of SMT

* Requires only parallel corpora (no linguistic rules).
* More fluent than early rule-based systems.
* Scalable across languages.

---

## ⚠️ Limitations of SMT

1. **Word Alignment Errors**

   * Difficulty in aligning long-distance dependencies.

2. **Idioms & Context**

   * Literal translation of idiomatic expressions.

3. **Data Hungry**

   * Requires large high-quality parallel corpora.

4. **Sentence-Level Translation**

   * Ignores document-level context.

---

## 📊 Evaluation

* Same as general MT:

  * Automatic → BLEU, METEOR, TER.
  * Human → Fluency, adequacy.

📖 [BLEU](https://en.wikipedia.org/wiki/BLEU)

---

## 📌 Summary

* SMT = **probabilistic translation framework**.
* Components: **translation model, language model, decoder**.
* Phrase-based SMT improved fluency over word-based SMT.
* Limitations: data-hungry, poor idiom handling, lacks context.
* Largely replaced by **Neural MT**.

---

## References

* [Statistical Machine Translation](https://en.wikipedia.org/wiki/Statistical_machine_translation)
* [IBM Alignment Models](https://en.wikipedia.org/wiki/IBM_alignment_models)
* [Language Model](https://en.wikipedia.org/wiki/Language_model)
* [BLEU](https://en.wikipedia.org/wiki/BLEU)



# Natural Language Processing – Lecture 42

## 🌍 Machine Translation – Neural Approaches

This lecture introduces **Neural Machine Translation (NMT)**, the dominant paradigm in modern MT, replacing SMT.

---

## 📖 What is Neural Machine Translation?

* **Definition:** An **end-to-end neural model** that directly maps a source sentence to a target sentence.
* Uses **deep learning architectures** (RNNs, CNNs, Transformers).
* Learns translation without explicitly modeling separate **translation + language models** (unlike SMT).

📖 [Neural Machine Translation (Wikipedia)](https://en.wikipedia.org/wiki/Neural_machine_translation)

---

## ⚙️ Sequence-to-Sequence (Seq2Seq) Model

* Introduced in 2014 (Sutskever et al., Google).
* Components:

  1. **Encoder** → reads source sentence.
  2. **Decoder** → generates target sentence.
* Both encoder and decoder are typically **RNNs (LSTMs/GRUs)**.

📖 [Sequence-to-Sequence Model](https://en.wikipedia.org/wiki/Sequence-to-sequence_model)

---

## 🔑 Attention Mechanism

* Introduced by Bahdanau et al. (2015).
* Instead of encoding the entire source into a single vector, the model **attends** to different source words at each decoding step.
* Greatly improved translation quality.

📖 [Attention Mechanism](https://en.wikipedia.org/wiki/Attention_%28machine_learning%29)

---

## 🌐 Transformer Model

* Introduced by Vaswani et al. (2017) in *“Attention is All You Need”*.
* Replaced RNNs with **self-attention layers**.
* Advantages:

  * Better at handling long sentences.
  * Parallelizable → faster training.
  * State-of-the-art in MT (used in Google Translate, DeepL).

📖 [Transformer Model](https://en.wikipedia.org/wiki/Transformer_%28machine_learning_model%29)

---

## ⚡ Strengths of NMT

1. Produces **fluent, natural translations**.
2. Learns **long-distance dependencies** (via attention/transformers).
3. End-to-end learning (no hand-crafted features).
4. Adaptable to **low-resource settings** with transfer learning.

---

## ⚠️ Limitations of NMT

1. **Data Hungry**

   * Requires very large parallel corpora.

2. **Rare Words**

   * Out-of-vocabulary issues (solved with **subword models like BPE**).

3. **Domain Adaptation**

   * Struggles when applied to a new domain.

4. **Hallucinations**

   * Sometimes generates fluent but incorrect translations.

📖 [Byte Pair Encoding](https://en.wikipedia.org/wiki/Byte_pair_encoding)

---

## 📊 Evaluation

* Same as SMT: BLEU, METEOR, ROUGE, TER.
* Human evaluation still necessary for fluency and adequacy.

📖 [BLEU](https://en.wikipedia.org/wiki/BLEU)

---

## 📌 Summary

* NMT = **end-to-end deep learning approach** to translation.
* Started with **Seq2Seq (RNNs)** → improved with **Attention** → now dominated by **Transformers**.
* Strengths: fluency, long-distance dependencies.
* Weaknesses: requires large data, struggles with rare words & domain shift.

---

## References

* [Neural Machine Translation](https://en.wikipedia.org/wiki/Neural_machine_translation)
* [Sequence-to-Sequence Model](https://en.wikipedia.org/wiki/Sequence-to-sequence_model)
* [Attention Mechanism](https://en.wikipedia.org/wiki/Attention_%28machine_learning%29)
* [Transformer Model](https://en.wikipedia.org/wiki/Transformer_%28machine_learning_model%29)
* [Byte Pair Encoding](https://en.wikipedia.org/wiki/Byte_pair_encoding)
* [BLEU](https://en.wikipedia.org/wiki/BLEU)



# Natural Language Processing – Lecture 43

## 🌍 Machine Translation – Advanced Topics

This lecture explores **advanced challenges and techniques** in modern **Neural Machine Translation (NMT)**.

---

## ⚠️ Challenges in NMT

### 1. **Low-Resource Languages**

* NMT requires large parallel corpora.
* Many languages lack sufficient data (e.g., African, Indigenous languages).
* Solutions:

  * Transfer learning from high-resource languages.
  * Multilingual NMT.

📖 [Low-Resource MT](https://en.wikipedia.org/wiki/Neural_machine_translation#Low-resource_languages)

---

### 2. **Domain Adaptation**

* NMT trained on news may fail on medical or legal text.
* Approaches:

  * Fine-tuning on domain-specific data.
  * Back-translation with synthetic data.

📖 [Domain Adaptation in NLP](https://en.wikipedia.org/wiki/Domain_adaptation)

---

### 3. **Handling Rare & OOV Words**

* Problem: Words not seen in training.
* Solutions:

  * **Subword models** (Byte Pair Encoding, SentencePiece).
  * **Character-level models** for morphologically rich languages.

📖 [Byte Pair Encoding](https://en.wikipedia.org/wiki/Byte_pair_encoding)

---

### 4. **Long Sentences & Context**

* NMT struggles with very long input sequences.
* Solutions:

  * Transformers with self-attention handle better than RNNs.
  * Document-level NMT (beyond sentence-level).

📖 [Transformer Model](https://en.wikipedia.org/wiki/Transformer_%28machine_learning_model%29)

---

## 🔑 Advanced Techniques

### 1. **Multilingual NMT**

* Train one model for multiple languages.
* Enables **zero-shot translation** (translate between languages without direct parallel data).
* Example: Google’s multilingual NMT system.

📖 [Multilingual NMT](https://en.wikipedia.org/wiki/Multilingual_neural_machine_translation)

---

### 2. **Back-Translation**

* Generate synthetic parallel data by translating monolingual target data back into the source language.
* Improves low-resource translation.

📖 [Back-Translation in MT](https://en.wikipedia.org/wiki/Neural_machine_translation#Back-translation)

---

### 3. **Pretrained Language Models for MT**

* Use large pretrained models (BERT, mBERT, mT5, XLM-R).
* Improve fluency and cross-lingual transfer.

📖 [mBERT](https://en.wikipedia.org/wiki/BERT_%28language_model%29#Multilingual_BERT)

---

### 4. **Evaluation Beyond BLEU**

* BLEU has limitations → does not capture semantic similarity.
* Alternatives:

  * **METEOR, chrF, COMET, BERTScore**.

📖 [BERTScore](https://arxiv.org/abs/1904.09675)

---

## 📊 Applications of Advanced MT

* **Cross-lingual Information Retrieval**
* **International Business & Diplomacy**
* **Healthcare** (translating medical records, instructions)
* **Education** (access to resources in multiple languages)

---

## 📌 Summary

* Advanced MT must handle **low-resource languages, domain adaptation, rare words, long contexts**.
* Techniques: **multilingual NMT, back-translation, pretrained LMs, document-level MT**.
* Evaluation evolving beyond BLEU to **semantic metrics** (COMET, BERTScore).

---

## References

* [Neural Machine Translation](https://en.wikipedia.org/wiki/Neural_machine_translation)
* [Low-Resource MT](https://en.wikipedia.org/wiki/Neural_machine_translation#Low-resource_languages)
* [Domain Adaptation](https://en.wikipedia.org/wiki/Domain_adaptation)
* [Byte Pair Encoding](https://en.wikipedia.org/wiki/Byte_pair_encoding)
* [Transformer Model](https://en.wikipedia.org/wiki/Transformer_%28machine_learning_model%29)
* [Multilingual NMT](https://en.wikipedia.org/wiki/Multilingual_neural_machine_translation)
* [BERTScore Paper](https://arxiv.org/abs/1904.09675)

