

# Natural Language Processing - Lecture Notes

## Course Overview

**Instructor:** Prof. Pawan Goyal
**Institution:** [IIT Kharagpur](https://www.iitkgp.ac.in/)
**Teaching Assistants:** Amrith Krishna, Mayank Singh

This 12-week course covers the theoretical and practical aspects of [Natural Language Processing (NLP)](https://en.wikipedia.org/wiki/Natural_language_processing), combining foundational algorithms with real-world applications.

---

## Week 1

### Lecture 1: Introduction to the Course

* Course structure: 12 weeks, 5 modules per week.
* Key textbooks:

  * [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/) by Jurafsky & Martin.
  * [Foundations of Statistical Natural Language Processing](https://mitpress.mit.edu/9780262133609/) by Manning & Schütze.
* Tools and resources:

  * Lecture slides
  * Python & Jupyter Notebooks for hands-on tasks
* Evaluation:

  * 25% Assignments (weekly, including coding tasks)
  * 75% Final Exam
* Topics covered:

  * [Text Processing](https://en.wikipedia.org/wiki/Text_processing)
  * [Language Modeling](https://en.wikipedia.org/wiki/Language_model)
  * [Morphology](https://en.wikipedia.org/wiki/Morphology_%28linguistics%29)
  * [Syntax and Parsing](https://en.wikipedia.org/wiki/Syntactic_parsing)
  * [Semantics](https://en.wikipedia.org/wiki/Semantics)
  * [Word Embeddings](https://en.wikipedia.org/wiki/Word_embedding)
  * [Topic Modeling](https://en.wikipedia.org/wiki/Topic_model)
  * Applications: Entity Linking, Text Summarization, Sentiment Analysis

---

### Lecture 2: What Do We Do in NLP?

#### Goals of NLP

* **Scientific Goal:** Teach computers to understand and process human language.
* **Engineering Goal:** Build systems to analyze text for real-world applications.

#### Applications

* [Machine Translation](https://en.wikipedia.org/wiki/Machine_translation)
* [Spelling Correction](https://en.wikipedia.org/wiki/Spell_checker)
* [Search Engine Query Completion](https://en.wikipedia.org/wiki/Autocomplete)
* [Information Extraction](https://en.wikipedia.org/wiki/Information_extraction)
* Domain-specific chatbots (e.g., in education, customer service)
* [Sentiment Analysis](https://en.wikipedia.org/wiki/Sentiment_analysis)
* [Spam Detection](https://en.wikipedia.org/wiki/Email_spam#Spam_filtering)
* [Text Summarization](https://en.wikipedia.org/wiki/Automatic_summarization)

---

### Lecture 3: Why is NLP Hard?

#### Challenges

* **Lexical Ambiguity:** One word, multiple meanings (e.g., *will will Will’s will*).
* **Structural Ambiguity:** Multiple interpretations of the sentence structure.
* **Vagueness and Imprecision:** Context-dependent expressions (e.g., *it's very warm here*).
* **Idioms and Non-standard Usage:** Meaning not deducible from words (e.g., *burn the midnight oil*).
* **Evolving Language:** Introduction of new words, e.g., *unfriend*, *retweet*.
* **Multilinguality:** Need for language detection and translation across diverse languages.
* **Social Media Noise:** Abbreviations, hashtags, mentions (e.g., *CU L8R*).

#### Examples

* Famous translation errors by companies like [Pepsi](https://en.wikipedia.org/wiki/Pepsi#Marketing) and [KFC](https://en.wikipedia.org/wiki/KFC#Advertising).
* The failure of Microsoft's [Tay chatbot](https://en.wikipedia.org/wiki/Tay_%28bot%29).

---

### Lecture 4: Empirical Laws

#### Content vs. Function Words

* **Function Words:** Articles, pronouns, prepositions (e.g., *the, and, to*).
* **Content Words:** Nouns, verbs (e.g., *dog, run, beauty*).

#### Type vs. Token

* **Token:** Each word occurrence
* **Type:** Unique words
* **Type-Token Ratio (TTR):**

$$
  \text{TTR} = \frac{\text{Number of Types}}{\text{Number of Tokens}}
$$

  Indicates lexical diversity.

#### Examples

* **Tom Sawyer corpus**:

  * Tokens: 71,370
  * Types: 8,018
  * TTR ≈ 0.112
* **Shakespeare's complete works**:

  * Higher TTR due to greater lexical variety.

---

## References

* [Natural Language Processing (Wikipedia)](https://en.wikipedia.org/wiki/Natural_language_processing)
* [Speech and Language Processing (Jurafsky & Martin)](https://web.stanford.edu/~jurafsky/slp3/)
* [Foundations of Statistical NLP (Manning & Schütze)](https://mitpress.mit.edu/9780262133609/)
* [Word Embedding](https://en.wikipedia.org/wiki/Word_embedding)
* [Topic Modeling](https://en.wikipedia.org/wiki/Topic_model)
* [Machine Translation](https://en.wikipedia.org/wiki/Machine_translation)
* [Information Extraction](https://en.wikipedia.org/wiki/Information_extraction)
* [Sentiment Analysis](https://en.wikipedia.org/wiki/Sentiment_analysis)
* [Autocomplete](https://en.wikipedia.org/wiki/Autocomplete)
* [Tay Bot (Microsoft)](https://en.wikipedia.org/wiki/Tay_%28bot%29)
* [Pepsi Marketing](https://en.wikipedia.org/wiki/Pepsi#Marketing)
* [KFC Advertising](https://en.wikipedia.org/wiki/KFC#Advertising)



### Lecture 5: Text Processing - Basics

#### Key Concepts

* **Text Normalization:** Preparing raw text for analysis.

  * Lowercasing
  * Removing punctuation
  * Removing stopwords
* **Tokenization:** Splitting text into meaningful units (tokens)

  * Word-level
  * Sentence-level
* **Stemming:** Reducing words to their root forms (e.g., *running* → *run*)

  * [Porter Stemmer](https://tartarus.org/martin/PorterStemmer/)
* **Lemmatization:** Converting words to their base or dictionary form (e.g., *better* → *good*)

  * Requires [POS tagging](https://en.wikipedia.org/wiki/Part-of-speech_tagging)
* **Regular Expressions (RegEx):** Useful for pattern matching and basic text cleaning

#### Tools

* [NLTK](https://www.nltk.org/)
* [spaCy](https://spacy.io/)
* [re (Python RegEx)](https://docs.python.org/3/library/re.html)

---

## Week 2

### Lecture 6: Spelling Correction - Edit Distance

#### Edit Distance

* Measures dissimilarity between two strings
* **Levenshtein Distance:** Minimum number of insertions, deletions, and substitutions required
* Applications:

  * Spell checking
  * [DNA sequence alignment](https://en.wikipedia.org/wiki/Sequence_alignment)

### Lecture 7: Weighted Edit Distance and Variations

* Different costs assigned to insertions, deletions, and substitutions
* Based on:

  * Keyboard distance
  * Phonetic similarity
* **Damerau-Levenshtein Distance:** Adds transposition (swap) as an operation
* Application:

  * More accurate spelling correction systems

### Lecture 8: Noisy Channel Model for Spelling Correction

#### Noisy Channel Framework

* Common in NLP problems like spelling correction and machine translation

* Goal: Find the most likely original sentence $S$ given the observed sentence $O$

$$
  \arg\max_S P(S|O) = \arg\max_S P(O|S) P(S)
$$

* Components:

  * **Language Model** $P(S)$
  * **Error Model** $P(O|S)$

#### Applications

* OCR post-processing
* Auto-correct
* Machine translation ([IBM Models](https://en.wikipedia.org/wiki/IBM_models))

---

## References

* [Tokenization](https://en.wikipedia.org/wiki/Tokenization_%28lexical_analysis%29)
* [Stemming](https://en.wikipedia.org/wiki/Stemming)
* [Lemmatization](https://en.wikipedia.org/wiki/Lemmatisation)
* [Regular Expression (RegEx)](https://en.wikipedia.org/wiki/Regular_expression)
* [Levenshtein Distance](https://en.wikipedia.org/wiki/Levenshtein_distance)
* [Noisy Channel Model](https://en.wikipedia.org/wiki/Noisy-channel_model)
* [Damerau-Levenshtein Distance](https://en.wikipedia.org/wiki/Damerau%E2%80%93Levenshtein_distance)



### Lecture 9: N-Gram Language Models

#### Key Concepts

* **N-Gram Model:** A type of [statistical language model](https://en.wikipedia.org/wiki/N-gram) that uses the conditional probability of a word given the preceding $n - 1$ words:

$$
  P(w_n | w_{n-1}, ..., w_{1})
$$

* **Common N-Grams:**

  * Unigram: $P(w)$
  * Bigram: $P(w_n | w_{n-1})$
  * Trigram: $P(w_n | w_{n-1}, w_{n-2})$

#### Assumptions

* **Markov Assumption:** Only the most recent $n - 1$ words matter.
* **Chain Rule of Probability** is used for decomposition.

#### Issues

* **Data Sparsity**
* **Zero Probabilities**

---

### Lecture 10: Evaluation of Language Models & Basic Smoothing

#### Evaluation Metrics

* **Perplexity:** Measures how well a probability model predicts a sample.

$$
  \text{Perplexity} = 2^{-\frac{1}{N} \sum \log_2 P(w_i)}
$$

* Lower perplexity indicates a better model.

#### Smoothing Techniques

* **Add-One (Laplace) Smoothing**
* **Add-k Smoothing**
* **Good-Turing Discounting**: Estimates the probability of unseen events
* **Backoff and Interpolation**: Combine higher and lower-order models

[More on Smoothing](https://en.wikipedia.org/wiki/Smoothing)

---

### Lecture 11: Tutorial I

#### Overview

* Hands-on practice covering:

  * Implementing n-gram models
  * Evaluating language models
  * Applying smoothing techniques
* Python-based tutorials using:

  * [NLTK](https://www.nltk.org/)
  * [Jupyter Notebooks](https://jupyter.org/)

---

## References

* [N-Gram](https://en.wikipedia.org/wiki/N-gram)
* [Markov Assumption](https://en.wikipedia.org/wiki/Markov_property)
* [Chain Rule (Probability)](https://en.wikipedia.org/wiki/Chain_rule_%28probability%29)
* [Perplexity](https://en.wikipedia.org/wiki/Perplexity)
* [Smoothing (NLP)](https://en.wikipedia.org/wiki/Smoothing)


## Week 3

### Lecture 12: Advanced Smoothing Models

#### Advanced Techniques

* **Kneser-Ney Smoothing:**

  * State-of-the-art for language modeling
  * Adjusts probabilities based on the diversity of contexts a word appears in

* **Absolute Discounting:**

  * Reduces each non-zero count by a fixed discount

* **Backoff vs. Interpolation:**

  * **Backoff:** Use lower-order models when higher-order data is sparse
  * **Interpolation:** Combine all model orders using weighted averages

#### Practical Use

These models improve robustness in sparse data environments like speech recognition and predictive typing.

---

### Lecture 13: Computational Morphology

#### What is Morphology?

* Study of word structure: roots, prefixes, suffixes
* Types:

  * **Inflectional Morphology:** Changes grammatical function (e.g., *cats*)
  * **Derivational Morphology:** Changes meaning or category (e.g., *happy* → *unhappy*)

#### Applications

* Lemmatization
* POS Tagging
* Information Retrieval

[Learn more](https://en.wikipedia.org/wiki/Morphology_%28linguistics%29)

---

### Lecture 14: Finite-State Methods for Morphology

#### Finite-State Transducers (FSTs)

* Map between surface and lexical forms
* Used for:

  * Tokenization
  * Morphological analysis and generation

#### Components

* **Lexicon**
* **Morphotactics**
* **Orthographic Rules**

[More on FSTs](https://en.wikipedia.org/wiki/Finite-state_transducer)

---

### Lecture 15: Introduction to POS Tagging

#### Part-of-Speech (POS) Tagging

* Assigning word classes (noun, verb, adjective) to each token

#### Techniques

* Rule-based tagging
* Probabilistic tagging (e.g., using Hidden Markov Models)

#### Applications

* Syntax parsing
* Named Entity Recognition
* Machine Translation

[POS Tagging](https://en.wikipedia.org/wiki/Part-of-speech_tagging)

---

### Lecture 16: Hidden Markov Models for POS Tagging

#### Hidden Markov Model (HMM)

* Models sequences where:

  * Observations = words
  * States = POS tags

* Uses:

  * **Emission probabilities** $P(word | tag)$
  * **Transition probabilities** $P(tag_i | tag_{i-1})$

#### Decoding

* **Viterbi Algorithm:** Finds the most probable tag sequence

[Hidden Markov Model](https://en.wikipedia.org/wiki/Hidden_Markov_model)

---

## References

* [Kneser-Ney Smoothing](https://en.wikipedia.org/wiki/Kneser%E2%80%93Ney_smoothing)
* [Finite-State Transducer](https://en.wikipedia.org/wiki/Finite-state_transducer)
* [Morphology (Linguistics)](https://en.wikipedia.org/wiki/Morphology_%28linguistics%29)
* [Part-of-speech Tagging](https://en.wikipedia.org/wiki/Part-of-speech_tagging)
* [Hidden Markov Model](https://en.wikipedia.org/wiki/Hidden_Markov_model)
* [Viterbi Algorithm](https://en.wikipedia.org/wiki/Viterbi_algorithm)



## Week 4

### Lecture 17: Viterbi Decoding and Parameter Learning in HMM

#### Viterbi Decoding Recap

* **Goal:** Find the most probable sequence of hidden states (POS tags) given observations (words).
* Uses **dynamic programming** to efficiently compute optimal sequences.

#### Parameter Estimation

* **Supervised Learning:** From labeled corpora like [Penn Treebank](https://catalog.ldc.upenn.edu/LDC99T42)
* **Transition Probabilities:** $P(t_i | t_{i-1})$
* **Emission Probabilities:** $P(w_i | t_i)$

---

### Lecture 18: Baum-Welch Algorithm

#### Expectation-Maximization for HMMs

* Used when **training on unlabeled data**
* Two steps:

  * **Expectation (E-step):** Compute expected counts
  * **Maximization (M-step):** Re-estimate model parameters

#### Applications

* Speech recognition
* POS tagging in unsupervised settings

[Baum-Welch Algorithm](https://en.wikipedia.org/wiki/Baum%E2%80%93Welch_algorithm)

---

### Lecture 19: Maximum Entropy Models - Part I

#### Motivation

* Address limitations of HMMs (e.g., strong independence assumptions)
* Use rich feature sets

#### Concepts

* **Exponential Form:**

$$
  P(y|x) = \frac{1}{Z(x)} \exp\left(\sum_i \lambda_i f_i(x, y)\right)
$$

* Features $f_i(x, y)$ are binary functions indicating presence/absence of linguistic properties.

[Maximum Entropy Model](https://en.wikipedia.org/wiki/Maximum_entropy_classifier)

---

### Lecture 20: Maximum Entropy Models - Part II

#### Optimization Techniques

* **GIS (Generalized Iterative Scaling)**
* **Improved Iterative Scaling**
* **L-BFGS:** Popular for high-dimensional NLP tasks

#### Applications

* POS tagging
* Named entity recognition (NER)
* Sentiment classification

[GIS Algorithm](https://en.wikipedia.org/wiki/Generalized_iterative_scaling)

---

### Lecture 21: Conditional Random Fields (CRFs)

#### CRFs vs. MaxEnt

* CRFs model the **entire sequence**, not just individual labels.
* Suitable for **structured prediction tasks**.

#### CRF Advantages

* Global normalization
* Captures contextual dependencies

#### Applications

* Sequence labeling (POS tagging, NER, Chunking)

[Conditional Random Fields](https://en.wikipedia.org/wiki/Conditional_random_field)

---

## References

* [Viterbi Algorithm](https://en.wikipedia.org/wiki/Viterbi_algorithm)
* [Baum-Welch Algorithm](https://en.wikipedia.org/wiki/Baum%E2%80%93Welch_algorithm)
* [Maximum Entropy Classifier](https://en.wikipedia.org/wiki/Maximum_entropy_classifier)
* [Conditional Random Field](https://en.wikipedia.org/wiki/Conditional_random_field)
* [Penn Treebank](https://catalog.ldc.upenn.edu/LDC99T42)


## Week 5

### Lecture 22: Syntax – Introduction

#### Syntax in NLP

* Concerned with sentence structure and grammar
* Involves identifying constituents and hierarchical relationships

#### Key Concepts

* **Phrase Structure Grammar**
* **Parse Trees**
* **Context-Free Grammars (CFGs)**

[Syntax (Linguistics)](https://en.wikipedia.org/wiki/Syntax)

---

### Lecture 23: Syntax – Parsing I

#### Parsing Basics

* **Goal:** Derive a parse tree from a sentence using grammar rules
* **Top-Down Parsing:** Starts from the root and expands
* **Bottom-Up Parsing:** Starts from the leaves (words) and builds up

#### Issues

* Ambiguity in parse trees
* Efficiency trade-offs

[Parsing](https://en.wikipedia.org/wiki/Parsing)

---

### Lecture 24: Syntax – CKY Algorithm and PCFGs

#### CKY Parsing Algorithm

* **Cocke-Kasami-Younger (CKY) Algorithm**
* Bottom-up parsing using **dynamic programming**
* Requires grammar in **Chomsky Normal Form (CNF)**

#### PCFGs (Probabilistic Context-Free Grammars)

* CFGs augmented with probabilities
* Allows ranking of multiple parse trees

[CKY Algorithm](https://en.wikipedia.org/wiki/CYK_algorithm)
[PCFGs](https://en.wikipedia.org/wiki/Stochastic_context-free_grammar)

---

### Lecture 25: Inside-Outside Probabilities (PCFG Training)

#### Inside-Outside Algorithm

* EM-based algorithm for training PCFGs from unannotated data
* **Inside probability:** Probability of generating a substring from a non-terminal
* **Outside probability:** Probability of generating the rest of the string around it

[Inside–Outside Algorithm](https://en.wikipedia.org/wiki/Inside–outside_algorithm)

---

### Lecture 26: Inside-Outside Probabilities (Continued)

* In-depth discussion of computation methods
* Use of dynamic programming tables
* Applications:

  * Unsupervised parsing
  * Grammar induction

---

## References

* [Syntax](https://en.wikipedia.org/wiki/Syntax)
* [Parsing](https://en.wikipedia.org/wiki/Parsing)
* [CKY Algorithm](https://en.wikipedia.org/wiki/CYK_algorithm)
* [PCFG (Stochastic CFG)](https://en.wikipedia.org/wiki/Stochastic_context-free_grammar)
* [Inside–Outside Algorithm](https://en.wikipedia.org/wiki/Inside–outside_algorithm)


## Week 6

### Lecture 27: Dependency Grammars and Parsing – Introduction

#### Dependency Grammar

* Focuses on binary relationships between **head** words and their **dependents**
* Each word (except root) has exactly one head
* Forms a **dependency tree**

#### Advantages

* More direct representation of syntactic functions (e.g., subject, object)
* Suitable for free word-order languages

[Dependency Grammar](https://en.wikipedia.org/wiki/Dependency_grammar)

---

### Lecture 28: Transition-Based Parsing – Formulation

#### Transition-Based Parsing

* Builds dependency trees incrementally using:

  * **Stack**
  * **Buffer**
  * **Transition System** (e.g., SHIFT, LEFT-ARC, RIGHT-ARC)

#### Parsing Strategy

* **Deterministic parsing**
* Suitable for linear-time processing

[Transition-Based Parsing](https://web.stanford.edu/~jurafsky/slp3/ed-slides/DependencyParsing.pdf)

---

### Lecture 29: Transition-Based Parsing – Learning

#### Training Transition Parsers

* Requires treebank data with annotated dependency structures
* Feature extraction from:

  * Stack top
  * Buffer front
  * History of transitions

#### Classifiers Used

* Logistic Regression
* Support Vector Machines (SVMs)
* Neural Networks

---

### Lecture 30: MST-Based Dependency Parsing

#### Graph-Based Parsing

* Converts parsing into a **maximum spanning tree (MST)** problem
* Edge weights learned from features

#### Parsing Algorithms

* **Chu-Liu/Edmonds’ Algorithm**: For non-projective trees
* **Eisner's Algorithm**: For projective trees (dynamic programming)

[MST Parsing](https://www.aclweb.org/anthology/P05-1013/)

---

### Lecture 31: MST-Based Dependency Parsing – Learning

#### Training Graph-Based Parsers

* Feature-based score functions for edges
* Learning with:

  * Perceptron
  * Structured SVMs
  * CRFs

#### Pros and Cons

* **Pros:** Handles non-projective structures
* **Cons:** Higher computational cost than transition-based parsers

---

## References

* [Dependency Grammar](https://en.wikipedia.org/wiki/Dependency_grammar)
* [Transition-Based Parsing (PDF)](https://web.stanford.edu/~jurafsky/slp3/ed-slides/DependencyParsing.pdf)
* [MST Parsing (McDonald et al., 2005)](https://www.aclweb.org/anthology/P05-1013/)
* [Structured Prediction](https://en.wikipedia.org/wiki/Structured_prediction)



## Week 7

### Lecture 32: Distributional Semantics – Introduction

#### Key Idea

* **Meaning is derived from context.**
* Based on [Distributional Hypothesis](https://en.wikipedia.org/wiki/Distributional_semantics):

  > "You shall know a word by the company it keeps." — J.R. Firth

#### Applications

* Synonym detection
* Thesaurus construction
* Word similarity

[Distributional Semantics](https://en.wikipedia.org/wiki/Distributional_semantics)

---

### Lecture 33: Distributional Models of Semantics

#### Vector Space Models (VSM)

* Words represented as vectors in high-dimensional space
* Based on **co-occurrence counts** from large corpora

#### Techniques

* **Term-Document Matrix**
* **Term-Term Matrix**
* **Dimensionality Reduction**:

  * [SVD (Singular Value Decomposition)](https://en.wikipedia.org/wiki/Singular_value_decomposition)
  * PCA

---

### Lecture 34: Distributional Semantics – Applications & Structured Models

#### Applications

* Information Retrieval
* Document clustering
* Word sense discrimination

#### Structured Models

* Address limitations of basic VSMs
* Capture syntactic dependencies and richer context
* Examples: Dependency-based VSMs, Tensor-based models

---

### Lecture 35: Word Embeddings – Part I

#### Introduction

* Dense vector representations of words
* Capture syntactic and semantic similarities

#### Models

* **Word2Vec** (Skip-gram, CBOW)
* **GloVe (Global Vectors)**

#### Benefits

* Reduce sparsity
* Enable analogical reasoning (e.g., *king* - *man* + *woman* ≈ *queen*)

[Word Embedding](https://en.wikipedia.org/wiki/Word_embedding)
[Word2Vec](https://en.wikipedia.org/wiki/Word2vec)
[GloVe](https://nlp.stanford.edu/projects/glove/)

---

### Lecture 36: Word Embeddings – Part II

#### Training Embeddings

* Based on context window or co-occurrence matrix
* Trained on large corpora (e.g., Wikipedia, Common Crawl)

#### Improvements

* **Subword embeddings:** (e.g., [FastText](https://fasttext.cc/))
  Handle rare words using character n-grams
* **Contextual embeddings:**
  (e.g., [ELMo](https://allennlp.org/elmo), [BERT](https://en.wikipedia.org/wiki/BERT_%28language_model%29))

---

## References

* [Distributional Semantics](https://en.wikipedia.org/wiki/Distributional_semantics)
* [SVD](https://en.wikipedia.org/wiki/Singular_value_decomposition)
* [Word Embedding](https://en.wikipedia.org/wiki/Word_embedding)
* [Word2Vec](https://en.wikipedia.org/wiki/Word2vec)
* [GloVe](https://nlp.stanford.edu/projects/glove/)
* [FastText](https://fasttext.cc/)
* [ELMo](https://allennlp.org/elmo)
* [BERT](https://en.wikipedia.org/wiki/BERT_%28language_model%29)



## Week 8

### Lecture 37: Lexical Semantics

#### Definition

* Study of word meanings and word relations

#### Key Concepts

* **Sense**: Different meanings of a word (polysemy)
* **Synonymy**: Words with similar meanings
* **Antonymy**: Opposites (e.g., *hot* ↔ *cold*)
* **Hyponymy/Hypernymy**: Word hierarchies (e.g., *dog* → *animal*)

[Lexical Semantics](https://en.wikipedia.org/wiki/Lexical_semantics)

---

### Lecture 38: Lexical Semantics – WordNet

#### What is WordNet?

* A large lexical database of English
* Groups words into **synsets** (sets of synonyms)
* Captures:

  * Semantic relationships (e.g., hypernyms, hyponyms)
  * Lexical relations (e.g., derivationally related forms)

#### Usage

* Word similarity
* Word sense disambiguation (WSD)
* Semantic search

[WordNet](https://wordnet.princeton.edu/)

---

### Lecture 39: Word Sense Disambiguation (WSD) – Part I

#### Goal

* Identify the correct sense of a word in context

#### Approaches

* **Knowledge-based**: Use dictionaries, thesauri
* **Supervised Learning**: Train on sense-labeled data
* **Unsupervised Methods**: Cluster word usages

[Word Sense Disambiguation](https://en.wikipedia.org/wiki/Word-sense_disambiguation)

---

### Lecture 40: Word Sense Disambiguation – Part II

#### Supervised Methods

* Feature extraction from context
* Classifiers: Naive Bayes, SVM, Neural Networks

#### Evaluation Metrics

* Precision, Recall, F1 Score
* Baseline: Most Frequent Sense

---

### Lecture 41: Novel Word Sense Detection

#### Motivation

* New senses evolve over time, especially in social media and slang

#### Techniques

* Sense induction via clustering
* Dynamic language modeling

#### Applications

* Lexicon expansion
* Trend analysis
* Sentiment adaptation

---

## References

* [Lexical Semantics](https://en.wikipedia.org/wiki/Lexical_semantics)
* [WordNet](https://wordnet.princeton.edu/)
* [Word Sense Disambiguation](https://en.wikipedia.org/wiki/Word-sense_disambiguation)
* [Polysemy](https://en.wikipedia.org/wiki/Polysemy)
* [Hypernym](https://en.wikipedia.org/wiki/Hypernym_and_hyponym)



## Week 9

### Lecture 42: Topic Models – Introduction

#### Topic Modeling

* Unsupervised learning method to discover latent topics in a collection of documents

#### Key Idea

* Documents are mixtures of topics
* Topics are distributions over words

#### Applications

* Document classification
* Recommendation systems
* Information retrieval

[Topic Model](https://en.wikipedia.org/wiki/Topic_model)

---

### Lecture 43: Latent Dirichlet Allocation (LDA) – Formulation

#### LDA Overview

* A generative probabilistic model for topic modeling
* Assumes:

  * Each document has a distribution over topics
  * Each topic has a distribution over words

#### Parameters

* $\alpha$: Dirichlet prior for topic distribution per document
* $\beta$: Dirichlet prior for word distribution per topic

[LDA](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation)

---

### Lecture 44: Gibbs Sampling for LDA – Applications

#### Gibbs Sampling

* A Markov Chain Monte Carlo (MCMC) method
* Samples from the posterior distribution of topics

#### Steps

1. Initialize random topic assignments
2. Iteratively resample topic for each word using conditional probabilities

#### Applications

* Text classification
* Thematic analysis

[Gibbs Sampling](https://en.wikipedia.org/wiki/Gibbs_sampling)

---

### Lecture 45: LDA Variants and Applications – Part I

#### Variants

* **Hierarchical LDA (hLDA):** Adds topic hierarchy
* **Supervised LDA (sLDA):** Incorporates response variables (e.g., ratings, labels)

#### Domain Applications

* Legal text analysis
* Biomedical literature mining
* Social media trend detection

---

### Lecture 46: LDA Variants and Applications – Part II

#### More Variants

* **Dynamic Topic Models:** Track topic evolution over time
* **Correlated Topic Models (CTM):** Allow topic co-occurrence modeling

#### Tools and Libraries

* [Gensim](https://radimrehurek.com/gensim/)
* [Mallet](http://mallet.cs.umass.edu/)
* [Scikit-learn](https://scikit-learn.org/)

---

## References

* [Topic Models](https://en.wikipedia.org/wiki/Topic_model)
* [Latent Dirichlet Allocation](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation)
* [Gibbs Sampling](https://en.wikipedia.org/wiki/Gibbs_sampling)
* [Gensim](https://radimrehurek.com/gensim/)
* [Mallet](http://mallet.cs.umass.edu/)
* [Hierarchical LDA](https://en.wikipedia.org/wiki/Hierarchical_Latent_Dirichlet_Allocation)



## Week 10

### Lecture 47: Entity Linking – Part I

#### Definition

* **Entity Linking (EL)** is the task of mapping named entities in text to their corresponding entries in a knowledge base (e.g., Wikipedia, Wikidata).

#### Components

* **Mention Detection:** Identify spans referring to entities
* **Candidate Generation:** Retrieve possible entities
* **Disambiguation:** Select the most appropriate entity

[Entity Linking](https://en.wikipedia.org/wiki/Entity_linking)

---

### Lecture 48: Entity Linking – Part II

#### Disambiguation Methods

* **String similarity** (e.g., Jaccard, edit distance)
* **Context similarity** (e.g., cosine similarity of embeddings)
* **Popularity-based ranking**

#### Tools

* [DBpedia Spotlight](https://www.dbpedia-spotlight.org/)
* [TagMe](https://sobigdata.d4science.org/web/tagme/)

---

### Lecture 49: Information Extraction – Introduction

#### Goal

* Automatically extract structured information (e.g., entities, relationships, events) from unstructured text

#### Key Tasks

* **Named Entity Recognition (NER)**
* **Relation Extraction**
* **Event Extraction**

[Information Extraction](https://en.wikipedia.org/wiki/Information_extraction)

---

### Lecture 50: Relation Extraction

#### Relation Extraction (RE)

* Identify semantic relationships between entities (e.g., *works for*, *born in*)

#### Approaches

* **Pattern-based**: Use hand-crafted rules
* **Supervised Learning**: Requires labeled data
* **Distant Supervision**: Use knowledge bases to auto-label data

[Relation Extraction](https://en.wikipedia.org/wiki/Information_extraction#Relation_extraction)

---

### Lecture 51: Distant Supervision

#### Concept

* Automatically generate training data for relation extraction using existing KBs (e.g., Freebase)

#### Challenges

* **Noisy Labels:** Due to wrong assumption that all sentences mentioning entity pairs express the relation
* **Multiple Instance Learning (MIL):** Handles noise in training data

[Distant Supervision](https://en.wikipedia.org/wiki/Distant_supervision_for_relation_extraction)

---

## References

* [Entity Linking](https://en.wikipedia.org/wiki/Entity_linking)
* [Information Extraction](https://en.wikipedia.org/wiki/Information_extraction)
* [Relation Extraction](https://en.wikipedia.org/wiki/Information_extraction#Relation_extraction)
* [Distant Supervision](https://en.wikipedia.org/wiki/Distant_supervision_for_relation_extraction)
* [DBpedia Spotlight](https://www.dbpedia-spotlight.org/)
* [TagMe](https://sobigdata.d4science.org/web/tagme/)



## Week 11

### Lecture 52: Text Summarization – LEXRANK

#### Goal

* Automatically generate concise summaries from longer documents

#### LEXRANK

* Graph-based summarization algorithm
* Builds a similarity graph of sentences
* Applies [PageRank](https://en.wikipedia.org/wiki/PageRank) to identify central sentences

#### Features

* Unsupervised
* Works well for extractive summarization

[Text Summarization](https://en.wikipedia.org/wiki/Automatic_summarization)

---

### Lecture 53: Optimization-Based Approaches for Summarization

#### Integer Linear Programming (ILP)

* Select optimal set of sentences to maximize relevance and minimize redundancy

#### Other Methods

* Submodular optimization
* Budget constraints (e.g., summary length)

#### Applications

* Multi-document summarization
* Scientific article summarization

---

### Lecture 54: Summarization Evaluation

#### Metrics

* **ROUGE (Recall-Oriented Understudy for Gisting Evaluation):**

  * Measures overlap with reference summaries
  * ROUGE-N, ROUGE-L, ROUGE-SU

#### Other Approaches

* Human evaluations (fluency, coherence, informativeness)

[ROUGE](https://en.wikipedia.org/wiki/ROUGE_%28metric%29)

---

### Lecture 55: Text Classification – Part I

#### Problem Definition

* Assign predefined categories (labels) to text documents

#### Models

* **Naive Bayes**
* **Logistic Regression**
* **Support Vector Machines (SVMs)**

#### Applications

* Spam detection
* Sentiment analysis
* News categorization

[Text Classification](https://en.wikipedia.org/wiki/Text_classification)

---

### Lecture 56: Text Classification – Part II

#### Advanced Techniques

* **Neural Networks**:

  * Feedforward
  * Convolutional (CNN)
  * Recurrent (RNN, LSTM)

* **Pretrained Embeddings** (Word2Vec, GloVe)

* **Transformers** (e.g., BERT)

#### Evaluation

* Accuracy, Precision, Recall, F1 Score

---

### Lecture 57: Tutorial II

#### Hands-on Practice

* Summarization using LexRank
* ROUGE evaluation
* Implementing and evaluating classifiers using:

  * [scikit-learn](https://scikit-learn.org/)
  * [TensorFlow](https://www.tensorflow.org/) or [PyTorch](https://pytorch.org/)

---

## References

* [Text Summarization](https://en.wikipedia.org/wiki/Automatic_summarization)
* [LEXRANK](https://web.eecs.umich.edu/~mihalcea/papers/Mihalcea.Tarau.JAIR04.pdf)
* [ROUGE](https://en.wikipedia.org/wiki/ROUGE_%28metric%29)
* [Text Classification](https://en.wikipedia.org/wiki/Text_classification)
* [Scikit-learn](https://scikit-learn.org/)
* [TensorFlow](https://www.tensorflow.org/)
* [PyTorch](https://pytorch.org/)



## Week 12

### Lecture 58: Tutorial III

### Lecture 59: Tutorial IV

### Lecture 60: Tutorial V

#### Topics Covered

* Recap and implementation exercises on:

  * Entity Linking
  * Summarization
  * Text Classification
* Evaluation using:

  * Accuracy
  * ROUGE scores
  * F1 metrics
* Tools used:

  * Python, Jupyter Notebooks
  * NLTK, spaCy, scikit-learn, PyTorch/TensorFlow

---

### Lecture 61: Sentiment Analysis – Introduction

#### Definition

* Determine the **polarity** (positive, negative, neutral) of text

#### Use Cases

* Product reviews
* Social media monitoring
* Market research

[Sentiment Analysis](https://en.wikipedia.org/wiki/Sentiment_analysis)

---

### Lecture 62: Sentiment Analysis – Affective Lexicons

#### Affective Lexicons

* Word lists annotated with emotion or sentiment labels
* Examples:

  * [SentiWordNet](https://sentiwordnet.isti.cnr.it/)
  * AFINN
  * NRC Emotion Lexicon

#### Usage

* Lexicon-based sentiment scoring
* Feature extraction for classifiers

---

### Lecture 63: Learning Affective Lexicons

#### Methods

* Bootstrapping from seed words
* Semi-supervised learning
* Label propagation

#### Applications

* Domain-specific lexicons
* Multilingual sentiment resources

---

### Lecture 64: Computing with Affective Lexicons

#### Feature Engineering

* Count-based features: positive/negative word counts
* Score-based features: sentiment scores
* Normalization and aggregation strategies

#### Challenges

* Sarcasm
* Context-dependence

---

### Lecture 65: Aspect-Based Sentiment Analysis (ABSA)

#### What is ABSA?

* Detect sentiment **specific to aspects** (e.g., *battery life*, *screen quality*)

#### Pipeline

1. Aspect extraction
2. Sentiment classification for each aspect

#### Applications

* Fine-grained opinion mining
* Product feature analysis

[Aspect-Based Sentiment Analysis](https://en.wikipedia.org/wiki/Sentiment_analysis#Aspect-based_sentiment_analysis)

---

## References

* [Sentiment Analysis](https://en.wikipedia.org/wiki/Sentiment_analysis)
* [SentiWordNet](https://sentiwordnet.isti.cnr.it/)
* [AFINN Lexicon](https://github.com/fnielsen/afinn)
* [NRC Emotion Lexicon](https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm)
* [Aspect-Based Sentiment Analysis](https://en.wikipedia.org/wiki/Sentiment_analysis#Aspect-based_sentiment_analysis)



# Natural Language Processing – Lecture 4

## 📊 Empirical Laws in Language

In this lecture, Prof. Pawan Goyal introduces empirical linguistic patterns found in real-world corpora, focusing on **word distributions**, **types vs. tokens**, and the distinction between **function** and **content** words.

---

## 🧱 Function Words vs Content Words

### 📌 Function Words

* Serve **grammatical roles** (e.g., *the*, *is*, *and*, *to*)
* Include **prepositions**, **pronouns**, **auxiliary verbs**, **conjunctions**, and **articles**
* Form a **closed class** (few new entries)

### 🔤 Content Words

* Carry **semantic meaning**: **nouns**, **verbs**, **adjectives**, etc.
* Form an **open class** (new words regularly added)

📖 Related: [Function and Content Words (Wikipedia)](https://en.wikipedia.org/wiki/Function_word)

---

## 🧪 Demonstration: Word Substitution

Two modified sentences were presented:

* One with **content words replaced** (meaning lost, structure visible)
* One with **function words replaced** (structure distorted, meaning retained)

**Conclusion:**

* **Function words** provide **syntactic structure**
* **Content words** convey **topic and meaning**

---

## 📚 Word Frequencies in a Corpus

Corpus: *Tom Sawyer* by [Mark Twain](https://en.wikipedia.org/wiki/Mark_Twain)

### Top Frequent Words:

* “the” – 3332 times
* “and”, “to”, “a”, “of” – all high frequency
* Mostly **function words**

### Notable Exception:

* **“Tom”** appears frequently due to the topic of the book

🔗 [Word Frequency](https://en.wikipedia.org/wiki/Word_frequency)

---

## 🔠 Type vs Token

### Definitions:

| Term      | Meaning                                                |
| --------- | ------------------------------------------------------ |
| **Token** | Each occurrence of a word in the corpus                |
| **Type**  | Unique word (distinct spelling/form) in the vocabulary |

> E.g., "will will" → 2 tokens, 1 type

---

## 📐 Type-Token Ratio (TTR)

**TTR = Unique Words (Types) / Total Words (Tokens)**

* **High TTR:** Many unique words, diverse vocabulary
* **Low TTR:** Repetitive usage

### Corpus Comparison:

| Text                   | Tokens | Types  | TTR   |
| ---------------------- | ------ | ------ | ----- |
| *Tom Sawyer*           | 71,370 | 8,018  | 0.112 |
| *Complete Shakespeare* | 88,400 | 29,066 | 0.329 |

🔗 [Type-Token Ratio (Wikipedia)](https://en.wikipedia.org/wiki/Lexical_density#Type%E2%80%93token_ratio)

---

## 📰 TTR by Text Genre

### Genres Compared:

* **Conversation**
* **Academic Prose**
* **News**
* **Fiction**

### Observation:

* **Conversation** tends to have the **lowest TTR** due to word repetition
* **Academic prose** typically has the **highest TTR**

📖 Related: [Corpus Linguistics](https://en.wikipedia.org/wiki/Corpus_linguistics)

---

## 📌 Summary

* Language exhibits **predictable patterns** in word frequency
* Distinguishing **function vs. content words** is crucial for text analysis
* **TTR** helps measure vocabulary diversity across genres
* Understanding these patterns lays the foundation for **language modeling**, **information retrieval**, and **text classification**

---

## References

* [Function Word](https://en.wikipedia.org/wiki/Function_word)
* [Word Frequency](https://en.wikipedia.org/wiki/Word_frequency)
* [Type-Token Ratio](https://en.wikipedia.org/wiki/Lexical_density#Type%E2%80%93token_ratio)
* [Corpus Linguistics](https://en.wikipedia.org/wiki/Corpus_linguistics)
* [Mark Twain](https://en.wikipedia.org/wiki/Mark_Twain)


