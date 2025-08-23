

# Natural Language Processing – Course Introduction

**Instructor:** Prof. Pawan Goyal
**Institution:** [IIT Kharagpur](https://www.iitkgp.ac.in/)

---

## 📘 Course Overview

* **Duration:** 12 weeks
* **Structure:** 5 modules per week
* **Support:** Two TAs – Amrith Krishna and Mayank Singh
* **Contact:** Provided in the course materials

---

## 📚 Recommended Books

1. **[Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/)** by Jurafsky and Martin (2nd or 3rd edition)
2. **[Foundations of Statistical Natural Language Processing](https://mitpress.mit.edu/9780262133609/foundations-of-statistical-natural-language-processing/)** by Manning and Schütze

Additional materials and [IPython Notebooks](https://jupyter.org/) will be shared for hands-on learning.

---

## 🧪 Evaluation Scheme

* **Assignments:** 25% (weekly)
* **Final Exam:** 75%

---

## 📌 Topics Covered

### Basic Topics

* **Text Processing:** Tokenization, stemming, lemmatization
* **[Language Modeling](https://en.wikipedia.org/wiki/Language_model)**
* **Morphology & Syntax**
* **[Semantics](https://en.wikipedia.org/wiki/Semantics):** Lexical and distributional semantics
* **[Word Embeddings](https://en.wikipedia.org/wiki/Word_embedding)**
* **[Topic Modeling](https://en.wikipedia.org/wiki/Topic_model)**

### Applications

* **[Entity Linking](https://en.wikipedia.org/wiki/Entity_linking)** & **[Information Extraction](https://en.wikipedia.org/wiki/Information_extraction)**
* **[Text Summarization](https://en.wikipedia.org/wiki/Automatic_summarization)**
* **[Text Classification](https://en.wikipedia.org/wiki/Text_classification)**
* **[Sentiment Analysis](https://en.wikipedia.org/wiki/Sentiment_analysis)**

---

## 🧠 NLP: Scientific vs Engineering Goals

* **Scientific:** Can machines truly understand human language?
* **Engineering:** Build practical tools (e.g., [Google Translate](https://translate.google.com/), search engines, chatbots)

---

## 🌍 Why Study NLP?

* Text is the largest store of human knowledge ([Wikipedia](https://www.wikipedia.org/), news, scientific papers, [social media](https://en.wikipedia.org/wiki/Social_media))
* Available in many languages: multilingual processing and [machine translation](https://en.wikipedia.org/wiki/Machine_translation) are essential

---

## 🧩 Challenges in NLP

### Ambiguities

* **Lexical Ambiguity:** e.g., *“Will Will will Will’s will?”*
* **Structural Ambiguity:** e.g., *“The man saw the boy with the binoculars”*
* **Vagueness:** e.g., *“It’s very warm”*

### Social Media & Informal Text

* **Non-standard language:** *CU L8R*, hashtags, emojis
* **New words/senses:** e.g., *Googling*, *unfriending*

### Other Complexities

* **[Idioms](https://en.wikipedia.org/wiki/Idiom)**
* **[Named Entity Recognition](https://en.wikipedia.org/wiki/Named-entity_recognition)**
* **Multilingual content**
* **Discourse context**

---

## 📈 Empirical Laws in Language

### Word Frequency

* Dominated by **function words** (e.g., *the*, *is*)
* Occasionally topic-specific **content words** (e.g., *Tom* in *Tom Sawyer*)

### Type-Token Ratio (TTR)

* **TTR = Unique Words / Total Words**
* Indicates richness and repetitiveness of vocabulary
* Varies by genre:

  * High TTR: academic texts
  * Low TTR: casual conversation

---

## 📎 Tools & Approaches

* **[Probabilistic Models](https://en.wikipedia.org/wiki/Statistical_natural_language_processing)**
* **[Parsing](https://en.wikipedia.org/wiki/Parsing)**
* **[Machine Learning](https://en.wikipedia.org/wiki/Machine_learning)**
* **Language-specific rules and corpora**

---

## References

* [Jurafsky & Martin – Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/)
* [Manning & Schütze – Foundations of Statistical NLP](https://mitpress.mit.edu/9780262133609/foundations-of-statistical-natural-language-processing/)
* [Natural Language Processing (Wikipedia)](https://en.wikipedia.org/wiki/Natural_language_processing)
* [Word Embedding](https://en.wikipedia.org/wiki/Word_embedding)
* [Topic Modeling](https://en.wikipedia.org/wiki/Topic_model)
* [Named Entity Recognition](https://en.wikipedia.org/wiki/Named-entity_recognition)
* [Machine Learning](https://en.wikipedia.org/wiki/Machine_learning)



# Natural Language Processing – Lecture 2

## 🎯 What Do We Do in NLP?

In this lecture, Prof. Pawan Goyal explores the real-world tasks and applications tackled in **Natural Language Processing (NLP)**, emphasizing both its ambitious and practical goals.

---

## 🎓 Goals of NLP

### Scientific Goal

* Understand how **humans process language**
* Teach **computers** to understand and respond in **natural language**

### Engineering Goal

* Build systems that **process language** for **practical use cases**
* Examples: Translation, summarization, search engines, chatbots

---

## ⚙️ Ambitious Applications

### 1. **Machine Translation**

* Tools like [Google Translate](https://translate.google.com/) are widely used but **not always accurate**
* **Word Sense Disambiguation (WSD)** is crucial (e.g., "cool" ≠ "cold")

### 2. **Conversational Agents**

* Open-domain chatbots (e.g., Microsoft’s [Tay](https://en.wikipedia.org/wiki/Tay_%28bot%29)) failed due to lack of control
* Domain-specific bots (e.g., course assistants) are more successful

---

## ✅ Practical Applications

### 🔎 Information Retrieval & Query Processing

* **Spelling correction**
* **Query completion** using [language models](https://en.wikipedia.org/wiki/Language_model)

### 🧠 Information Extraction

* Extract structured facts (e.g., names, roles, dates) from **unstructured text**
* Applications in [Named Entity Recognition (NER)](https://en.wikipedia.org/wiki/Named-entity_recognition)

### 🤖 Educational Assistants

* Use of chatbots in courses to answer routine queries with high accuracy

### 🗣️ Sentiment Analysis

* Analyzing opinions from [social media](https://en.wikipedia.org/wiki/Social_media), reviews, political discourse

### 🚫 Spam Detection

* Filters in email and platforms like YouTube and Twitter
* Classify based on **textual patterns**

### 🌐 Machine Translation Services

* Translating full webpages and documents

### 📰 Text Summarization

* Creating concise summaries from news or scientific articles

---

## 🛠 Challenges in NLP Applications

* Systems can make **blunders**, especially when:

  * Lacking **contextual understanding**
  * Working in **open domains**

* Yet NLP is good enough for:

  * **Search engines**
  * **Assistants**
  * **Language services**

---

## 📌 Summary

NLP includes both high-level research and grounded engineering. Its ultimate goal is to:

* Build intelligent systems that **understand and process human language**
* Create real-world tools that solve practical problems

---

## References

* [Google Translate](https://translate.google.com/)
* [Word Sense Disambiguation](https://en.wikipedia.org/wiki/Word-sense_disambiguation)
* [Microsoft Tay Bot](https://en.wikipedia.org/wiki/Tay_%28bot%29)
* [Language Models](https://en.wikipedia.org/wiki/Language_model)
* [Named Entity Recognition](https://en.wikipedia.org/wiki/Named-entity_recognition)
* [Sentiment Analysis](https://en.wikipedia.org/wiki/Sentiment_analysis)
* [Text Summarization](https://en.wikipedia.org/wiki/Automatic_summarization)




# Natural Language Processing – Lecture 3

## 🤔 Why is NLP Hard?

This lecture explores the inherent complexities and ambiguities in natural languages that make **NLP a challenging field**. It discusses types of ambiguities, linguistic phenomena, and why language understanding is far from trivial.

---

## 🔄 Types of Ambiguity in Language

### 1. **Lexical Ambiguity**

* A word has multiple meanings.

* Example: *“Will Will will Will’s will?”*

  * Modal verb, proper noun, future action, noun (legal document)

* Example: *“Rose rose to put rose roes on her rows of roses.”*

* Famous: *“Buffalo buffalo Buffalo buffalo buffalo buffalo Buffalo buffalo.”*

  * Uses "buffalo" as city, animal, and verb (to bully)

[Lexical Ambiguity → Wikipedia](https://en.wikipedia.org/wiki/Lexical_ambiguity)

---

### 2. **Structural Ambiguity**

* Different parse trees yield different meanings.

* Example: *“The man saw the boy with the binoculars.”*

  * Who has the binoculars?

* Example: *“Flying planes can be dangerous.”*

  * Flying is dangerous OR flying planes are dangerous?

[Structural Ambiguity → Wikipedia](https://en.wikipedia.org/wiki/Syntactic_ambiguity)

---

## 🧩 Other Language Complexities

### 🔁 Vagueness and Imprecision

* Example: *“It’s very warm here.”*

  * What temperature is “warm”?

* Example: *“I’m sure she must have.”*

  * Indicates uncertainty

---

## 😂 Ambiguity in Humor

* Jokes often rely on ambiguity:

  * *“Why is the teacher wearing sunglasses?” → “Because the class is bright.”*

---

## 🗞️ Ambiguity in Headlines

* Example: *“Hospitals are sued by 7 foot doctors.”*
* Example: *“Stolen painting found by tree.”*
* Example: *“Teacher strikes idle kids.”*

---

## 🧠 Exercise: "I Made Her Duck"

Find multiple interpretations:

1. I cooked a duck for her.
2. I cooked the duck that belongs to her.
3. I created a toy duck for her.
4. I forced her to lower her head.
5. I turned her into a duck (magic).

Ambiguity arises due to:

* **Syntactic Category Ambiguity** (noun vs. verb)
* **Possessive vs. Dative Interpretation** of “her”
* **Verb Usage**: transitive, ditransitive, action-transitive
* **Speech Ambiguity** (e.g., “I’m aid her duck” vs. “I made her duck”)

---

## 🧮 Parsing Explosion

* Sentence: *“I saw the man on the hill in Texas with the telescope…”*
* Number of parses increases rapidly with sentence length

  * Example: 132 parses for one sentence (relates to [Catalan numbers](https://en.wikipedia.org/wiki/Catalan_number))

---

## 🗣️ Why Is Language Ambiguous?

* **Efficiency:** Humans prefer concise expressions.
* **Shared Knowledge:** Listeners resolve ambiguity using context.
* NLP systems lack this shared background knowledge.

---

## 🤖 Natural vs. Programming Languages

| Feature      | Natural Language            | Programming Language     |
| ------------ | --------------------------- | ------------------------ |
| Ambiguity    | High                        | None                     |
| Grammar      | Implicit, context-dependent | Explicit, formal         |
| Parsing Time | Variable, non-deterministic | Deterministic, efficient |

---

## 🧵 Challenges in Modern Text (e.g., Social Media)

* Non-standard forms: *CU L8R*, @mentions, hashtags
* [Code-switching](https://en.wikipedia.org/wiki/Code-switching), new slang
* Ambiguous segmentation (e.g., “New York-New Haven Railroad”)
* Idioms: e.g., *“burn the midnight oil”*
* Evolving usage: *“unfriend”*, *“retweet”*, *“Google”* as a verb

---

## 📏 NLP Requires

* Knowledge of:

  * **Language structure**
  * **World knowledge**
  * **Efficient integration methods**

* **Probabilistic models** are used to:

  * Handle ambiguity
  * Predict word meaning and structure
  * Support applications like [speech recognition](https://en.wikipedia.org/wiki/Speech_recognition)

---

## References

* [Lexical Ambiguity](https://en.wikipedia.org/wiki/Lexical_ambiguity)
* [Syntactic Ambiguity](https://en.wikipedia.org/wiki/Syntactic_ambiguity)
* [Catalan Numbers](https://en.wikipedia.org/wiki/Catalan_number)
* [Speech Recognition](https://en.wikipedia.org/wiki/Speech_recognition)
* [Code-Switching](https://en.wikipedia.org/wiki/Code-switching)
* [Idioms in Language](https://en.wikipedia.org/wiki/Idiom)



# Natural Language Processing – Lecture 4

## 📊 Empirical Laws in Language

In this lecture, Prof. Pawan Goyal introduces empirical linguistic patterns found in real-world corpora, focusing on **word distributions**, **types vs. tokens**, and the distinction between **function** and **content** words.



## 🧱 Function Words vs Content Words

### 📌 Function Words

* Serve **grammatical roles** (e.g., *the*, *is*, *and*, *to*)
* Include **prepositions**, **pronouns**, **auxiliary verbs**, **conjunctions**, and **articles**
* Form a **closed class** (few new entries)

### 🔤 Content Words

* Carry **semantic meaning**: **nouns**, **verbs**, **adjectives**, etc.
* Form an **open class** (new words regularly added)

📖 Related: [Function and Content Words (Wikipedia)](https://en.wikipedia.org/wiki/Function_word)



## 🧪 Demonstration: Word Substitution

Two modified sentences were presented:

* One with **content words replaced** (meaning lost, structure visible)
* One with **function words replaced** (structure distorted, meaning retained)

**Conclusion:**

* **Function words** provide **syntactic structure**
* **Content words** convey **topic and meaning**



## 📚 Word Frequencies in a Corpus

Corpus: *Tom Sawyer* by [Mark Twain](https://en.wikipedia.org/wiki/Mark_Twain)

### Top Frequent Words:

* “the” – 3332 times
* “and”, “to”, “a”, “of” – all high frequency
* Mostly **function words**

### Notable Exception:

* **“Tom”** appears frequently due to the topic of the book

🔗 [Word Frequency](https://en.wikipedia.org/wiki/Word_frequency)



## 🔠 Type vs Token

### Definitions:

| Term      | Meaning                                                |
| --------- | ------------------------------------------------------ |
| **Token** | Each occurrence of a word in the corpus                |
| **Type**  | Unique word (distinct spelling/form) in the vocabulary |

> E.g., "will will" → 2 tokens, 1 type



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



## 📌 Summary

* Language exhibits **predictable patterns** in word frequency
* Distinguishing **function vs. content words** is crucial for text analysis
* **TTR** helps measure vocabulary diversity across genres
* Understanding these patterns lays the foundation for **language modeling**, **information retrieval**, and **text classification**



## References

* [Function Word](https://en.wikipedia.org/wiki/Function_word)
* [Word Frequency](https://en.wikipedia.org/wiki/Word_frequency)
* [Type-Token Ratio](https://en.wikipedia.org/wiki/Lexical_density#Type%E2%80%93token_ratio)
* [Corpus Linguistics](https://en.wikipedia.org/wiki/Corpus_linguistics)
* [Mark Twain](https://en.wikipedia.org/wiki/Mark_Twain)



# Natural Language Processing – Lecture 5

## 📜 Empirical Laws (Continued)

This lecture builds upon Lecture 4, diving deeper into **statistical patterns in language** and their applications in **NLP**. It explores **Zipf’s Law**, **Heaps’ Law**, and **other statistical observations** from linguistic corpora.



## ⚖️ Zipf’s Law

### Statement

* In natural language, the **frequency of any word** is **inversely proportional to its rank** in the frequency table.
* Formally:

$$
  f(r) \propto \frac{1}{r}
$$

  where:

  * $f(r)$ = frequency of word with rank $r$
  * $r$ = rank of the word (1 = most frequent)

### Example

* Rank 1 word (*the*) is \~10× more frequent than rank 10 word.
* Observed across many languages and corpora.

📖 [Zipf’s Law (Wikipedia)](https://en.wikipedia.org/wiki/Zipf%27s_law)



## 📈 Heaps’ Law

### Statement

* As corpus size increases, the **number of unique words (vocabulary size)** grows sublinearly.
* Formula:

$$
  V(N) = kN^\beta
$$

  where:

  * $V(N)$ = number of distinct words in corpus of size $N$
  * $k$, $\beta$ = constants (typically $0.4 < \beta < 0.6$)

### Implication

* Even in massive corpora, new words **keep appearing**, but at a **diminishing rate**.

📖 [Heaps’ Law (Wikipedia)](https://en.wikipedia.org/wiki/Heaps%27_law)



## 📊 Statistical Observations

1. **Word Frequencies**:

   * A few words dominate (function words like *the*, *is*, *of*).
   * Most words occur **rarely**.

2. **Vocabulary Growth**:

   * **Infinite potential vocabulary** (due to new words, neologisms, proper names).

3. **Stop Words**:

   * Very frequent but carry little topical meaning.
   * Commonly removed in tasks like **information retrieval**.

📖 [Stop Words (Wikipedia)](https://en.wikipedia.org/wiki/Stop_words)



## 🧠 Applications in NLP

* **Language Modeling** → Predict next word based on probabilities
* **Information Retrieval** → Ignore stop words, focus on content words
* **Text Summarization** → Identify key content words
* **Speech Recognition** → Use frequency models to disambiguate words



## 📌 Key Takeaways

* **Zipf’s Law** → Word frequencies follow a predictable power-law distribution
* **Heaps’ Law** → Vocabulary grows with corpus size but sublinearly
* **Implication**: Language is both **repetitive** (common words) and **creative** (new words keep appearing)



## References

* [Zipf’s Law](https://en.wikipedia.org/wiki/Zipf%27s_law)
* [Heaps’ Law](https://en.wikipedia.org/wiki/Heaps%27_law)
* [Stop Words](https://en.wikipedia.org/wiki/Stop_words)
* [Language Modeling](https://en.wikipedia.org/wiki/Language_model)
* [Information Retrieval](https://en.wikipedia.org/wiki/Information_retrieval)


# Natural Language Processing – Lecture 6

## 🛠 Text Preprocessing

This lecture introduces **text preprocessing**, the **first step in NLP pipelines**. Preprocessing transforms **raw text** into a structured format suitable for computational models.



## 🧹 Why Preprocessing?

* Raw text is **noisy, unstructured, and inconsistent**.
* Preprocessing improves:

  * **Efficiency** (smaller vocabulary, reduced redundancy)
  * **Accuracy** (clearer representation of meaning)
  * **Generalization** (removes irrelevant variations)



## 🔑 Key Preprocessing Steps

### 1. **Tokenization**

* Splitting text into **words**, **sentences**, or **subwords**.
* Challenges:

  * Handling punctuation: *“U.S.A.” vs “USA”*
  * Contractions: *“don’t” → do + not*
  * Languages without spaces: [Chinese word segmentation](https://en.wikipedia.org/wiki/Word_segmentation)

📖 [Tokenization (Wikipedia)](https://en.wikipedia.org/wiki/Lexical_analysis#Tokenization)



### 2. **Normalization**

* Making text consistent:

  * Lowercasing
  * Removing punctuation
  * Handling numbers (e.g., *123 → NUM*)

📖 [Text Normalization](https://en.wikipedia.org/wiki/Text_normalization)



### 3. **Stemming**

* Reducing words to their **root form** (often crude, rule-based).
* Example:

  * *running → run*
  * *studies → studi*

📖 [Stemming](https://en.wikipedia.org/wiki/Stemming)



### 4. **Lemmatization**

* Mapping words to **dictionary form** using linguistic knowledge.
* Example:

  * *better → good*
  * *was → be*

📖 [Lemmatization](https://en.wikipedia.org/wiki/Lemmatisation)



### 5. **Stop Word Removal**

* Removing very frequent words with little semantic value (e.g., *the, is, of*).
* Helps in tasks like **search engines** and **topic modeling**.

📖 [Stop Words](https://en.wikipedia.org/wiki/Stop_words)



### 6. **Handling Rare Words / OOV (Out-of-Vocabulary)**

* Replace rare words with a placeholder (e.g., `<UNK>`).
* Alternative: **Subword models** ([Byte Pair Encoding](https://en.wikipedia.org/wiki/Byte_pair_encoding)).



## 📊 Example Workflow

Sentence: *“The children’s studies were running quickly.”*

1. **Tokenization** → \[The] \[children’s] \[studies] \[were] \[running] \[quickly]
2. **Lowercasing** → \[the] \[children’s] \[studies] \[were] \[running] \[quickly]
3. **Stemming** → \[the] \[children] \[studi] \[were] \[run] \[quick]
4. **Lemmatization** → \[the] \[child] \[study] \[be] \[run] \[quickly]
5. **Stop word removal** → \[child] \[study] \[run] \[quickly]



## 🧠 Why It Matters?

* Preprocessing **directly impacts model performance**.
* Some modern models ([BERT](https://en.wikipedia.org/wiki/BERT_%28language_model%29), [GPT](https://en.wikipedia.org/wiki/GPT-3)) rely on **subword tokenization** rather than heavy preprocessing.
* Choice of preprocessing depends on:

  * **Task requirements**
  * **Model type**
  * **Language specifics**



## 📌 Summary

* Text preprocessing is **essential** for NLP tasks.
* Core steps: **tokenization, normalization, stemming, lemmatization, stop-word removal, rare word handling**.
* Modern deep learning models use **subword-based methods** but still rely on preprocessing foundations.



## References

* [Tokenization](https://en.wikipedia.org/wiki/Lexical_analysis#Tokenization)
* [Text Normalization](https://en.wikipedia.org/wiki/Text_normalization)
* [Stemming](https://en.wikipedia.org/wiki/Stemming)
* [Lemmatization](https://en.wikipedia.org/wiki/Lemmatisation)
* [Stop Words](https://en.wikipedia.org/wiki/Stop_words)
* [Byte Pair Encoding](https://en.wikipedia.org/wiki/Byte_pair_encoding)
* [BERT](https://en.wikipedia.org/wiki/BERT_%28language_model%29)
* [GPT Models](https://en.wikipedia.org/wiki/GPT-3)

# Natural Language Processing – Lecture 7

## 🔢 Edit Distance and String Similarity

This lecture introduces **edit distance** (also called **Levenshtein distance**) and its role in **text similarity**, **spell correction**, and **information retrieval**.


## 📌 Motivation

* Natural language text often contains:

  * **Typos**
  * **Spelling variations** (e.g., *color* vs. *colour*)
  * **OCR errors**
* We need a way to **quantify similarity** between two strings.


## ✂️ Edit Distance

### Definition

* **Minimum number of operations** required to transform one string into another.
* Allowed operations:

  1. **Insertion**
  2. **Deletion**
  3. **Substitution**

📖 [Edit Distance (Wikipedia)](https://en.wikipedia.org/wiki/Edit_distance)


### Example

* String 1: **kitten**
* String 2: **sitting**

Operations:

1. kitten → sitten (substitution: k → s)
2. sitten → sittin (substitution: e → i)
3. sittin → sitting (insertion: g)

**Edit Distance = 3**


## 🧮 Dynamic Programming Approach

Matrix-based computation:

* Rows = characters of word 1
* Columns = characters of word 2
* Each cell = minimum edit distance up to that prefix

Time complexity: **O(m × n)** (where m and n are string lengths)

📖 [Levenshtein Distance](https://en.wikipedia.org/wiki/Levenshtein_distance)


## ⚡ Variants

1. **Hamming Distance**

   * Counts substitutions only
   * Requires strings of equal length
   * Example: *karolin* vs *kathrin* → distance = 3
     📖 [Hamming Distance](https://en.wikipedia.org/wiki/Hamming_distance)

2. **Damerau-Levenshtein Distance**

   * Includes **transposition** (swap of adjacent letters)
   * Example: *caht* vs *chat* → distance = 1

3. **Weighted Edit Distance**

   * Different costs for different operations
   * Useful in speech recognition, OCR


## 🔍 Applications

1. **Spell Checking**

   * Find dictionary word with minimum edit distance to input
   * Example: *recieve → receive*

2. **Information Retrieval**

   * Match queries with spelling variations

3. **Plagiarism Detection**

   * Compare similarity of documents

4. **Computational Biology**

   * DNA sequence alignment (edit distance between gene sequences)

📖 [Applications of Edit Distance](https://en.wikipedia.org/wiki/Edit_distance#Applications)


## 🧠 Example Task

Query: *intension*
Candidate dictionary words:

* *intention* (distance 1)
* *in tension* (distance 2)

Likely correction: **intention**


## 📌 Summary

* **Edit distance** is a fundamental similarity metric in NLP.
* Variants (Hamming, Damerau-Levenshtein, weighted) adapt it to different use cases.
* Widely applied in **spell checking, search engines, plagiarism detection, and bioinformatics**.


## References

* [Edit Distance](https://en.wikipedia.org/wiki/Edit_distance)
* [Levenshtein Distance](https://en.wikipedia.org/wiki/Levenshtein_distance)
* [Hamming Distance](https://en.wikipedia.org/wiki/Hamming_distance)
* [Damerau–Levenshtein Distance](https://en.wikipedia.org/wiki/Damerau%E2%80%93Levenshtein_distance)
* [Applications of Edit Distance](https://en.wikipedia.org/wiki/Edit_distance#Applications)


# Natural Language Processing – Lecture 8

## 📊 N-Gram Language Models

This lecture introduces **n-gram models**, a fundamental approach for **language modeling** and **probabilistic text analysis**.


## 🧠 What is a Language Model?

* A **language model (LM)** assigns a **probability** to a sequence of words.
* Example:

  * $P(\text{“I am a student”})$ > $P(\text{“Student a am I”})$

📖 [Language Model (Wikipedia)](https://en.wikipedia.org/wiki/Language_model)


## 🔢 The Chain Rule of Probability

For a sequence of words $w_1, w_2, ..., w_n$:

$$
P(w_1, w_2, ..., w_n) = P(w_1) \cdot P(w_2|w_1) \cdot P(w_3|w_1, w_2) \cdot ... \cdot P(w_n|w_1, ..., w_{n-1})
$$

* But exact computation is infeasible due to **data sparsity**.


## ✂️ Markov Assumption

* Approximate by considering only **last few words**:

$$
  P(w_n | w_1, w_2, ..., w_{n-1}) \approx P(w_n | w_{n-(n-1)}, ..., w_{n-1})
$$



## 📌 N-Gram Models

### 1. **Unigram Model**

* Assumes independence:

$$
  P(w_1, w_2, ..., w_n) = \prod_i P(w_i)
$$

### 2. **Bigram Model**

* Considers pairs of words:

$$
  P(w_1, w_2, ..., w_n) \approx \prod_i P(w_i | w_{i-1})
$$

### 3. **Trigram Model**

* Considers triples of words:

$$
  P(w_1, w_2, ..., w_n) \approx \prod_i P(w_i | w_{i-2}, w_{i-1})
$$

📖 [N-Gram Model](https://en.wikipedia.org/wiki/N-gram)



## 📊 Estimation from Corpus

* **Maximum Likelihood Estimation (MLE):**

  * Bigram probability:

$$
    P(w_i | w_{i-1}) = \frac{C(w_{i-1}, w_i)}{C(w_{i-1})}
$$

  where:

  * $C(x)$ = count of occurrence in corpus



## ⚠️ Data Sparsity Problem

* Many valid word sequences may **never appear** in training data.
* Example:

  * "He likes mango juice" may not occur in corpus, but is valid.



## 🧮 Smoothing Techniques

1. **Add-One (Laplace) Smoothing**

   * Add 1 to all counts.
   * Problem: Overestimates rare events.

2. **Add-k Smoothing**

   * Add small $k$ instead of 1.

3. **Good-Turing Smoothing**

   * Re-estimates probabilities of **unseen events**.

4. **Backoff and Interpolation**

   * Use higher-order n-gram if available, otherwise backoff to lower-order.

📖 [Smoothing (NLP)](https://en.wikipedia.org/wiki/Additive_smoothing)


## 🔍 Applications of N-Gram Models

* **Spell correction**
* **Query completion** in search engines
* **Speech recognition**
* **Text generation**



## 📌 Summary

* **N-gram models** approximate language probability using limited context.
* **Unigram, bigram, trigram** models balance complexity vs accuracy.
* **Smoothing** addresses the problem of unseen words.
* They form the basis for many NLP tasks, though modern deep learning models (e.g., [RNNs](https://en.wikipedia.org/wiki/Recurrent_neural_network), [Transformers](https://en.wikipedia.org/wiki/Transformer_%28machine_learning_model%29)) now dominate.



## References

* [Language Model](https://en.wikipedia.org/wiki/Language_model)
* [N-Gram](https://en.wikipedia.org/wiki/N-gram)
* [Additive Smoothing](https://en.wikipedia.org/wiki/Additive_smoothing)
* [Good-Turing Estimation](https://en.wikipedia.org/wiki/Good%E2%80%93Turing_frequency_estimation)
* [Recurrent Neural Network](https://en.wikipedia.org/wiki/Recurrent_neural_network)
* [Transformer Model](https://en.wikipedia.org/wiki/Transformer_%28machine_learning_model%29)


# Natural Language Processing – Lecture 9

## 🧮 Evaluation of Language Models

This lecture covers **how to evaluate n-gram language models**, focusing on metrics like **perplexity**, and explores their applications.



## 🎯 Why Evaluate?

* To check **how well a language model predicts unseen text**.
* Evaluation helps in:

  * Comparing models (bigram vs trigram, etc.)
  * Measuring **generalization** beyond training data



## 📊 Evaluation Metrics

### 1. **Likelihood**

* Measure: Probability assigned to a test corpus
* Problem: Direct probabilities are extremely small (due to long sequences)



### 2. **Cross-Entropy**

$$
H = -\frac{1}{N} \sum_{i=1}^{N} \log_2 P(w_i | context)
$$

* $N$ = number of words
* Lower $H$ means better prediction

📖 [Cross-Entropy (Wikipedia)](https://en.wikipedia.org/wiki/Cross_entropy)



### 3. **Perplexity**

* Standard metric for language models:

$$
PP(W) = 2^H = P(w_1, w_2, ..., w_N)^{-\frac{1}{N}}
$$

* Interpretation: **Average branching factor** of the model
* Lower perplexity = better model

📖 [Perplexity (Wikipedia)](https://en.wikipedia.org/wiki/Perplexity)



## ⚡ Practical Notes on Perplexity

* Sensitive to **training/test mismatch**
* Works best when:

  * Training and test corpora are from the **same domain**
* Example:

  * A news-trained model may fail on conversational data



## 🧩 Applications of N-Gram Language Models

1. **Spell Correction**

   * Rank candidate words by probability

2. **Autocomplete / Query Completion**

   * Predict next word in search engines

3. **Speech Recognition**

   * Disambiguate between homophones using context

4. **Machine Translation**

   * Ensure fluency of translated sentences

📖 [Statistical Language Modeling Applications](https://en.wikipedia.org/wiki/Language_model#Applications)



## 📌 Example

Sentence: *“I want to eat …”*

* Bigram model:

  * $P(\text{“food”}|\text{eat})$ vs $P(\text{“sleep”}|\text{eat})$
  * Likely: "food"



## 📌 Summary

* Evaluation of language models is essential to measure performance.
* **Perplexity** is the most widely used metric.
* Lower perplexity = better predictive power.
* Applications: **spell correction, autocomplete, speech recognition, translation**.



## References

* [Cross-Entropy](https://en.wikipedia.org/wiki/Cross_entropy)
* [Perplexity](https://en.wikipedia.org/wiki/Perplexity)
* [Language Model Applications](https://en.wikipedia.org/wiki/Language_model#Applications)


# Natural Language Processing – Lecture 10

## 📊 Part-of-Speech (POS) Tagging – Introduction

This lecture introduces **Part-of-Speech (POS) tagging**, a key step in NLP for assigning **grammatical categories** (e.g., noun, verb, adjective) to words.


## 🧠 What is POS Tagging?

* **Definition:** Assigning a syntactic category (POS tag) to each word in a sentence.
* Example:

  * *“Time flies like an arrow”*

    * Time → Noun
    * flies → Verb
    * like → Preposition
    * an → Article
    * arrow → Noun

📖 [POS Tagging (Wikipedia)](https://en.wikipedia.org/wiki/Part-of-speech_tagging)


## 📚 POS Tagsets

* **Penn Treebank Tagset** (most widely used in English NLP)

  * NN = noun, VB = verb, JJ = adjective, RB = adverb
* Other tagsets:

  * Brown Corpus Tagset
  * Universal POS Tagset (cross-lingual)

📖 [Penn Treebank](https://en.wikipedia.org/wiki/Penn_Treebank)


## 🧩 Ambiguity in POS Tagging

* Words can have **multiple POS tags** depending on context.
* Example:

  * *“Book a flight”* → book = Verb
  * *“Read a book”* → book = Noun

This is a core challenge in tagging.


## 🔢 Approaches to POS Tagging

### 1. **Rule-Based Tagging**

* Handcrafted rules based on grammar + context
* Example: “if a word ends in *-ing*, it’s probably a verb”
* Limitation: Requires **linguistic expertise**

📖 [Rule-Based Tagging](https://en.wikipedia.org/wiki/Part-of-speech_tagging#Rule-based_tagging)


### 2. **Statistical Tagging**

* Uses probability models (trained from annotated corpora).
* Example methods:

  * **Hidden Markov Models (HMMs)**
  * **N-gram models** for tag sequences
* Finds the **most likely sequence of tags** given the sentence.

📖 [Hidden Markov Model](https://en.wikipedia.org/wiki/Hidden_Markov_model)


### 3. **Transformation-Based Tagging (TBL)**

* Introduced by Eric Brill (called **Brill Tagger**).
* Learns transformation rules from a tagged corpus.
* Hybrid of rule-based and statistical.

📖 [Brill Tagger](https://en.wikipedia.org/wiki/Brill_tagger)


### 4. **Neural Tagging**

* Modern approach: Deep learning models

  * **RNNs, LSTMs, Transformers** ([BERT](https://en.wikipedia.org/wiki/BERT_%28language_model%29))
* Use **contextual embeddings** to resolve ambiguities.
* Achieve state-of-the-art accuracy.


## 🔍 Applications of POS Tagging

* **Information extraction**
* **Named Entity Recognition (NER)**
* **Parsing**
* **Machine translation**
* **Speech recognition**


## 📌 Example Sentence

Sentence: *“Can you can a can as a canner can can a can?”*

* can → modal verb
* can → main verb (preserve food)
* can → noun (container)

Demonstrates **POS ambiguity** and importance of context.


## 📌 Summary

* POS tagging assigns **syntactic roles** to words.
* Challenges: **ambiguity, context-dependence**.
* Approaches:

  * **Rule-based**
  * **Statistical (HMM, n-grams)**
  * **Transformation-based**
  * **Neural (modern SOTA)**
* Essential for higher-level NLP tasks.


## References

* [POS Tagging](https://en.wikipedia.org/wiki/Part-of-speech_tagging)
* [Penn Treebank](https://en.wikipedia.org/wiki/Penn_Treebank)
* [Hidden Markov Models](https://en.wikipedia.org/wiki/Hidden_Markov_model)
* [Brill Tagger](https://en.wikipedia.org/wiki/Brill_tagger)
* [BERT](https://en.wikipedia.org/wiki/BERT_%28language_model%29)


# Natural Language Processing – Lecture 11

## 🔎 Part-of-Speech (POS) Tagging – Hidden Markov Models

This lecture explains how **Hidden Markov Models (HMMs)** can be applied to **POS tagging**, building on the statistical approach introduced earlier.


## 🧠 Recap: POS Tagging Problem

* Input: Sentence = sequence of words
* Output: Sequence of **POS tags** (one per word)
* Challenge: Words can belong to **multiple categories** depending on context

  * Example: *book* → Noun (*“Read a book”*) vs Verb (*“Book a flight”*)

📖 [POS Tagging (Wikipedia)](https://en.wikipedia.org/wiki/Part-of-speech_tagging)


## 🎲 Hidden Markov Models (HMMs)

### Key Idea

* Model tagging as a **sequence prediction problem**.
* Assume:

  * **Tags** are hidden states
  * **Words** are observed emissions

📖 [Hidden Markov Model (Wikipedia)](https://en.wikipedia.org/wiki/Hidden_Markov_model)


### HMM Components

1. **States** → POS tags (e.g., NN, VB, JJ)
2. **Observations** → Words in the sentence
3. **Transition Probabilities**

   * $P(t_i | t_{i-1})$ → probability of a tag given the previous tag
4. **Emission Probabilities**

   * $P(w_i | t_i)$ → probability of a word given its tag


## 🔢 The Tagging Task

Goal:
Find the **most likely sequence of tags** $T = t_1, t_2, …, t_n$ for given words $W = w_1, w_2, …, w_n$.

Formally:

$$
\hat{T} = \arg\max_T P(T | W)
$$

Using Bayes’ Rule:

$$
P(T | W) \propto P(W | T) \cdot P(T)
$$

* $P(T)$ → Transition probabilities
* $P(W|T)$ → Emission probabilities


## 🧮 Example

Sentence: *“Fish swim”*

* Possible tags:

  * *Fish*: Noun or Verb
  * *Swim*: Verb or Noun

HMM resolves ambiguity by maximizing joint probability of tags + words.


## 🛠 The Viterbi Algorithm

* **Dynamic programming algorithm** for finding the best tag sequence.
* Efficiently computes the **most likely path** through HMM states.
* Time complexity: $O(N \cdot T^2)$

  * $N$ = sentence length
  * $T$ = number of tags

📖 [Viterbi Algorithm (Wikipedia)](https://en.wikipedia.org/wiki/Viterbi_algorithm)


## ⚠️ Challenges with HMM POS Tagging

1. **Data sparsity** – Rare transitions or emissions may have zero probability

   * Solution: **Smoothing** techniques
2. **Unknown words (OOV problem)**

   * Handle with morphological rules or character-level models
3. **Independence assumptions** (Markov, emission) are often too simplistic


## 📊 Modern Alternatives

* **Conditional Random Fields (CRFs)**
* **Neural models** (RNNs, LSTMs, Transformers like [BERT](https://en.wikipedia.org/wiki/BERT_%28language_model%29))
* Outperform HMMs by capturing **longer dependencies** and **rich features**

📖 [Conditional Random Fields](https://en.wikipedia.org/wiki/Conditional_random_field)


## 📌 Summary

* POS tagging with HMM:

  * Tags = hidden states
  * Words = emissions
  * Transition + emission probabilities define model
* **Viterbi algorithm** used to find best tag sequence
* Limitations: data sparsity, OOV handling, independence assumptions
* Modern approaches: **CRFs, neural networks** outperform HMMs


## References

* [POS Tagging](https://en.wikipedia.org/wiki/Part-of-speech_tagging)
* [Hidden Markov Model](https://en.wikipedia.org/wiki/Hidden_Markov_model)
* [Viterbi Algorithm](https://en.wikipedia.org/wiki/Viterbi_algorithm)
* [Conditional Random Field](https://en.wikipedia.org/wiki/Conditional_random_field)
* [BERT](https://en.wikipedia.org/wiki/BERT_%28language_model%29)


# Natural Language Processing – Lecture 13

## 🌳 Introduction to Syntax and Parsing

This lecture introduces **syntax** in NLP, focusing on how sentences are structured and how **parsers** analyze grammatical relations between words.



## 🧠 What is Syntax?

* **Syntax** = Study of **sentence structure** and how words combine to form phrases/clauses.
* Important for:

  * Understanding **grammatical relationships**
  * Building higher-level NLP applications (translation, QA, etc.)

📖 [Syntax (Wikipedia)](https://en.wikipedia.org/wiki/Syntax)


## 📖 Phrase Structure in Language

* Sentences are not just sequences of words — they have **hierarchical structure**.
* Example:

  * Sentence: *“The boy saw the man with a telescope.”*
  * Ambiguity:

    * Did the **boy** have the telescope?
    * Or the **man**?

📖 [Parse Trees](https://en.wikipedia.org/wiki/Parse_tree)


## 🧩 Grammar Formalisms

### 1. **Context-Free Grammar (CFG)**

* A grammar consists of:

  * **Non-terminals** (syntactic categories like NP = Noun Phrase, VP = Verb Phrase)
  * **Terminals** (actual words)
  * **Production rules**
  * **Start symbol** (usually S = Sentence)

* Example rules:

  * $S \to NP \; VP$
  * $NP \to Det \; N$
  * $VP \to V \; NP$

📖 [Context-Free Grammar](https://en.wikipedia.org/wiki/Context-free_grammar)



### 2. **Parse Trees**

* Show **hierarchical structure** of a sentence.
* Example:

  * *“The cat sat on the mat”*
  * Tree structure:

    * S → NP VP
    * NP → Det N
    * VP → V PP


## ⚙️ Parsing

### Definition

* **Parsing** = Process of analyzing a sentence according to a grammar to produce its structure.

📖 [Parsing (Wikipedia)](https://en.wikipedia.org/wiki/Parsing)



### Types of Parsers

1. **Top-Down Parsing**

   * Start from root (S) and expand until terminals match input words
   * May generate many invalid parses

2. **Bottom-Up Parsing**

   * Start from words and combine upwards until reaching root
   * May attempt invalid combinations

3. **Chart Parsing**

   * Dynamic programming approach
   * Avoids recomputation of substructures

📖 [Chart Parser](https://en.wikipedia.org/wiki/Chart_parser)


## ⚠️ Parsing Ambiguities

* Multiple valid parses are often possible.
* Example: *“I saw the man with the telescope.”*

  * Two possible parse trees
* Leads to **syntactic ambiguity** problem.

📖 [Syntactic Ambiguity](https://en.wikipedia.org/wiki/Syntactic_ambiguity)



## 📊 Applications of Parsing

* **Machine Translation** (syntactic alignment between languages)
* **Information Extraction** (finding subject–verb–object relations)
* **Question Answering** (interpreting query structure)
* **Dialogue Systems**


## 📌 Summary

* **Syntax** studies how words form grammatical structures.
* **CFGs** and **parse trees** model hierarchical structure.
* **Parsing algorithms** (top-down, bottom-up, chart) build syntactic trees.
* Parsing faces **ambiguity** but is critical for high-level NLP tasks.



## References

* [Syntax](https://en.wikipedia.org/wiki/Syntax)
* [Parse Tree](https://en.wikipedia.org/wiki/Parse_tree)
* [Context-Free Grammar](https://en.wikipedia.org/wiki/Context-free_grammar)
* [Parsing](https://en.wikipedia.org/wiki/Parsing)
* [Chart Parser](https://en.wikipedia.org/wiki/Chart_parser)
* [Syntactic Ambiguity](https://en.wikipedia.org/wiki/Syntactic_ambiguity)


# Natural Language Processing – Lecture 14

## 🌳 Parsing Algorithms

This lecture continues the study of **syntax and parsing**, focusing on **algorithms** for building parse trees from context-free grammars (CFGs).



## ⚙️ Parsing as a Computational Problem

* Input: Sentence = sequence of words
* Output: **Parse tree** showing syntactic structure
* Task: Decide whether the sentence belongs to the grammar and, if yes, generate its structure

📖 [Parsing (Wikipedia)](https://en.wikipedia.org/wiki/Parsing)


## 🔼 Top-Down Parsing

### Approach

* Start from **start symbol (S)**
* Expand using grammar rules until terminals match input

### Example

* Grammar:

  * $S \to NP \; VP$
  * $NP \to Det \; N$
  * $VP \to V \; NP$
* Input: *“The cat chased the dog”*

### Pros

* Easy to implement
* Explores possible parses systematically

### Cons

* May generate many **invalid parses**
* Redundant work (recomputing same subtrees)

📖 [Top-Down Parsing](https://en.wikipedia.org/wiki/Top-down_parsing)



## 🔽 Bottom-Up Parsing

### Approach

* Start from **words (terminals)**
* Combine into larger constituents until reaching S

### Example

* Input: *“The cat chased the dog”*
* Begin with words → form NP and VP → assemble into S

### Pros

* Efficient when valid parses exist

### Cons

* May build many **incorrect structures**

📖 [Bottom-Up Parsing](https://en.wikipedia.org/wiki/Bottom-up_parsing)



## 📊 Chart Parsing (Dynamic Programming)

* **Dynamic programming** method that avoids redundant recomputation
* Stores intermediate results in a **chart** (table)

### Example Algorithm: **CYK Parser**

* Works with **CFGs in Chomsky Normal Form (CNF)**
* Time complexity: $O(n^3)$ for sentence length $n$
* Useful for theoretical and practical parsing tasks

📖 [CYK Algorithm](https://en.wikipedia.org/wiki/CYK_algorithm)



## ⚠️ Parsing Ambiguity

* Many sentences allow **multiple parse trees**
* Example: *“I saw the man with the telescope”*

  * Ambiguity: Who had the telescope?

### Handling Ambiguity

1. Generate **all possible parses**
2. Rank parses using **probabilistic models** (e.g., [Probabilistic CFG](https://en.wikipedia.org/wiki/Stochastic_context-free_grammar))



## 🧩 Applications of Parsing

* **Machine Translation** (syntactic alignment across languages)
* **Information Extraction** (subject–verb–object relations)
* **Text Summarization** (understanding sentence structure)
* **Dialogue Systems**



## 📌 Summary

* Parsing algorithms analyze sentence structure using CFGs.
* **Top-down** and **bottom-up** methods are simple but inefficient.
* **Chart parsing** (CYK) reduces redundant work via dynamic programming.
* Ambiguity remains a challenge → handled using **probabilistic parsing**.



## References

* [Parsing](https://en.wikipedia.org/wiki/Parsing)
* [Top-Down Parsing](https://en.wikipedia.org/wiki/Top-down_parsing)
* [Bottom-Up Parsing](https://en.wikipedia.org/wiki/Bottom-up_parsing)
* [CYK Algorithm](https://en.wikipedia.org/wiki/CYK_algorithm)
* [Probabilistic CFG](https://en.wikipedia.org/wiki/Stochastic_context-free_grammar)


# Natural Language Processing – Lecture 15

## 📊 Probabilistic Parsing

This lecture introduces **probabilistic approaches to parsing**, where parse trees are ranked using probabilities instead of treating all parses equally.


## ⚠️ Why Probabilistic Parsing?

* **Problem:** Many sentences are **syntactically ambiguous**.

  * Example: *“I saw the man with the telescope.”*
  * Ambiguity: Who had the telescope?
* **Solution:** Assign probabilities to parses → choose the most likely one.

📖 [Syntactic Ambiguity](https://en.wikipedia.org/wiki/Syntactic_ambiguity)


## 🧠 Probabilistic Context-Free Grammars (PCFGs)

### Definition

* A **Context-Free Grammar (CFG)** where each production rule has a **probability**.
* Example:

  * $S \to NP \; VP$ $P = 1.0$
  * $NP \to Det \; N$ $P = 0.6$
  * $NP \to NP \; PP$ $P = 0.4$

📖 [Stochastic Context-Free Grammar](https://en.wikipedia.org/wiki/Stochastic_context-free_grammar)



### Parse Probability

* The probability of a parse tree is the **product of the rule probabilities** used.

$$
P(\text{Parse}) = \prod_i P(\text{Rule}_i)
$$

* Select parse with **maximum probability**.



## 🧮 Example

Sentence: *“The cat saw the dog with a telescope”*

* Parse 1: *dog has telescope*
* Parse 2: *cat used telescope*
* PCFG assigns probabilities → more likely parse chosen.


## ⚙️ Estimating Probabilities

1. **From a Treebank**

   * Example: [Penn Treebank](https://en.wikipedia.org/wiki/Penn_Treebank)
   * Count frequency of rule usage in annotated corpus.
   * Rule probability:

$$
     P(A \to \beta) = \frac{\text{Count}(A \to \beta)}{\text{Count}(A)}
$$

2. **Smoothing**

   * Handle rare/unseen rules
   * Techniques: Add-k, Good-Turing, backoff


## 📊 Parsing with PCFGs

* Algorithms:

  * **Probabilistic CYK Parsing**
  * **Probabilistic Chart Parsing**
* Select the **most probable parse tree**.

📖 [CYK Algorithm](https://en.wikipedia.org/wiki/CYK_algorithm)


## ⚠️ Limitations of PCFGs

1. **Context Independence**

   * Assumes independence of rules (unrealistic).
2. **Preference Bias**

   * Overgenerates shallow parses (shorter trees often get higher probability).
3. **Lexical Information Missing**

   * Doesn’t incorporate word-level preferences.


## 🔍 Improvements Beyond PCFG

* **Lexicalized PCFGs** → include head words in rules
* **Dependency Parsing** → focus on word-to-word relations
* **Neural Parsers** → use embeddings & deep models (e.g., [Transformers](https://en.wikipedia.org/wiki/Transformer_%28machine_learning_model%29))


## 📌 Summary

* **PCFGs** assign probabilities to grammar rules → rank parses.
* Parse probability = product of rule probabilities.
* Trained from **treebanks** like Penn Treebank.
* PCFGs improve parsing but have limitations → addressed by lexicalized and neural models.


## References

* [Syntactic Ambiguity](https://en.wikipedia.org/wiki/Syntactic_ambiguity)
* [Stochastic Context-Free Grammar](https://en.wikipedia.org/wiki/Stochastic_context-free_grammar)
* [Penn Treebank](https://en.wikipedia.org/wiki/Penn_Treebank)
* [CYK Algorithm](https://en.wikipedia.org/wiki/CYK_algorithm)
* [Transformer Model](https://en.wikipedia.org/wiki/Transformer_%28machine_learning_model%29)



# Natural Language Processing – Lecture 16

## 🔗 Dependency Parsing

This lecture introduces **dependency parsing**, an alternative to phrase-structure parsing, where syntax is represented as **relations between words** rather than hierarchical constituents.


## 🧠 What is Dependency Parsing?

* Focuses on **binary relations** between words: **head → dependent**.
* Each word (except the root) depends on another word.
* Produces a **dependency tree** instead of a phrase-structure tree.

📖 [Dependency Grammar](https://en.wikipedia.org/wiki/Dependency_grammar)


## 📊 Example

Sentence: *“The cat chased the mouse”*

* Dependencies:

  * chased (ROOT)
  * cat → subject of chased
  * mouse → object of chased
  * the → determiner of cat
  * the → determiner of mouse

Result: **Head-dependent tree** showing grammatical functions.


## 📚 Dependency Relations

Common dependency labels (from [Universal Dependencies](https://universaldependencies.org/)):

* **nsubj** → nominal subject
* **dobj** → direct object
* **det** → determiner
* **amod** → adjectival modifier
* **prep** → prepositional modifier

Example: *“She ate the red apple.”*

* nsubj(she, ate)
* dobj(apple, ate)
* det(the, apple)
* amod(red, apple)


## ⚙️ Dependency Parsing vs Phrase-Structure Parsing

| Aspect         | Phrase-Structure Parsing           | Dependency Parsing                          |
| -------------- | ---------------------------------- | ------------------------------------------- |
| Representation | Constituents (NP, VP, etc.)        | Word-to-word dependencies                   |
| Tree           | Hierarchical, nested phrases       | Flat head–dependent arcs                    |
| Useful for     | Syntax analysis, linguistic theory | Information extraction, relation extraction |

📖 [Constituency vs Dependency](https://en.wikipedia.org/wiki/Dependency_grammar#Comparison_with_constituency_grammar)


## 🛠 Dependency Parsing Algorithms

### 1. **Transition-Based Parsing**

* Incrementally build parse tree using actions:

  * SHIFT, REDUCE, LEFT-ARC, RIGHT-ARC
* Example: **Arc-Standard Algorithm**
* Efficient: Linear-time parsing ($O(n)$)

📖 [Transition-Based Parsing](https://en.wikipedia.org/wiki/Dependency_grammar#Transition-based_parsing)


### 2. **Graph-Based Parsing**

* Treat parsing as finding **maximum spanning tree** over words.
* Use global optimization over possible parses.
* Typically slower ($O(n^2)$ or $O(n^3)$).

📖 [Graph-Based Dependency Parsing](https://en.wikipedia.org/wiki/Dependency_grammar#Graph-based_parsing)


## ⚠️ Challenges in Dependency Parsing

* **Ambiguity**: Multiple valid parses possible.
* **Non-projective dependencies** (crossing arcs) → common in free word-order languages like Czech, Russian.
* Requires **large annotated corpora** (e.g., Universal Dependencies).

📖 [Non-projective Parsing](https://en.wikipedia.org/wiki/Dependency_grammar#Projectivity)


## 🔍 Applications of Dependency Parsing

* **Information Extraction** → subject–verb–object relations
* **Relation Extraction** → who did what to whom
* **Question Answering** → syntactic dependencies help identify answers
* **Machine Translation** → syntax-informed alignments



## 📌 Summary

* Dependency parsing represents syntax as **head-dependent relations**.
* Two main approaches: **transition-based** (efficient) and **graph-based** (global).
* Challenges: ambiguity, non-projective structures, domain adaptation.
* Widely applied in **IE, QA, MT, semantic parsing**.


## References

* [Dependency Grammar](https://en.wikipedia.org/wiki/Dependency_grammar)
* [Universal Dependencies](https://universaldependencies.org/)
* [Transition-Based Parsing](https://en.wikipedia.org/wiki/Dependency_grammar#Transition-based_parsing)
* [Graph-Based Parsing](https://en.wikipedia.org/wiki/Dependency_grammar#Graph-based_parsing)
* [Projectivity in Parsing](https://en.wikipedia.org/wiki/Dependency_grammar#Projectivity)


# Natural Language Processing – Lecture 17

## 🧠 Semantics in NLP – Introduction

This lecture introduces **semantics**, the study of **meaning in language**, and its role in NLP systems.


## 📖 What is Semantics?

* **Semantics** = study of **word, phrase, and sentence meaning**.
* Goes beyond syntax (structure) to capture **interpretation**.
* Essential for tasks like **translation, question answering, dialogue, and summarization**.

📖 [Semantics (Wikipedia)](https://en.wikipedia.org/wiki/Semantics)


## 🔑 Levels of Meaning

1. **Word-Level Semantics**

   * Meaning of individual words
   * Example: "bank" → financial institution vs riverbank

2. **Sentence-Level Semantics**

   * Compositional meaning from structure + words
   * Example: *“John loves Mary”* vs *“Mary loves John”*

3. **Discourse-Level Semantics**

   * Meaning across multiple sentences
   * Example: Pronoun resolution (*“John went home. He slept.”*)

📖 [Lexical Semantics](https://en.wikipedia.org/wiki/Lexical_semantics)


## ⚠️ Challenges in Semantics

1. **Ambiguity**

   * Lexical: "bark" → tree bark vs dog sound
   * Structural: *“old men and women”*

2. **Polysemy vs Homonymy**

   * Polysemy: word with related senses (*paper = material, publication*)
   * Homonymy: unrelated senses (*bat = animal, cricket bat*)

📖 [Polysemy](https://en.wikipedia.org/wiki/Polysemy) | [Homonymy](https://en.wikipedia.org/wiki/Homonymy)


## 🧮 Representing Meaning

### 1. **First-Order Logic (FOL)**

* Expresses meaning formally with **predicates, variables, quantifiers**.
* Example:

  * Sentence: *“Every student passed the exam”*
  * Representation: $\forall x \, [Student(x) \rightarrow Passed(x, exam)]$

📖 [First-Order Logic](https://en.wikipedia.org/wiki/First-order_logic)


### 2. **Semantic Roles**

* Assigns roles in events (who did what to whom).
* Example: *“Mary gave John a book”*

  * Agent = Mary
  * Recipient = John
  * Theme = Book

📖 [Semantic Role Labeling](https://en.wikipedia.org/wiki/Semantic_role_labeling)


### 3. **Distributional Semantics**

* Word meaning derived from **context of use**.
* Based on the **distributional hypothesis**:

  * *“Words that occur in similar contexts have similar meanings”*.
* Example: "dog" and "cat" occur in similar contexts → semantically close.

📖 [Distributional Semantics](https://en.wikipedia.org/wiki/Distributional_semantics)


## 🧩 Lexical Resources

* **WordNet** → semantic network of words (synonyms, hypernyms, hyponyms).
* Example: "car" → synonym set includes "automobile".

📖 [WordNet](https://en.wikipedia.org/wiki/WordNet)



## 🔍 Applications of Semantics

* **Word Sense Disambiguation (WSD)**
* **Information Retrieval (IR)** → semantic search
* **Machine Translation**
* **Question Answering**
* **Text Summarization**


## 📌 Summary

* Semantics = study of **meaning in language**.
* Levels: word, sentence, discourse.
* Ambiguities (lexical, structural) complicate understanding.
* Representations: **FOL, semantic roles, distributional semantics**.
* Lexical resources like **WordNet** support semantic tasks.


## References

* [Semantics](https://en.wikipedia.org/wiki/Semantics)
* [Lexical Semantics](https://en.wikipedia.org/wiki/Lexical_semantics)
* [Polysemy](https://en.wikipedia.org/wiki/Polysemy)
* [Homonymy](https://en.wikipedia.org/wiki/Homonymy)
* [First-Order Logic](https://en.wikipedia.org/wiki/First-order_logic)
* [Semantic Role Labeling](https://en.wikipedia.org/wiki/Semantic_role_labeling)
* [Distributional Semantics](https://en.wikipedia.org/wiki/Distributional_semantics)
* [WordNet](https://en.wikipedia.org/wiki/WordNet)


# Natural Language Processing – Lecture 18

## 🧠 Word Sense Disambiguation (WSD)

This lecture explores **Word Sense Disambiguation (WSD)**, a core task in semantics that deals with resolving **ambiguity in word meanings**.



## 📖 What is WSD?

* **Definition:** Task of determining which sense of a word is used in a given context.
* Example:

  * *“He deposited money in the bank.”* → bank = financial institution
  * *“He sat by the bank of the river.”* → bank = riverside

📖 [Word Sense Disambiguation (Wikipedia)](https://en.wikipedia.org/wiki/Word-sense_disambiguation)



## ⚠️ Why is WSD Hard?

1. **Polysemy** → One word with multiple related senses (*paper = material, article*).
2. **Homonymy** → Word with unrelated senses (*bat = animal, cricket bat*).
3. **Context Dependence** → Meaning depends on surrounding words.
4. **Granularity** → Fine-grained distinctions in resources like WordNet.

📖 [Polysemy](https://en.wikipedia.org/wiki/Polysemy) | [Homonymy](https://en.wikipedia.org/wiki/Homonymy)



## 🛠 Approaches to WSD

### 1. **Knowledge-Based Methods**

* Use **dictionaries, thesauri, lexical resources**.
* Example: **Lesk Algorithm**

  * Sense disambiguation by **overlap of dictionary definitions**.

📖 [Lesk Algorithm](https://en.wikipedia.org/wiki/Lesk_algorithm)



### 2. **Supervised Learning**

* Treat WSD as a **classification problem**.
* Input: word in context
* Output: correct sense label
* Requires **annotated corpora** (e.g., SemCor).

📖 [Supervised WSD](https://en.wikipedia.org/wiki/Word-sense_disambiguation#Supervised_methods)



### 3. **Unsupervised Learning**

* Cluster word occurrences into groups → assume each cluster = sense.
* No annotated data required.
* Limitation: clusters may not map cleanly to dictionary senses.

📖 [Unsupervised Learning](https://en.wikipedia.org/wiki/Unsupervised_learning)



### 4. **Neural Approaches**

* Use **word embeddings** + **contextual embeddings (BERT, ELMo, GPT)**.
* Context captures sense directly.
* Example: BERT can distinguish *“bank (money)”* vs *“bank (river)”*.

📖 [BERT](https://en.wikipedia.org/wiki/BERT_%28language_model%29)


## 📊 Evaluation of WSD

* **Gold-standard datasets**: SemCor, Senseval, SemEval.
* Metrics: **Precision, Recall, F1**.
* Example: Accuracy = % of correctly disambiguated words.

📖 [Senseval / SemEval](https://en.wikipedia.org/wiki/SemEval)



## 🔍 Applications of WSD

* **Machine Translation** (choosing correct sense in target language).
* **Information Retrieval** (semantic search).
* **Question Answering** (understanding query intent).
* **Text Summarization** (selecting right meaning for key terms).


## 📌 Summary

* **WSD = identifying correct sense of a word in context**.
* Ambiguity arises from **polysemy, homonymy, context dependence**.
* Approaches: **knowledge-based, supervised, unsupervised, neural**.
* Applications: **MT, IR, QA, summarization**.



## References

* [Word Sense Disambiguation](https://en.wikipedia.org/wiki/Word-sense_disambiguation)
* [Polysemy](https://en.wikipedia.org/wiki/Polysemy)
* [Homonymy](https://en.wikipedia.org/wiki/Homonymy)
* [Lesk Algorithm](https://en.wikipedia.org/wiki/Lesk_algorithm)
* [Supervised Methods](https://en.wikipedia.org/wiki/Word-sense_disambiguation#Supervised_methods)
* [Unsupervised Learning](https://en.wikipedia.org/wiki/Unsupervised_learning)
* [BERT](https://en.wikipedia.org/wiki/BERT_%28language_model%29)
* [SemEval](https://en.wikipedia.org/wiki/SemEval)


# Natural Language Processing – Lecture 19

## 🧩 Distributional Semantics

This lecture covers **distributional semantics**, an approach to modeling word meaning based on **context of usage** in large corpora.



## 📖 The Distributional Hypothesis

* **Idea:** *“Words that occur in similar contexts tend to have similar meanings.”*
* Example:

  * "dog" and "cat" both occur near *pet, animal, fur, feed* → semantically similar.

📖 [Distributional Semantics (Wikipedia)](https://en.wikipedia.org/wiki/Distributional_semantics)


## 📊 Representing Words in Vector Space

### 1. **Co-occurrence Matrices**

* Rows = target words
* Columns = context words
* Entries = frequency (or weighted frequency) of co-occurrence
* Example: “dog” appears often with “bark, pet, animal”

📖 [Vector Space Model](https://en.wikipedia.org/wiki/Vector_space_model)


### 2. **Dimensionality Reduction**

* Raw co-occurrence vectors are **high-dimensional & sparse**
* Apply techniques like:

  * **Singular Value Decomposition (SVD)**
  * **Latent Semantic Analysis (LSA)**

📖 [Latent Semantic Analysis](https://en.wikipedia.org/wiki/Latent_semantic_analysis)


## 🛠 Word Similarity

* Compute similarity between word vectors using:

  * **Cosine similarity**
  * **Euclidean distance**

Example:

* sim(dog, cat) > sim(dog, car)

📖 [Cosine Similarity](https://en.wikipedia.org/wiki/Cosine_similarity)


## ⚙️ Weighting Schemes

### 1. **TF-IDF (Term Frequency–Inverse Document Frequency)**

* Reduces weight of frequent but uninformative words.

### 2. **PPMI (Positive Pointwise Mutual Information)**

* Measures association strength between word and context.

📖 [TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) | [PMI](https://en.wikipedia.org/wiki/Pointwise_mutual_information)



## 🌍 From Word Vectors to Word Embeddings

* Distributional semantics → foundation of **neural word embeddings**.
* Word2Vec, GloVe, FastText build on these ideas.

📖 [Word Embedding](https://en.wikipedia.org/wiki/Word_embedding)



## 🔍 Applications of Distributional Semantics

* **Synonym detection**
* **Thesaurus construction**
* **Information retrieval** (semantic search)
* **Text classification & clustering**
* **Word embeddings training**



## 📌 Summary

* **Distributional hypothesis:** meaning from context.
* Represented using **co-occurrence matrices** and reduced with **SVD/LSA**.
* Word similarity measured via **cosine similarity** or PMI.
* Basis for **modern embeddings** like Word2Vec, GloVe, FastText.



## References

* [Distributional Semantics](https://en.wikipedia.org/wiki/Distributional_semantics)
* [Vector Space Model](https://en.wikipedia.org/wiki/Vector_space_model)
* [Latent Semantic Analysis](https://en.wikipedia.org/wiki/Latent_semantic_analysis)
* [Cosine Similarity](https://en.wikipedia.org/wiki/Cosine_similarity)
* [TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)
* [Pointwise Mutual Information](https://en.wikipedia.org/wiki/Pointwise_mutual_information)
* [Word Embedding](https://en.wikipedia.org/wiki/Word_embedding)

# Natural Language Processing – Lecture 20

## 🔡 Word Embeddings – Introduction

This lecture introduces **word embeddings**, dense vector representations of words that capture **semantic and syntactic relationships**.


## 📖 What are Word Embeddings?

* A method of representing words as **low-dimensional continuous vectors**.
* Unlike sparse one-hot vectors, embeddings capture **semantic similarity**.
* Example:

  * *vector(dog)* ≈ *vector(cat)* (close in space)
  * *vector(dog)* far from *vector(car)*

📖 [Word Embedding (Wikipedia)](https://en.wikipedia.org/wiki/Word_embedding)


## 🔢 From Distributional Semantics to Embeddings

* Based on the **distributional hypothesis**: words in similar contexts have similar meanings.
* Word embeddings are **learned automatically** from large corpora.
* Improve over **co-occurrence matrices + SVD** by using **neural networks**.

📖 [Distributional Semantics](https://en.wikipedia.org/wiki/Distributional_semantics)


## 🧩 Properties of Word Embeddings

1. **Semantic Similarity**

   * Cosine similarity between vectors captures meaning.
   * Example: sim(“king”, “queen”) > sim(“king”, “car”).

2. **Analogical Reasoning**

   * Famous property:

$$
     \text{vector(king)} - \text{vector(man)} + \text{vector(woman)} \approx \text{vector(queen)}
$$

3. **Clustering**

   * Similar words form clusters: animals, professions, places.


## 🛠 Techniques for Learning Word Embeddings

### 1. **Word2Vec**

* Introduced by Mikolov et al. (2013).
* Two architectures:

  * **CBOW (Continuous Bag-of-Words)** → predict word from context.
  * **Skip-gram** → predict context from word.

📖 [Word2Vec](https://en.wikipedia.org/wiki/Word2vec)


### 2. **GloVe (Global Vectors)**

* Uses **global co-occurrence statistics**.
* Factorizes word–context matrix.

📖 [GloVe](https://en.wikipedia.org/wiki/GloVe_%28machine_learning%29)


### 3. **FastText**

* Extends Word2Vec by using **subword information**.
* Helps handle rare/unknown words.

📖 [FastText](https://en.wikipedia.org/wiki/FastText)


## 📊 Comparing Representations

| Representation       | Type   | Dimensionality | Captures Meaning? |   |      |
| -------------------- | ------ | -------------- | ----------------- | - | ---- |
| One-Hot Vector       | Sparse |                | V                 |   | ❌ No |
| Co-occurrence Vector | Sparse | High           | ✅ Partial         |   |      |
| Word Embedding       | Dense  | 100–300        | ✅ Yes             |   |      |


## 🔍 Applications of Word Embeddings

* **Information Retrieval** (semantic search)
* **Text Classification** (sentiment analysis, spam detection)
* **Machine Translation** (cross-lingual embeddings)
* **Question Answering**
* **Recommendation Systems**


## 📌 Summary

* Word embeddings = **dense, low-dimensional word vectors**.
* Capture **semantic similarity & analogical relationships**.
* Learned using **Word2Vec, GloVe, FastText**.
* Widely used in **IR, MT, QA, classification**.


## References

* [Word Embedding](https://en.wikipedia.org/wiki/Word_embedding)
* [Distributional Semantics](https://en.wikipedia.org/wiki/Distributional_semantics)
* [Word2Vec](https://en.wikipedia.org/wiki/Word2vec)
* [GloVe](https://en.wikipedia.org/wiki/GloVe_%28machine_learning%29)
* [FastText](https://en.wikipedia.org/wiki/FastText)


# Natural Language Processing – Lecture 21

## 🧮 Word2Vec: Neural Word Embeddings

This lecture explains **Word2Vec**, a widely used method for learning **dense word embeddings** using shallow neural networks.


## 📖 Word2Vec Overview

* Introduced by **Mikolov et al. (2013)** at Google.
* Learns word vectors from large corpora using a **prediction-based approach**.
* Two architectures:

  1. **Continuous Bag-of-Words (CBOW)**
  2. **Skip-gram**

📖 [Word2Vec (Wikipedia)](https://en.wikipedia.org/wiki/Word2vec)


## ⚙️ CBOW Model

### Idea

* Predict the **target word** given its **context words**.

Example:

* Context: *“the \_\_\_ barks loudly”*
* Predict: *“dog”*

### Architecture

* Input: One-hot vectors for context words
* Hidden layer: Shared weight matrix → word embeddings
* Output: Softmax over vocabulary → predicted word


## ⚙️ Skip-Gram Model

### Idea

* Opposite of CBOW: predict **context words** given a **target word**.

Example:

* Input: *“dog”*
* Predict: *“the, barks, loudly”*

### Advantage

* Works better for **rare words**.


## 🔢 Training Word2Vec

### Objective Function

* Maximize probability of predicting correct words.
* Skip-gram with context window $c$:

$$
  \frac{1}{T} \sum_{t=1}^{T} \sum_{-c \leq j \leq c, j \neq 0} \log P(w_{t+j} | w_t)
$$


## ⚡ Efficiency Tricks

1. **Negative Sampling**

   * Instead of softmax over all words, sample a few **negative examples**.
   * Greatly speeds up training.

2. **Hierarchical Softmax**

   * Organize vocabulary as a binary tree.
   * Compute probabilities in $O(\log V)$ instead of $O(V)$.

📖 [Negative Sampling](https://en.wikipedia.org/wiki/Word2vec#Negative_sampling)


## 🧩 Properties of Word2Vec Embeddings

* Capture **semantic similarity** (cosine similarity of vectors).
* Capture **analogical reasoning**:

  * *king – man + woman ≈ queen*
* Cluster words by meaning (animals, professions, places).


## 📊 Applications

* **Semantic Search** → query expansion using embeddings
* **Machine Translation** → bilingual embeddings
* **Recommendation Systems** → items treated as words
* **Text Classification** → features for classifiers



## 📌 Summary

* **Word2Vec** learns embeddings via **CBOW** (predict word from context) or **Skip-gram** (predict context from word).
* Uses **negative sampling** or **hierarchical softmax** for efficiency.
* Embeddings capture **semantic similarity and analogies**.
* Widely applied in NLP and beyond.



## References

* [Word2Vec](https://en.wikipedia.org/wiki/Word2vec)
* [Negative Sampling](https://en.wikipedia.org/wiki/Word2vec#Negative_sampling)
* [Neural Word Embeddings](https://en.wikipedia.org/wiki/Word_embedding)

