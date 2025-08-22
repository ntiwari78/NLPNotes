

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

