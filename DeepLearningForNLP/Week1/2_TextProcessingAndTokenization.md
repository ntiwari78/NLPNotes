

# 📘 Study Guide: Text Processing Basics and Tokenization

---

## 🧱 1. Basics of Text Processing

### 🔹 Identification of Units

* **Tokens**: Individual units (words, punctuation, etc.)
* **Types (Unique Words)**: Distinct tokens
* **Vocabulary**: Set of all types in a text

### 🔹 Heaps’ Law

* Describes vocabulary growth
* Relationship: **V (types)** vs. **N (tokens)**

### 🔹 Corpus Variations

* **Language Diversity**
* **Abbreviations / Code-Switching**
* **Genre & Domain Knowledge**

---

## ✂️ 2. Traditional Tokenization

### 🔹 White Space Tokenizer

* **Common in English**
* **Challenges**: Clitics, punctuation

### 🔹 Handling Unseen Words

* **Out-of-Vocabulary (OOV)**
* **Unknown Token (Unk)**: Problems with oversimplification

### 🔹 Language-Specific Challenges

* **Chinese Segmentation**
* **Sanskrit Sandhi Rules**
* **Morphologically Rich Languages**

---

## 🔄 3. Other Text Processing Steps

* **Lemmatization**
* **Stemming**
* **Sentence Segmentation**
* **Stop Word Removal**
* **Casing (Lowercasing)**

---

## 🧠 4. Modern Tokenization (LLMs)

### 🔹 Subword Tokenization

* **Morpheme-based units**
* **Pre-tokenization vs Tokenization**

### 🔹 Byte Pair Encoding (BPE)

* **Token Learner Phase**
* **Token Segmenter Phase**
* **Iterative Merging**

### 🔹 Alternative Algorithms

* **WordPiece**
* **SentencePiece**

### 🔹 LLM Implementations

* **Models**: GPT-2/3/4, LLaMA 2
* **Vocabulary Sizes**: 30k–100k tokens

---

## ⚠️ 5. Implications

* **Semantic Loss in Subwords**
* **Token-based Cost Disparity**
* **Language Frequency Bias**

---

## 📘 Study Guide: Text Processing and Tokenization

---

### ✍️ Quiz: Short-Answer Questions

Answer in 2–3 sentences based on the course content.

1. What is the primary first step in processing text for an AI or NLP model, and what does it involve?
2. Explain the difference between a "token" and a "type" in the context of a text corpus.
3. Briefly describe Heap's Law and its relationship between vocabulary size and the number of tokens in a corpus.
4. What is an out-of-vocabulary (OOV) token, and what is the traditional method for handling one?
5. Why is simple whitespace tokenization not a viable method for languages like Chinese or Sanskrit?
6. In modern deep learning-based NLP, what is the distinction between "pre-tokenization" and "tokenization"?
7. Describe the initial state of the vocabulary at the beginning of the Byte-Pair Encoding (BPE) token learner process.
8. According to the BPE algorithm, how is the decision made to merge two adjacent tokens into a new, single token?
9. What are the two main problems associated with replacing rare or unseen words with a generic "unknown" (<unk>) token?
10. Explain a significant real-world implication of subword tokenization related to the cost of using large language models.

---

### ✅ Answer Key

1. **Text processing begins with identifying tokens**, converting sequences of characters into discrete units. These tokens are then mapped to IDs from a model's vocabulary for downstream use.

2. A **token** is any instance of a word in text, while a **type** is a unique word. "The cat sat on the mat" contains six tokens but only five types (as "the" appears twice).

3. **Heap’s Law** models the vocabulary growth in relation to tokens. It says V (types) grows roughly as V = K·N^β where N is tokens and 0.5 < β < 1, implying diminishing returns in new word discovery.

4. An **OOV token** is a word not present in the vocabulary. Traditionally, it is replaced with a special **<unk> token**, losing specific word information.

5. **Whitespace tokenization fails in languages like Chinese or Sanskrit** because word boundaries are not marked by spaces and require complex rules (e.g., Chinese segmentation or Sanskrit sandhi rules).

6. **Pre-tokenization** splits text into rough units like whitespace-separated words. **Tokenization** then uses algorithms like BPE to split these into subwords for robust vocabulary handling.

7. The **initial vocabulary in BPE** consists of all characters in the corpus plus an end-of-word marker. Each word is split into characters before merges begin.

8. **BPE merges the most frequent adjacent token pair** in the training data. This process is repeated until a desired vocabulary size is reached.

9. First, **<unk> tokens remove meaningful differences** between unseen words. Second, **the model can't regenerate these words** since they are collapsed into a single generic token.

10. Tokenization affects model cost since **billing is often per-token**. Less frequent languages get split into more tokens, increasing the **financial cost of generation** compared to well-represented ones.

---

### 🧠 Essay Questions

Respond to each question in paragraph form, using concepts directly from the lecture:

1. **Tokenization Evolution**: Trace the development from whitespace tokenization to BPE. What problems of fixed vocabularies were addressed?
2. **BPE Phases**: Describe the "token learner" and "token segmenter" phases of BPE using examples (e.g., "low", "lowest", "newer").
3. **Processing Challenges**: Describe three challenges in text processing such as code-switching, morphological complexity, or domain variation.
4. **Vocabulary in NLP**: Discuss how vocabulary size relates to tokens (Heap’s Law), how OOV words are handled, and how subwords solve key issues.
5. **Negative Impacts of Subwords**: Explain unintended problems like **semantic loss in sentiment analysis** and **cost disparity** in underrepresented languages.

---

### 📚 Glossary of Key Terms

| Term                              | Definition                                                                    |
| --------------------------------- | ----------------------------------------------------------------------------- |
| **Byte-Pair Encoding (BPE)**      | Subword algorithm that builds vocabulary by merging frequent character pairs. |
| **Code-switching**                | Mixing multiple languages/scripts in a sentence.                              |
| **Corpus**                        | A large, structured set of text data.                                         |
| **Heap's Law**                    | Vocabulary size grows sub-linearly with token count: V = K·N^β.               |
| **Lemmatization**                 | Reduces words to dictionary root forms.                                       |
| **Morpheme**                      | Smallest meaningful language unit (e.g., "un-likely-est").                    |
| **Out-of-Vocabulary (OOV) Token** | A word not found in the model's vocabulary.                                   |
| **Pre-tokenization**              | Early splitting of text into basic units (e.g., using spaces).                |
| **SentencePiece**                 | Subword tokenization method that works on raw text.                           |
| **Stemming**                      | Strips suffixes to reduce words to roots.                                     |
| **Subwords**                      | Token units smaller than words, generated from algorithms like BPE.           |
| **Token**                         | Instance of a word or subword in text.                                        |
| **Tokenization**                  | Splitting text into tokens (e.g., words or subwords).                         |
| **Type**                          | Unique word in vocabulary.                                                    |
| **Unknown Token (<unk>)**         | A placeholder token used for all unseen words.                                |
| **Vocabulary**                    | A set of known types used by a model, each with an ID.                        |
| **Whitespace Tokenization**       | Simple tokenization method based on space characters.                         |
| **WordPiece**                     | Subword method used in BERT that optimizes based on likelihood increase.      |

