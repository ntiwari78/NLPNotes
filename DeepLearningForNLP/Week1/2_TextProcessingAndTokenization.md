

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

