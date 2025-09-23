# References
- Lecture1 Notes https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/readings/cs224n_winter2023_lecture1_notes_draft.pdf
- Le ture1 Slides https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/slides/cs224n-spr2024-lecture01-wordvecs1.pdf
# Stanford's Intro Lecture on Language AI. Here Are the 5 Most Surprising Takeaways.
It’s hard to ignore the explosion of AI tools like ChatGPT. They have captured the public imagination, demonstrating an almost magical ability to write, reason, and create. From drafting emails to generating code, these models are becoming a part of our daily digital lives. But while using these tools is fascinating, I’ve found that understanding the foundational principles that make them work is even more so.
Beneath the surface of slick user interfaces lies a deep and surprisingly human story about language itself. To uncover the "secret origin story" of today's AI, I dove into the introductory lecture of Stanford's renowned course on Natural Language Processing with Deep Learning (CS224N), taught by Professor Christopher Manning. The lecture peels back the curtain, revealing that the biggest breakthroughs are not just about bigger computers, but about profound ideas concerning how we think, communicate, and create meaning.
This post distills the five most surprising and impactful insights from that lecture. They offer a new lens through which to view not only artificial intelligence, but also the very nature of human language.
## 1. Language Isn't Just for Talking—It's a Tool for Thinking
Professor Manning begins not with code, but with a fundamental question: what is the purpose of language? While the obvious answer is communication, he argues its primary power might be its role in enabling higher-level cognition. Language isn't just how we talk to each other; it's the operating system for how we think with complexity and depth.
He contrasts human intelligence with that of chimpanzees. In many respects, chimps are remarkably intelligent—they use tools, they can plan, and they even have better short-term memory than humans. Yet, language is the "enormous differentiator" that allowed humans to achieve ascendency. It provides a "scaffolding to do much more detailed thought and planning," allowing us to organize complex ideas in ways other species cannot.
This cognitive leap was supercharged by the invention of writing. At only 5,000 years old, writing is a very recent tool, but its impact was monumental. It allowed knowledge to be shared across time and space, decoupling learning from physical presence. This innovation, Manning notes, was enough to take humanity from the "Bronze Age" to "mobile phones" in a geological blink of an eye.
## 2. The Meaning of a Word Is the Company It Keeps
This next idea reframes the very definition of "meaning" in a way that is essential to understanding AI. How does a computer learn what a word "means"? The traditional approach, denotational semantics, treats meaning like a dictionary entry: a word points to a specific idea. Early computational systems like WordNet were built on this principle, creating vast thesaurus-like networks. However, these systems were brittle, missed nuance, and were always incomplete.
To make this concrete, Manning notes that WordNet might list "proficient" as a synonym for "good." But as he points out, while you might say "that was a good shot," you would almost never say "that was a proficient shot." It just sounds weird. That's the kind of human nuance these systems missed.
The breakthrough for modern AI came from a different idea: distributional semantics. This concept holds that a word's meaning is derived from the contexts in which it appears. The British linguist J.R. Firth summarized this principle perfectly in 1957:
You shall know a word by the company it keeps
This is the core insight behind "word vectors" or "embeddings." By analyzing massive amounts of text, an algorithm can learn that words like hotel and motel appear in very similar contexts (near words like booking, room, stay, etc.). As a result, it assigns them mathematically similar representations, capturing their semantic relationship without ever being explicitly taught a dictionary definition.
## 3. It All Starts with Randomness
What struck me as deeply counter-intuitive was the starting point for this entire process. To train an AI language model, you begin with complete and utter randomness. The process for an algorithm like Word2Vec starts by assigning a vector of random numbers to every single word in its vocabulary. Initially, these vectors mean nothing. The word "king" is just as different from "queen" as it is from "aardvark."
The structure emerges through a simple, repetitive process. The algorithm iterates through a massive body of text (a "corpus"), looking at a word and its neighbors. For each word, it makes tiny adjustments to those random vectors, nudging them to become slightly better at predicting which words are likely to appear near each other. Over billions of these tiny adjustments, a rich and complex semantic structure emerges from the statistical noise. Professor Manning captures the astonishing nature of this process, noting that "...the miracle of what happens in these deep learning spaces is we do get something useful out." He continues:
...it feels like magic I mean it doesn't really seem like you know we could just start with nothing we could start with random word vectors and a pile of text and say uh do some math and we will get something useful out
This is the power of modern deep learning. Instead of being programmed with human-coded rules about grammar or meaning, the system discovers those intricate relationships on its own, simply by optimizing its ability to predict context in raw text.
## 4. AI Can Draw a Train on a Bridge That Has No Tracks
Today's AI extends beyond text to models that connect words and images. But these systems still learn from statistical patterns, not from a grounded understanding of the physical world. This leads to outputs that are often plausible but factually impossible.
Manning illustrates this with an image generated by DALL-E 2 from the prompt, "a train going over the Golden Gate Bridge." The AI successfully produces a beautiful image of exactly that. There's just one problem. As he humorously clarifies for anyone not from the Bay Area: "the important thing to know is no trains go over the Golden Gate Bridge."
The model makes this error because it doesn't "know" what a bridge or a train is in the real world. It has simply seen countless images of trains on bridges and countless images of the Golden Gate Bridge. It has learned a strong statistical correlation between the concepts of "train" and "bridge," leading it to generate a plausible combination without any grounding in the factual reality of that specific bridge. It can generate what looks right without knowing what is right.
## 5. Language Is Ultimately About People
Perhaps the most important takeaway is that language is far more than a system for exchanging facts. It is a messy, flexible, and deeply social tool used with "imprecision and nuance and emotion." Languages aren't static systems; they are constantly being constructed and reconstructed by their users, with most innovation happening among young people.
This human element is what makes natural language processing so profoundly difficult. To truly understand language is to understand human intention. Professor Manning highlights this with a powerful quote from Stanford psychologist Herb Clark:
The common misconception is that language use has primarily to do with words and what they mean. It doesn't. It has primarily to do with people and what they mean.
This insight from Herb Clark is the key to understanding why models like ChatGPT feel so different. Their breakthrough wasn't just more data; it was in being trained to understand the messy, human-centric intent behind our words, misspellings and all. Their ability to understand a request like "please draft a polite email... my 9-year-old song..." and fulfill the user's underlying goal is a direct result of grappling with the human-centric nature of language. The goal isn't just to process words, but to understand what people mean.

## Conclusion: A New Way of Seeing Language

Peeking behind the curtain of modern AI reveals that its foundations are not just algorithms, but a deep engagement with fundamental questions about humanity. The challenges of building machines that can understand us force us to look more closely at how we understand each other—through context, shared experience, and social intent.
From the realization that language is a tool for thought (Takeaway 1), to deriving meaning from context (Takeaway 2), to discovering that meaning from statistical noise (Takeaway 3), we see a clear thread: AI's progress is tied to emulating how humans create and share knowledge. The final two takeaways remind us of the system's limits (it learns correlation, not truth) and its ultimate purpose (to understand what people mean). As these models get better at wielding our primary tool for thought, it leaves us with a fascinating question: what new kinds of thinking might become possible?

---
---

# Word2Vec: From Symbols to Meaningful Vectors


## Traditional Symbolic Word Representation

Historically, words in computational systems were treated as **discrete symbols**. A common technique used was **[one-hot encoding](https://en.wikipedia.org/wiki/One-hot)**, where:

* Each word corresponds to a unique index in a large vector.
* The vector contains a `1` at the word’s index and `0`s elsewhere.

### Limitations of One-Hot Vectors

* **No similarity**: Vectors for similar words like *“motel”* and *“hotel”* are **orthogonal** (dot product = 0), implying no relationship.
* **Manual enhancement needed**: Methods like **query expansion** attempted to add similarity manually.
* **Pre-neural resources**: Tools like [WordNet](https://wordnet.princeton.edu/) defined synonym relationships and taxonomies, but they:

  * Were often **incomplete**
  * Missed **nuances**
  * Lacked coverage of **modern slang**

---

## Distributional Semantics: The Core Idea

The foundation of Word2Vec is **[distributional semantics](https://en.wikipedia.org/wiki/Distributional_semantics)**, which suggests:

> *"You shall know a word by the company it keeps."*

### Key Insight

* The **meaning of a word** can be inferred from its **surrounding context**.
* For example, analyzing contexts around the word *“banking”* helps define its meaning by associating it with frequently co-occurring terms.

---

## Word2Vec: Learning Word Embeddings

Introduced in **2013**, [Word2Vec](https://en.wikipedia.org/wiki/Word2vec) is a simple, efficient algorithm for learning **word vector representations** (or **word embeddings**) from a large **text corpus**.

### Key Characteristics

* Generates **dense**, **low-dimensional** vectors.
* Similar words get **similar vector representations**.

---

## How Word2Vec Works

1. **Iterate through text**:

   * Treat each word as a **centre word (c)**.
   * Define a **window size** to capture surrounding **context words (o)**.

2. **Predict the context**:

   * Model attempts to **predict context words** given the centre word.
   * Uses co-occurrence statistics from the corpus.

3. **Maximise likelihood**:

   * Adjusts vector parameters to **increase probability** of correct context words.
   * Learns from actual data occurrences.

---

## The Objective Function

The goal is to **maximize the likelihood** of the observed context words across the corpus.

* Mathematically, this is:

  * The **product of all context word probabilities** given centre words.
  * Converted to a **minimization** problem using **negative log-likelihood**.

> 📌 Using the log simplifies the computation by turning a product into a sum.

---

## Softmax for Probability Calculation

Word2Vec uses the **[softmax function](https://en.wikipedia.org/wiki/Softmax_function)** to compute the probability of a context word $o$ given a centre word $c$.

### Dual Vector System

* Each word has two vectors:

  * $\mathbf{v}_c$: For when the word is the **centre word**
  * $\mathbf{u}_o$: For when the word is an **outside/context word**

These vectors are trained to **maximize co-occurrence likelihood** under the softmax framework.

---

## References

* [One-Hot Encoding – Wikipedia](https://en.wikipedia.org/wiki/One-hot)
* [WordNet – Princeton University](https://wordnet.princeton.edu/)
* [Distributional Semantics – Wikipedia](https://en.wikipedia.org/wiki/Distributional_semantics)
* [Word2Vec – Wikipedia](https://en.wikipedia.org/wiki/Word2vec)
* [Softmax Function – Wikipedia](https://en.wikipedia.org/wiki/Softmax_function)

---
---

# Human Language and Its Meaning

## The Significance of Human Language

Language distinguishes humans from even our closest animal relatives, such as [chimpanzees](https://en.wikipedia.org/wiki/Chimpanzee). While chimps can:

* Use tools
* Plan ahead
* Possess strong short-term memory

It is **language** that has enabled humans to dominate the planet.

### Two Crucial Roles of Language

1. **Communication**
   Facilitates cooperation, essential to human social and technological success.

2. **Higher-Level Thought**
   Serves as scaffolding for detailed reasoning, planning, and problem-solving.

### The Power of Writing

The invention of **[writing](https://en.wikipedia.org/wiki/Writing)** (about 5,000 years ago) transformed human society by allowing:

* Preservation of knowledge across time and space
* Acceleration of technological and social progress

### Language as a Social Tool

Language is:

* Not just factual or literal
* Flexible and emotional
* Constantly evolving, especially among younger generations

As psychologist [Herbert Clark](https://profiles.stanford.edu/herbert-clark) noted:

> “Language use has primarily to do with people and what they mean.”

---

## Computational Approaches to Meaning

### Denotational Semantics and WordNet

One classical approach is **[denotational semantics](https://en.wikipedia.org/wiki/Denotational_semantics)**, where:

* Words (signifiers) map to real-world things or ideas (signifieds)
* Example: The word *“tree”* denotes all actual trees

This approach is intuitive for **programming languages**, but problematic for human language.

#### WordNet: A Pre-Neural Resource

[WordNet](https://wordnet.princeton.edu/) was an early computational resource for meaning, defining:

* **Synonyms** (e.g., ‘proficient’ ≈ ‘good’)
* **Hierarchies** (e.g., panda → carnivore)

However, WordNet:

* **Misses nuance** (e.g., “a proficient shot” sounds unnatural)
* **Lacks modern slang**
* Requires **manual curation**

---

## Distributional Semantics: Meaning from Context

This modern approach is grounded in [distributional semantics](https://en.wikipedia.org/wiki/Distributional_semantics), based on J.R. Firth’s principle:

> “You shall know a word by the company it keeps.”

### How It Works

* Analyze contexts in which words appear
* Learn meaning based on co-occurring words

#### Example

The meaning of *“banking”* is inferred by its frequent co-occurrence with terms like:

* *“crisis”*
* *“government”*
* *“debt”*

This leads to **word embeddings**—dense vector representations where:

* Similar words have similar vectors
* Contrast with **[one-hot vectors](https://en.wikipedia.org/wiki/One-hot)**, which imply no similarity

---

## Handling Polysemy (Multiple Meanings)

One limitation of early word vector models is **polysemy**—a single word with multiple meanings.

### Single-Vector Representation

Each word is mapped to **one vector**, which becomes:

* An **average** of all meanings
* Example: *“Star”* might align with both:

  * Astronomical terms (e.g., *“nebula”*)
  * Fame-related terms (e.g., *“celebrity”*)

### Towards Contextual Embeddings

Later models (like [BERT](https://en.wikipedia.org/wiki/BERT_%28language_model%29)) address this by learning **context-specific representations**.

---

## Modern Multimodal Meaning

Recent **[foundation models](https://en.wikipedia.org/wiki/Foundation_model)** are **multimodal**, operating across:

* Text
* Images
* Other data types

### Capabilities

These models can:

* Answer questions about images
* Link language with visual understanding

> Example: Answering “What is unusual about this image?” by combining visual input and linguistic context.

---

## References

* [Chimpanzee – Wikipedia](https://en.wikipedia.org/wiki/Chimpanzee)
* [Writing – Wikipedia](https://en.wikipedia.org/wiki/Writing)
* [Herbert Clark – Stanford](https://profiles.stanford.edu/herbert-clark)
* [Denotational Semantics – Wikipedia](https://en.wikipedia.org/wiki/Denotational_semantics)
* [WordNet – Princeton University](https://wordnet.princeton.edu/)
* [Distributional Semantics – Wikipedia](https://en.wikipedia.org/wiki/Distributional_semantics)
* [One-Hot – Wikipedia](https://en.wikipedia.org/wiki/One-hot)
* [BERT – Wikipedia](https://en.wikipedia.org/wiki/BERT_%28language_model%29)
* [Foundation Model – Wikipedia](https://en.wikipedia.org/wiki/Foundation_model)

---
---

# Generative AI Models in NLP and Their Evolution into Multimodal Systems

## The Breakthrough in Text Generation

For many years, **[Natural Language Processing (NLP)](https://en.wikipedia.org/wiki/Natural_language_processing)** systems struggled with **text generation**. While there was progress in understanding text, generated output often lacked fluency and coherence.

This changed dramatically around **2019** with the emergence of **[Large Language Models (LLMs)](https://en.wikipedia.org/wiki/Large_language_model)** like [GPT-2](https://en.wikipedia.org/wiki/GPT-2), which could generate fluent, coherent, and contextually plausible text.

### How LLMs Generate Text

The core mechanism is conceptually simple:

* The model generates **one word at a time**, conditioning on all the preceding text.
* It predicts the most probable **next word**, then repeats this process.

#### Example

**Prompt**:
"A train carriage containing controlled nuclear materials was stolen in Cincinnati today..."

**Generated Text**:
"The incident occurred on the downtown train line... the US Department of Energy said it is working with the Federal Railroad Administration to find the thief..."

This example demonstrates:

* Grammatical fluency
* Basic reasoning
* Integration of **world knowledge** (e.g., regulatory roles, geographical context)

---

## From Generation to Interaction

Later models, such as **[ChatGPT](https://en.wikipedia.org/wiki/ChatGPT)** and **GPT-4**, improved dramatically in **following user instructions** and **engaging in dialogue**.

### Example Use Case

**User Input**:
"Please draft a polite email to my boss Jeremy that I would not be able to come into the office for the next two days because my 9-year-old song Peter is angry with me..."

**Model Response**:
Generates a **polite and coherent email**, correctly interpreting and correcting the typo from *"song"* to *"son"*.

This shows increased sophistication in:

* Error correction
* Contextual understanding
* Task-oriented generation

---

## The Rise of Multimodal Foundation Models

LLMs have evolved into **[Foundation Models](https://en.wikipedia.org/wiki/Foundation_model)** that can handle multiple **modalities**—types of data beyond text.

These include:

* Images
* Audio
* Video
* Biological sequences (e.g., DNA, RNA)

### Key Multimodal Capabilities

#### 1. **Text-to-Image Generation**

Models like [DALL·E](https://en.wikipedia.org/wiki/DALL-E) can create images from natural language prompts.

* **Example Prompt**:
  "A train going over the Golden Gate Bridge"

* **Capabilities**:

  * Iterative refinement (e.g., "with the bay in the background")
  * Style variation (e.g., "detailed pencil drawing")

> 🛈 Note: These models can generate **realistic but fictional scenes**, such as trains crossing the Golden Gate Bridge (which doesn't happen in reality).

#### 2. **Image-to-Text Analysis**

Multimodal models can **interpret images** and respond to language-based queries about them.

* **Example**:
  Show an image and ask, *"What is unusual about this image?"*

* **Implication**:
  This demonstrates an integrated ability to **link visual and linguistic information**.

---

## References

* [Natural Language Processing – Wikipedia](https://en.wikipedia.org/wiki/Natural_language_processing)
* [Large Language Models – Wikipedia](https://en.wikipedia.org/wiki/Large_language_model)
* [GPT-2 – Wikipedia](https://en.wikipedia.org/wiki/GPT-2)
* [ChatGPT – Wikipedia](https://en.wikipedia.org/wiki/ChatGPT)
* [Foundation Model – Wikipedia](https://en.wikipedia.org/wiki/Foundation_model)
* [DALL·E – Wikipedia](https://en.wikipedia.org/wiki/DALL-E)

