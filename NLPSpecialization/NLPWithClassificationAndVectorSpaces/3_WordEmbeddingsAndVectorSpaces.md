# Word Embeddings and Vector Spaces - Complete Learning Guide

## Table of Contents
1. [Core Concepts](#core-concepts)
2. [Building Vector Representations](#building-vector-representations)
3. [Similarity Metrics](#similarity-metrics)
4. [Vector Arithmetic & Analogies](#vector-arithmetic--analogies)
5. [Dimensionality Reduction with PCA](#dimensionality-reduction-with-pca)
6. [Complete Implementation](#complete-implementation)
7. [Practice Questions & Answers](#practice-questions--answers)
8. [External Resources](#external-resources)



## Core Concepts

### What are Vector Space Models?

Vector Space Models (VSMs) represent words as vectors in a multi-dimensional space where:
- **Similar words** are close together
- **Different words** are far apart
- **Relationships** between words are captured as vector operations

### John Firth's Principle (1957)
> "You shall know a word by the company it keeps"

This fundamental NLP concept means that words appearing in similar contexts tend to have similar meanings.

### Why Vector Representations?

1. **Capture Semantic Meaning**: Words with similar meanings have similar vectors
2. **Enable Mathematical Operations**: Can compute similarity, perform analogies
3. **Dense Representations**: More efficient than one-hot encoding
4. **Transfer Learning**: Pre-trained vectors can be used across tasks

### Applications
- **Information Retrieval**: Search engines, document ranking
- **Machine Translation**: Cross-lingual word mappings
- **Question Answering**: Finding semantically similar questions
- **Text Classification**: Document categorization
- **Chatbots**: Understanding user intent



## Building Vector Representations

### 1. Word-by-Word Co-occurrence Matrix

Build vectors based on how often words appear together within a window of size `k`.

**Example**: With window size k=2, for corpus:
- "I like simple data"
- "I prefer simple raw data"

Co-occurrence matrix for word "data":
```
         simple  raw  like  I  prefer
data       2     1    1    0    1
```

### 2. Word-by-Document Matrix

Count how often words appear in different document categories.

**Example**: Word frequencies across document types:
```
              Entertainment  Economy  Machine Learning
data              500         6620        9320
film              7000        4000        1000
algorithm         100         200         8500
```

### Key Difference
- **Word-by-Word**: Captures local context (nearby words)
- **Word-by-Document**: Captures global context (document topics)



## Similarity Metrics

### 1. Euclidean Distance

**Formula**:
```
d(v, w) = √Σ(vᵢ - wᵢ)²
```

**Properties**:
- Measures absolute distance between points
- Sensitive to vector magnitude
- Range: [0, ∞)
- Smaller distance = more similar

**When to use**:
- Vectors have similar magnitudes
- Absolute position matters

### 2. Cosine Similarity

**Formula**:
```
cos(θ) = (v · w) / (||v|| × ||w||)
```

**Properties**:
- Measures angle between vectors
- Invariant to vector magnitude
- Range: [-1, 1] (in NLP often [0, 1])
- Higher cosine = more similar

**When to use**:
- Comparing documents of different lengths
- Direction matters more than magnitude

### Comparison Example

Consider three document vectors about topics:
- Food: (100, 50)
- Agriculture: (1000, 500)  
- History: (1200, 100)

**Euclidean Distance**:
- d(Food, Agriculture) = large (misleading due to size difference)
- d(Agriculture, History) = smaller

**Cosine Similarity**:
- cos(Food, Agriculture) = high (correctly shows topical similarity)
- cos(Agriculture, History) = lower



## Vector Arithmetic & Analogies

### Word Analogies using Vector Math

Classic example: **"King - Man + Woman = Queen"**

The relationship is captured as:
```
vec(Queen) ≈ vec(King) - vec(Man) + vec(Woman)
```

### How It Works

1. **Find relationship vector**: `relation = vec(Washington) - vec(USA)`
2. **Apply to new word**: `vec(Russia) + relation`
3. **Find nearest neighbor**: The closest vector is likely "Moscow"

### Applications
- Finding capitals of countries
- Gender relationships (actor → actress)
- Tense relationships (walk → walked)
- Comparative relationships (good → better → best)

---

## Dimensionality Reduction with PCA

### Why PCA?

Word vectors often have hundreds of dimensions. PCA helps:
- **Visualize** high-dimensional data in 2D/3D
- **Reduce noise** by keeping principal components
- **Speed up** downstream computations
- **Discover patterns** in word relationships

### PCA Algorithm Steps

1. **Mean-center the data**: Subtract mean from each feature
2. **Compute covariance matrix**: Capture variance relationships
3. **Calculate eigenvectors/eigenvalues**: Find principal components
4. **Sort by eigenvalues**: Order by explained variance
5. **Project data**: Transform to lower dimensions

### Key Concepts

- **Eigenvectors**: Directions of maximum variance (principal components)
- **Eigenvalues**: Amount of variance explained by each component
- **Explained variance ratio**: Percentage of information retained



## Complete Implementation

### Comprehensive Python Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import re
from scipy.spatial.distance import cosine, euclidean
from sklearn.decomposition import PCA

class VectorSpaceModel:
    """
    Complete implementation of Vector Space Models for NLP
    """
    
    def __init__(self, window_size=2):
        self.window_size = window_size
        self.vocabulary = set()
        self.word_to_index = {}
        self.index_to_word = {}
        self.cooccurrence_matrix = None
        self.word_vectors = {}
        
    def preprocess_text(self, text):
        """
        Basic text preprocessing
        """
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation and split
        words = re.findall(r'\b\w+\b', text)
        return words
    
    def build_vocabulary(self, corpus):
        """
        Build vocabulary from corpus
        
        Args:
            corpus: List of documents (strings)
        """
        all_words = []
        for doc in corpus:
            words = self.preprocess_text(doc)
            all_words.extend(words)
        
        # Get unique words
        self.vocabulary = set(all_words)
        
        # Create mappings
        self.word_to_index = {word: i for i, word in enumerate(sorted(self.vocabulary))}
        self.index_to_word = {i: word for word, i in self.word_to_index.items()}
        
        print(f"Vocabulary size: {len(self.vocabulary)}")
        
    def build_cooccurrence_matrix(self, corpus):
        """
        Build word-by-word co-occurrence matrix
        
        Args:
            corpus: List of documents
        """
        n_words = len(self.vocabulary)
        self.cooccurrence_matrix = np.zeros((n_words, n_words))
        
        for doc in corpus:
            words = self.preprocess_text(doc)
            
            # Count co-occurrences within window
            for i, word in enumerate(words):
                if word not in self.word_to_index:
                    continue
                    
                word_idx = self.word_to_index[word]
                
                # Look at context window
                start = max(0, i - self.window_size)
                end = min(len(words), i + self.window_size + 1)
                
                for j in range(start, end):
                    if i != j and words[j] in self.word_to_index:
                        context_idx = self.word_to_index[words[j]]
                        self.cooccurrence_matrix[word_idx][context_idx] += 1
        
        # Convert to word vectors
        for word, idx in self.word_to_index.items():
            self.word_vectors[word] = self.cooccurrence_matrix[idx]
            
        return self.cooccurrence_matrix
    
    def build_word_document_matrix(self, corpus, labels=None):
        """
        Build word-by-document matrix
        
        Args:
            corpus: List of documents
            labels: Optional document labels/categories
        """
        if labels is None:
            labels = list(range(len(corpus)))
        
        unique_labels = list(set(labels))
        n_words = len(self.vocabulary)
        n_docs = len(unique_labels)
        
        # Initialize matrix
        word_doc_matrix = np.zeros((n_words, n_docs))
        
        # Count word frequencies per document category
        for doc, label in zip(corpus, labels):
            words = self.preprocess_text(doc)
            label_idx = unique_labels.index(label)
            
            for word in words:
                if word in self.word_to_index:
                    word_idx = self.word_to_index[word]
                    word_doc_matrix[word_idx][label_idx] += 1
        
        # Store as word vectors
        for word, idx in self.word_to_index.items():
            self.word_vectors[word] = word_doc_matrix[idx]
            
        return word_doc_matrix
    
    def euclidean_distance(self, word1, word2):
        """
        Calculate Euclidean distance between two word vectors
        """
        if word1 not in self.word_vectors or word2 not in self.word_vectors:
            return None
        
        vec1 = self.word_vectors[word1]
        vec2 = self.word_vectors[word2]
        
        return np.linalg.norm(vec1 - vec2)
    
    def cosine_similarity(self, word1, word2):
        """
        Calculate cosine similarity between two word vectors
        """
        if word1 not in self.word_vectors or word2 not in self.word_vectors:
            return None
        
        vec1 = self.word_vectors[word1]
        vec2 = self.word_vectors[word2]
        
        # Handle zero vectors
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0
        
        return np.dot(vec1, vec2) / (norm1 * norm2)
    
    def find_most_similar(self, word, n=5, metric='cosine'):
        """
        Find most similar words to a given word
        
        Args:
            word: Target word
            n: Number of similar words to return
            metric: 'cosine' or 'euclidean'
        """
        if word not in self.word_vectors:
            return []
        
        similarities = []
        
        for other_word in self.word_vectors:
            if other_word == word:
                continue
            
            if metric == 'cosine':
                sim = self.cosine_similarity(word, other_word)
                if sim is not None:
                    similarities.append((other_word, sim))
            else:  # euclidean
                dist = self.euclidean_distance(word, other_word)
                if dist is not None:
                    similarities.append((other_word, -dist))  # Negative for sorting
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:n]
    
    def word_analogy(self, word1, word2, word3):
        """
        Solve word analogy: word1 - word2 + word3 = ?
        
        Example: king - man + woman = queen
        """
        if not all(w in self.word_vectors for w in [word1, word2, word3]):
            return None
        
        # Calculate target vector
        target_vec = (self.word_vectors[word1] - 
                     self.word_vectors[word2] + 
                     self.word_vectors[word3])
        
        # Find closest word to target vector
        best_word = None
        best_similarity = -float('inf')
        
        for word in self.word_vectors:
            if word in [word1, word2, word3]:
                continue
            
            vec = self.word_vectors[word]
            similarity = np.dot(target_vec, vec) / (np.linalg.norm(target_vec) * np.linalg.norm(vec))
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_word = word
        
        return best_word, best_similarity
    
    def reduce_dimensions_pca(self, n_components=2):
        """
        Reduce word vector dimensions using PCA
        
        Args:
            n_components: Number of principal components
            
        Returns:
            Dictionary of reduced vectors
        """
        # Stack all word vectors
        words = list(self.word_vectors.keys())
        X = np.array([self.word_vectors[word] for word in words])
        
        # Apply PCA
        pca = PCA(n_components=n_components)
        X_reduced = pca.fit_transform(X)
        
        # Create reduced vectors dictionary
        reduced_vectors = {word: X_reduced[i] for i, word in enumerate(words)}
        
        # Print explained variance
        print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
        print(f"Total variance explained: {sum(pca.explained_variance_ratio_):.2%}")
        
        return reduced_vectors, pca
    
    def visualize_words(self, words_to_plot=None, reduced_vectors=None):
        """
        Visualize word vectors in 2D
        
        Args:
            words_to_plot: List of words to visualize (None = all)
            reduced_vectors: Pre-computed 2D vectors (None = compute with PCA)
        """
        if reduced_vectors is None:
            reduced_vectors, _ = self.reduce_dimensions_pca(n_components=2)
        
        if words_to_plot is None:
            words_to_plot = list(reduced_vectors.keys())[:50]  # Limit for clarity
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        for word in words_to_plot:
            if word in reduced_vectors:
                vec = reduced_vectors[word]
                plt.scatter(vec[0], vec[1], alpha=0.6)
                plt.annotate(word, (vec[0], vec[1]), fontsize=9, alpha=0.7)
        
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.title('Word Vectors Visualization (PCA)')
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def create_document_vector(self, document):
        """
        Create vector representation for a document
        
        Args:
            document: Text document
            
        Returns:
            Document vector (sum of word vectors)
        """
        words = self.preprocess_text(document)
        
        # Initialize with zeros
        if self.word_vectors:
            vec_size = len(next(iter(self.word_vectors.values())))
            doc_vector = np.zeros(vec_size)
        else:
            return None
        
        # Sum word vectors
        word_count = 0
        for word in words:
            if word in self.word_vectors:
                doc_vector += self.word_vectors[word]
                word_count += 1
        
        # Average (optional - can also just use sum)
        if word_count > 0:
            doc_vector /= word_count
        
        return doc_vector

# Demonstration and usage examples
def demonstrate_vector_spaces():
    """
    Demonstrate vector space operations with examples
    """
    print("="*60)
    print("VECTOR SPACE MODEL DEMONSTRATION")
    print("="*60)
    
    # Sample corpus
    corpus = [
        "The cat sat on the mat",
        "The dog sat on the log",
        "Cats and dogs are pets",
        "The mat was comfortable",
        "I love my pet cat",
        "Dogs are loyal animals",
        "Cats are independent animals",
        "The weather is nice today",
        "Machine learning is fascinating",
        "Natural language processing uses vectors"
    ]
    
    # Initialize model
    vsm = VectorSpaceModel(window_size=2)
    
    # Build vocabulary
    vsm.build_vocabulary(corpus)
    
    # Build co-occurrence matrix
    print("\nBuilding co-occurrence matrix...")
    vsm.build_cooccurrence_matrix(corpus)
    
    # Find similar words
    print("\n" + "="*60)
    print("WORD SIMILARITY EXAMPLES")
    print("="*60)
    
    test_words = ['cat', 'dog', 'mat']
    
    for word in test_words:
        print(f"\nMost similar to '{word}':")
        similar = vsm.find_most_similar(word, n=3)
        for similar_word, score in similar:
            print(f"  {similar_word:15} similarity: {score:.3f}")
    
    # Calculate specific similarities
    print("\n" + "="*60)
    print("SIMILARITY METRICS COMPARISON")
    print("="*60)
    
    word_pairs = [('cat', 'dog'), ('cat', 'mat'), ('cat', 'learning')]
    
    for w1, w2 in word_pairs:
        cos_sim = vsm.cosine_similarity(w1, w2)
        euc_dist = vsm.euclidean_distance(w1, w2)
        
        if cos_sim is not None:
            print(f"\n'{w1}' vs '{w2}':")
            print(f"  Cosine similarity:  {cos_sim:.3f}")
            print(f"  Euclidean distance: {euc_dist:.3f}")
    
    # PCA and visualization
    print("\n" + "="*60)
    print("DIMENSIONALITY REDUCTION WITH PCA")
    print("="*60)
    
    reduced_vectors, pca = vsm.reduce_dimensions_pca(n_components=2)
    
    # Document vectors
    print("\n" + "="*60)
    print("DOCUMENT VECTOR EXAMPLES")
    print("="*60)
    
    test_docs = [
        "cats and dogs",
        "machine learning",
        "the cat sat"
    ]
    
    for doc in test_docs:
        vec = vsm.create_document_vector(doc)
        if vec is not None:
            print(f"\nDocument: '{doc}'")
            print(f"Vector shape: {vec.shape}")
            print(f"Vector norm: {np.linalg.norm(vec):.3f}")
    
    return vsm

# Advanced PCA implementation from scratch
def compute_pca_from_scratch(X, n_components=2):
    """
    PCA implementation from scratch for educational purposes
    
    Args:
        X: Data matrix (m samples × n features)
        n_components: Number of components to keep
        
    Returns:
        X_reduced: Transformed data
        explained_variance: Variance explained by each component
    """
    # Step 1: Mean-center the data
    X_mean = np.mean(X, axis=0)
    X_centered = X - X_mean
    
    # Step 2: Compute covariance matrix
    n_samples = X.shape[0]
    cov_matrix = np.dot(X_centered.T, X_centered) / (n_samples - 1)
    
    # Step 3: Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Step 4: Sort by eigenvalues (descending)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Step 5: Select top n_components
    top_eigenvectors = eigenvectors[:, :n_components]
    top_eigenvalues = eigenvalues[:n_components]
    
    # Step 6: Transform data
    X_reduced = np.dot(X_centered, top_eigenvectors)
    
    # Calculate explained variance
    total_variance = np.sum(eigenvalues)
    explained_variance = top_eigenvalues / total_variance
    
    print(f"Shape before PCA: {X.shape}")
    print(f"Shape after PCA: {X_reduced.shape}")
    print(f"Explained variance per component: {explained_variance}")
    print(f"Total variance explained: {np.sum(explained_variance):.2%}")
    
    return X_reduced, explained_variance

# Run demonstrations
if __name__ == "__main__":
    # Main demonstration
    vsm = demonstrate_vector_spaces()
    
    # PCA from scratch example
    print("\n" + "="*60)
    print("PCA FROM SCRATCH EXAMPLE")
    print("="*60)
    
    # Create sample data
    np.random.seed(42)
    sample_data = np.random.randn(100, 50)  # 100 samples, 50 features
    
    # Apply PCA
    reduced_data, variance = compute_pca_from_scratch(sample_data, n_components=3)
```



## Practice Questions & Answers

### Conceptual Questions

**Q1: What is the fundamental principle behind word embeddings?**

**A1:** The distributional hypothesis, stated by John Firth: "You shall know a word by the company it keeps." Words that appear in similar contexts tend to have similar meanings. This allows us to represent words as vectors based on their co-occurrence patterns with other words.

**Q2: What's the difference between word-by-word and word-by-document matrices?**

**A2:**
- **Word-by-word**: Captures local context by counting co-occurrences within a small window (e.g., 2-5 words). Better for syntactic relationships.
- **Word-by-document**: Captures global context by counting word frequencies across document categories. Better for topical/semantic relationships.

**Q3: When should you use cosine similarity vs Euclidean distance?**

**A3:**
- **Cosine similarity**: When comparing documents of different lengths or when direction matters more than magnitude (e.g., document similarity)
- **Euclidean distance**: When absolute position matters and vectors have similar scales (e.g., clustering in normalized spaces)

### Mathematical Questions

**Q4: Calculate the cosine similarity between vectors v=[3,4] and w=[4,3].**

**A4:**
```
cos(θ) = (v·w) / (||v|| × ||w||)
       = (3×4 + 4×3) / (√(9+16) × √(16+9))
       = (12 + 12) / (5 × 5)
       = 24 / 25
       = 0.96

High similarity (close to 1)
```

**Q5: Given word vectors: king=[2,5], man=[1,2], woman=[1,3], calculate the analogy "king - man + woman".**

**A5:**
```
result = king - man + woman
       = [2,5] - [1,2] + [1,3]
       = [2-1+1, 5-2+3]
       = [2, 6]

This would be the predicted vector for "queen"
```

**Q6: If a cooccurrence matrix has 1000 unique words, what's the matrix size and why might this be problematic?**

**A6:**
- Matrix size: 1000 × 1000 = 1,000,000 entries
- Problems:
  - **Memory**: Requires significant storage
  - **Sparsity**: Most entries are zero (words don't co-occur)
  - **Computation**: Operations on large matrices are expensive
- Solutions: Use sparse matrix representations, dimensionality reduction, or modern embeddings (Word2Vec, GloVe)

### Implementation Questions

**Q7: How do you handle out-of-vocabulary (OOV) words in vector space models?**

**A7:** Several approaches:
1. **Ignore**: Skip OOV words (loses information)
2. **UNK token**: Map all OOV words to a special vector
3. **Subword embeddings**: Use character n-grams (FastText approach)
4. **Zero vector**: Assign zero vector (neutral contribution)
5. **Random initialization**: Assign random vector
6. **Contextual embeddings**: Use models like BERT that generate vectors dynamically

**Q8: Why is PCA useful for word embeddings?**

**A8:** PCA serves multiple purposes:
1. **Visualization**: Reduce to 2D/3D for plotting
2. **Noise reduction**: Keep only principal components
3. **Computational efficiency**: Fewer dimensions = faster operations
4. **Pattern discovery**: Reveals major axes of variation
5. **Storage**: Compressed representations

**Q9: How do you create a document vector from word vectors?**

**A9:** Common approaches:
1. **Sum/Average**: Add or average all word vectors
2. **Weighted average**: Use TF-IDF or other weights
3. **Max pooling**: Take maximum value per dimension
4. **Doc2Vec**: Learn document embeddings directly
5. **Transformers**: Use [CLS] token from BERT-like models

### Debugging Questions

**Q10: Your word similarities seem random. What could be wrong?**

**A10:** Common issues:
1. **Window size too small**: Missing important context
2. **Corpus too small**: Insufficient co-occurrence data
3. **No preprocessing**: Punctuation, case sensitivity issues
4. **Sparse matrices**: Most words have zero co-occurrences
5. **Normalization**: Vectors not normalized for cosine similarity
6. **Bug in indexing**: Word-to-index mapping incorrect

**Q11: PCA reduces your 300D vectors to 2D but loses 95% variance. Is this acceptable?**

**A11:** It depends on the use case:
- **For visualization**: Acceptable - main goal is to see rough clustering
- **For downstream tasks**: Not acceptable - too much information lost
- **Alternatives**:
  - Use more components (10-50)
  - Try t-SNE for better 2D visualization
  - Use the original high-dimensional vectors for actual tasks
  - Consider other dimensionality reduction methods (LDA, UMAP)


## External Resources

### Foundational Papers
1. [Mikolov et al. (2013) - Word2Vec Paper](https://arxiv.org/abs/1301.3781) - Efficient Estimation of Word Representations
2. [Pennington et al. (2014) - GloVe](https://nlp.stanford.edu/projects/glove/) - Global Vectors for Word Representation
3. [Bojanowski et al. (2017) - FastText](https://arxiv.org/abs/1607.04606) - Enriching Word Vectors with Subword Information
4. https://aman.ai/coursera-nlp/vector-spaces/

### Books & Courses
1. [Speech and Language Processing - Jurafsky & Martin](https://web.stanford.edu/~jurafsky/slp3/) - Chapter 6 on Vector Semantics
2. [Stanford CS224N - NLP with Deep Learning](http://web.stanford.edu/class/cs224n/) - Comprehensive NLP course
3. [Natural Language Processing - Eisenstein](https://github.com/jacobeisenstein/gt-nlp-class/blob/master/notes/eisenstein-nlp-notes.pdf)

### Video Lectures
1. [Stanford CS224N Lecture 1 - Word Vectors](https://www.youtube.com/watch?v=8rXD5-xhemo)
2. [Word Embedding Explained - Luis Serrano](https://www.youtube.com/watch?v=viZrOnJclY0)
3. [Illustrated Word2Vec - Jay Alammar](http://jalammar.github.io/illustrated-word2vec/)

### Interactive Tutorials
1. [Gensim Word2Vec Tutorial](https://radimrehurek.com/gensim/models/word2vec.html)
2. [TensorFlow Embeddings Guide](https://www.tensorflow.org/text/guide/word_embeddings)
3. [Word Embedding Visualization](https://projector.tensorflow.org/) - TensorFlow Embedding Projector

### Implementation Libraries
1. [Gensim](https://radimrehurek.com/gensim/) - Topic modeling and word embeddings
2. [spaCy](https://spacy.io/usage/vectors-similarity) - Industrial-strength NLP with vectors
3. [Hugging Face Transformers](https://huggingface.co/docs/transformers/) - State-of-the-art embeddings

### Advanced Topics
1. [BERT Paper](https://arxiv.org/abs/1810.04805) - Contextualized word embeddings
2. [ELMo Paper](https://arxiv.org/abs/1802.05365) - Deep contextualized representations
3. [Sentence-BERT](https://www.sbert.net/) - Sentence embeddings

### Dimensionality Reduction
1. [PCA Explained Visually](https://setosa.io/ev/principal-component-analysis/)
2. [t-SNE Paper](https://lvdmaaten.github.io/tsne/) - Laurens van der Maaten
3. [UMAP](https://umap-learn.readthedocs.io/) - Modern dimensionality reduction

### Practice Resources
1. [Google's Word2Vec Pretrained Vectors](https://code.google.com/archive/p/word2vec/)
2. [GloVe Pretrained Vectors](https://nlp.stanford.edu/projects/glove/)
3. [FastText Pretrained Vectors](https://fasttext.cc/docs/en/crawl-vectors.html)

### Research & Applications
1. [Papers with Code - Word Embeddings](https://paperswithcode.com/task/word-embeddings)
2. [ACL Anthology](https://aclanthology.org/) - NLP research papers
3. [Semantic Scholar](https://www.semanticscholar.org/) - AI-powered research tool


## Summary

### Key Takeaways

1. **Vector Space Models** represent words as points in multi-dimensional space where distances encode semantic relationships

2. **Two Main Approaches**:
   - Co-occurrence matrices (word-by-word, word-by-document)
   - Learned embeddings (Word2Vec, GloVe, FastText)

3. **Similarity Metrics**:
   - Euclidean distance: Absolute distance, sensitive to magnitude
   - Cosine similarity: Angular similarity, magnitude-invariant

4. **Vector Arithmetic** enables solving analogies and finding relationships between words

5. **PCA** reduces dimensions while preserving maximum variance, essential for visualization

### Evolution of Word Embeddings

```
One-hot Encoding (sparse, no semantics)
           ↓
Co-occurrence Matrices (captures context)
           ↓
SVD/PCA (dimensionality reduction)
           ↓
Word2Vec/GloVe (efficient learning)
           ↓
FastText (subword information)
           ↓
Contextual Embeddings (BERT, GPT)
```

### When to Use What

| Method | Use Case | Pros | Cons |
|--------|----------|------|------|
| Co-occurrence | Small corpus, interpretability | Simple, intuitive | Sparse, high-dimensional |
| Word2Vec | Medium corpus, efficiency | Fast, good quality | Requires tuning |
| GloVe | Large corpus, performance | Captures global statistics | Memory intensive |
| FastText | OOV handling, morphology | Handles unseen words | Larger model size |
| BERT/GPT | Context-dependent meaning | State-of-the-art | Computationally expensive |

### Next Steps
1. Implement word embeddings from scratch to understand the mechanics
2. Experiment with pre-trained embeddings (Word2Vec, GloVe)
3. Learn about contextual embeddings (BERT, GPT)
4. Apply embeddings to downstream tasks (classification, similarity)
5. Explore multilingual and domain-specific embeddings
