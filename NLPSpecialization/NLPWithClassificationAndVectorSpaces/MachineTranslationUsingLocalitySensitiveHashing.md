# Machine Translation with Locality Sensitive Hashing - Complete Learning Guide

## Table of Contents
1. [Core Concepts](#core-concepts)
2. [Mathematical Foundations](#mathematical-foundations)
3. [Translation Matrix Learning](#translation-matrix-learning)
4. [K-Nearest Neighbors Search](#k-nearest-neighbors-search)
5. [Locality Sensitive Hashing](#locality-sensitive-hashing)
6. [Complete Implementation](#complete-implementation)
7. [Practice Questions & Answers](#practice-questions--answers)
8. [External Resources](#external-resources)



## Core Concepts

### What is Machine Translation?

Machine Translation (MT) is the task of automatically converting text from one language to another while preserving meaning. Modern approaches use word embeddings and vector transformations.

### Key Components

1. **Word Embeddings**: Vector representations of words in each language
2. **Alignment Matrix**: Linear transformation aligning vector spaces
3. **Nearest Neighbor Search**: Finding closest translations
4. **Optimization**: Learning the best transformation matrix

### The Translation Pipeline

```
English Word → English Embedding → Transform (R matrix) → French Space → Find Nearest → French Word
```

### Why Locality Sensitive Hashing?

- **Problem**: Finding nearest neighbors in high-dimensional space is O(n×d) - very slow!
- **Solution**: LSH reduces search space by hashing similar vectors to same buckets
- **Trade-off**: Approximate results but much faster O(1) average lookup



## Mathematical Foundations

### Frobenius Norm

The Frobenius norm measures the magnitude of a matrix, generalizing vector norm to matrices.

**Definition**:
```
||A||_F = √(Σᵢⱼ |aᵢⱼ|²)
```

**Example**:
For matrix A = [[2, 2], [2, 2]]:
```
||A||_F = √(2² + 2² + 2² + 2²) = √16 = 4
```

**Why Important**: Used to measure the difference between predicted and actual translations.

### Dot Product & Matrix Multiplication

**Dot Product**: Measures similarity between vectors
```
a · b = Σᵢ aᵢbᵢ
```

**Matrix Multiplication**: Transforms entire sets of vectors
```
XR = Y (approximately)
where X = English vectors, R = transformation, Y = French vectors
```



## Translation Matrix Learning

### The Optimization Problem

**Goal**: Find matrix R such that XR ≈ Y

**Loss Function**:
```
Loss = ||XR - Y||²_F / m
```
where m is the number of training word pairs

### Gradient Descent Algorithm

1. **Initialize**: R randomly or with identity matrix
2. **Compute Gradient**:
   ```
   g = (2/m) × X^T(XR - Y)
   ```
3. **Update**:
   ```
   R = R - α × g
   ```
4. **Repeat** until convergence

### Why This Works

- Word embeddings capture semantic relationships
- Similar concepts have similar vector representations across languages
- Linear transformation can align these spaces


## K-Nearest Neighbors Search

### The Challenge

After transformation XR, the result may not exactly match any French word vector. We need to find the K closest French vectors.

### Naive Approach

```python
def naive_knn(query, vectors, k):
    distances = [distance(query, v) for v in vectors]
    return sorted(range(len(distances)), key=lambda i: distances[i])[:k]
```

**Problem**: O(n×d) complexity - too slow for large vocabularies!

### Distance Metrics

- **Euclidean**: √Σ(aᵢ - bᵢ)²
- **Cosine**: 1 - (a·b)/(||a||×||b||)
- **Manhattan**: Σ|aᵢ - bᵢ|



## Locality Sensitive Hashing

### Core Idea

Hash similar items to the same bucket with high probability, dissimilar items to different buckets.

### How LSH Works

1. **Random Hyperplanes**: Generate random normal vectors
2. **Hash Function**: For each hyperplane, compute sign of dot product
3. **Hash Value**: Concatenate binary values
4. **Multiple Hash Tables**: Use different hyperplane sets for better coverage

### LSH Algorithm

```
1. Generate k random hyperplanes (normal vectors)
2. For each vector v:
   - For each hyperplane i with normal nᵢ:
     - If v · nᵢ ≥ 0: hᵢ = 1
     - Else: hᵢ = 0
   - Hash value = concatenate all hᵢ values
3. Store vector in bucket[hash_value]
```

### Multiple Hash Tables

Using L different sets of hyperplanes increases recall:
- Probability of finding true neighbor increases
- Trade-off: More memory and computation


## Complete Implementation

### Comprehensive Python Implementation

```python
import numpy as np
from collections import defaultdict
import heapq
from typing import List, Tuple, Dict

class MachineTranslator:
    """
    Machine Translation using word embeddings and LSH
    """
    
    def __init__(self, embedding_dim=300):
        self.embedding_dim = embedding_dim
        self.R = None  # Translation matrix
        self.source_embeddings = {}  # English embeddings
        self.target_embeddings = {}  # French embeddings
        self.target_words = []  # List of target language words
        self.target_matrix = None  # Matrix of target embeddings
        
    def load_embeddings(self, source_file, target_file, max_words=5000):
        """
        Load pre-trained word embeddings
        
        Args:
            source_file: Path to source language embeddings
            target_file: Path to target language embeddings
            max_words: Maximum number of words to load
        """
        # Simulated loading (in practice, load from files like GloVe)
        np.random.seed(42)
        
        # Generate sample embeddings for demonstration
        source_words = ['hello', 'world', 'cat', 'dog', 'house', 'car', 
                       'happy', 'sad', 'big', 'small']
        target_words = ['bonjour', 'monde', 'chat', 'chien', 'maison', 'voiture',
                       'heureux', 'triste', 'grand', 'petit']
        
        for i, (sw, tw) in enumerate(zip(source_words, target_words)):
            # Create related but different embeddings
            base = np.random.randn(self.embedding_dim)
            self.source_embeddings[sw] = base + np.random.randn(self.embedding_dim) * 0.1
            self.target_embeddings[tw] = base + np.random.randn(self.embedding_dim) * 0.15
            
        self.target_words = list(self.target_embeddings.keys())
        self.target_matrix = np.array([self.target_embeddings[w] for w in self.target_words])
        
        print(f"Loaded {len(self.source_embeddings)} source embeddings")
        print(f"Loaded {len(self.target_embeddings)} target embeddings")
    
    def create_training_matrices(self, word_pairs):
        """
        Create aligned training matrices from word pairs
        
        Args:
            word_pairs: List of (source_word, target_word) tuples
            
        Returns:
            X: Source language matrix
            Y: Target language matrix
        """
        X = []
        Y = []
        
        for source_word, target_word in word_pairs:
            if source_word in self.source_embeddings and target_word in self.target_embeddings:
                X.append(self.source_embeddings[source_word])
                Y.append(self.target_embeddings[target_word])
        
        return np.array(X), np.array(Y)
    
    def compute_loss(self, X, Y, R):
        """
        Compute the Frobenius norm loss
        
        Args:
            X: Source embeddings matrix
            Y: Target embeddings matrix  
            R: Translation matrix
            
        Returns:
            Loss value
        """
        m = X.shape[0]
        diff = np.dot(X, R) - Y
        loss = np.sum(diff * diff) / m  # Squared Frobenius norm
        return loss
    
    def compute_gradient(self, X, Y, R):
        """
        Compute gradient of loss with respect to R
        
        Args:
            X: Source embeddings matrix
            Y: Target embeddings matrix
            R: Translation matrix
            
        Returns:
            Gradient matrix
        """
        m = X.shape[0]
        gradient = (2.0 / m) * np.dot(X.T, np.dot(X, R) - Y)
        return gradient
    
    def train_translation_matrix(self, word_pairs, learning_rate=0.01, 
                                epochs=1000, verbose=True):
        """
        Learn translation matrix using gradient descent
        
        Args:
            word_pairs: Training pairs of (source, target) words
            learning_rate: Learning rate for gradient descent
            epochs: Number of training iterations
            verbose: Print progress
            
        Returns:
            Training history (losses)
        """
        # Create training matrices
        X, Y = self.create_training_matrices(word_pairs)
        
        if X.shape[0] == 0:
            raise ValueError("No valid word pairs found in embeddings")
        
        # Initialize R randomly
        self.R = np.random.randn(self.embedding_dim, self.embedding_dim) * 0.01
        
        losses = []
        
        for epoch in range(epochs):
            # Compute loss
            loss = self.compute_loss(X, Y, self.R)
            losses.append(loss)
            
            # Compute gradient
            gradient = self.compute_gradient(X, Y, self.R)
            
            # Update R
            self.R -= learning_rate * gradient
            
            # Print progress
            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss = {loss:.4f}")
        
        return losses
    
    def translate_word(self, word, k=5):
        """
        Translate a word using the learned matrix
        
        Args:
            word: Source language word
            k: Number of translation candidates
            
        Returns:
            List of (target_word, distance) tuples
        """
        if word not in self.source_embeddings:
            return []
        
        if self.R is None:
            raise ValueError("Translation matrix not trained yet")
        
        # Transform source embedding
        source_vec = self.source_embeddings[word]
        transformed = np.dot(source_vec, self.R)
        
        # Find k nearest neighbors
        distances = []
        for target_word in self.target_words:
            target_vec = self.target_embeddings[target_word]
            dist = np.linalg.norm(transformed - target_vec)
            distances.append((target_word, dist))
        
        # Sort and return top k
        distances.sort(key=lambda x: x[1])
        return distances[:k]


class LocalitySensitiveHash:
    """
    LSH for approximate nearest neighbor search
    """
    
    def __init__(self, n_hyperplanes=10, n_tables=5):
        """
        Initialize LSH
        
        Args:
            n_hyperplanes: Number of hyperplanes per table
            n_tables: Number of hash tables
        """
        self.n_hyperplanes = n_hyperplanes
        self.n_tables = n_tables
        self.hyperplanes = []
        self.hash_tables = []
        self.vectors = {}  # Store vectors by ID
        
    def fit(self, vectors, vector_ids=None):
        """
        Build LSH index
        
        Args:
            vectors: Array of vectors (n_samples × n_features)
            vector_ids: Optional IDs for vectors
        """
        n_samples, n_features = vectors.shape
        
        if vector_ids is None:
            vector_ids = list(range(n_samples))
        
        # Store vectors
        for i, vec_id in enumerate(vector_ids):
            self.vectors[vec_id] = vectors[i]
        
        # Create hash tables
        for table_idx in range(self.n_tables):
            # Generate random hyperplanes for this table
            hyperplanes = np.random.randn(self.n_hyperplanes, n_features)
            hyperplanes = hyperplanes / np.linalg.norm(hyperplanes, axis=1, keepdims=True)
            self.hyperplanes.append(hyperplanes)
            
            # Create hash table
            hash_table = defaultdict(list)
            
            # Hash all vectors
            for i, vec_id in enumerate(vector_ids):
                hash_val = self._hash_vector(vectors[i], hyperplanes)
                hash_table[hash_val].append(vec_id)
            
            self.hash_tables.append(hash_table)
    
    def _hash_vector(self, vector, hyperplanes):
        """
        Compute hash value for a vector
        
        Args:
            vector: Input vector
            hyperplanes: Set of hyperplane normals
            
        Returns:
            Hash value (string of bits)
        """
        # Compute dot products with all hyperplanes
        dots = np.dot(hyperplanes, vector)
        
        # Convert to binary hash
        hash_bits = (dots >= 0).astype(int)
        
        # Convert to string for use as dictionary key
        return ''.join(map(str, hash_bits))
    
    def query(self, query_vector, k=5):
        """
        Find approximate k nearest neighbors
        
        Args:
            query_vector: Query vector
            k: Number of neighbors
            
        Returns:
            List of (vector_id, distance) tuples
        """
        candidates = set()
        
        # Check all hash tables
        for table_idx in range(self.n_tables):
            hyperplanes = self.hyperplanes[table_idx]
            hash_val = self._hash_vector(query_vector, hyperplanes)
            
            # Get vectors in same bucket
            if hash_val in self.hash_tables[table_idx]:
                bucket_vectors = self.hash_tables[table_idx][hash_val]
                candidates.update(bucket_vectors)
        
        # Compute actual distances to candidates
        distances = []
        for vec_id in candidates:
            vec = self.vectors[vec_id]
            dist = np.linalg.norm(query_vector - vec)
            distances.append((vec_id, dist))
        
        # Sort and return top k
        distances.sort(key=lambda x: x[1])
        return distances[:k]


class EnhancedTranslator:
    """
    Machine translator with LSH for efficient search
    """
    
    def __init__(self, embedding_dim=300):
        self.translator = MachineTranslator(embedding_dim)
        self.lsh = None
        
    def setup(self, source_embeddings, target_embeddings):
        """
        Setup translator with embeddings
        """
        self.translator.source_embeddings = source_embeddings
        self.translator.target_embeddings = target_embeddings
        self.translator.target_words = list(target_embeddings.keys())
        
        # Build LSH index for target embeddings
        target_vectors = np.array([target_embeddings[w] for w in self.translator.target_words])
        self.lsh = LocalitySensitiveHash(n_hyperplanes=10, n_tables=5)
        self.lsh.fit(target_vectors, self.translator.target_words)
        
    def train(self, word_pairs, **kwargs):
        """
        Train translation matrix
        """
        return self.translator.train_translation_matrix(word_pairs, **kwargs)
    
    def translate_fast(self, word, k=5):
        """
        Fast translation using LSH
        
        Args:
            word: Source word to translate
            k: Number of candidates
            
        Returns:
            Translation candidates with distances
        """
        if word not in self.translator.source_embeddings:
            return []
        
        if self.translator.R is None:
            raise ValueError("Translation matrix not trained")
        
        # Transform source embedding
        source_vec = self.translator.source_embeddings[word]
        transformed = np.dot(source_vec, self.translator.R)
        
        # Use LSH for fast search
        candidates = self.lsh.query(transformed, k=k*2)  # Get more candidates
        
        # Refine with actual distances
        results = []
        for word_id, _ in candidates[:k]:
            target_vec = self.translator.target_embeddings[word_id]
            dist = np.linalg.norm(transformed - target_vec)
            results.append((word_id, dist))
        
        results.sort(key=lambda x: x[1])
        return results[:k]


# Demonstration functions
def demonstrate_translation():
    """
    Demonstrate basic machine translation
    """
    print("="*60)
    print("MACHINE TRANSLATION DEMONSTRATION")
    print("="*60)
    
    # Create translator
    translator = MachineTranslator(embedding_dim=50)  # Smaller for demo
    
    # Load embeddings
    translator.load_embeddings(None, None)
    
    # Training pairs (aligned words)
    word_pairs = [
        ('hello', 'bonjour'),
        ('world', 'monde'),
        ('cat', 'chat'),
        ('dog', 'chien'),
        ('house', 'maison'),
        ('happy', 'heureux')
    ]
    
    print("\nTraining translation matrix...")
    losses = translator.train_translation_matrix(
        word_pairs, 
        learning_rate=0.01,
        epochs=500,
        verbose=False
    )
    
    print(f"Initial loss: {losses[0]:.4f}")
    print(f"Final loss: {losses[-1]:.4f}")
    
    # Test translations
    print("\n" + "="*60)
    print("TRANSLATION RESULTS")
    print("="*60)
    
    test_words = ['hello', 'cat', 'car', 'big']
    
    for word in test_words:
        translations = translator.translate_word(word, k=3)
        if translations:
            print(f"\n'{word}' translates to:")
            for target, dist in translations:
                print(f"  {target:12} (distance: {dist:.3f})")
    
    return translator


def demonstrate_lsh():
    """
    Demonstrate Locality Sensitive Hashing
    """
    print("\n" + "="*60)
    print("LOCALITY SENSITIVE HASHING DEMONSTRATION")
    print("="*60)
    
    # Generate sample data
    np.random.seed(42)
    n_vectors = 1000
    n_features = 100
    
    # Create clustered data
    vectors = []
    for i in range(10):  # 10 clusters
        center = np.random.randn(n_features)
        for j in range(100):  # 100 vectors per cluster
            vec = center + np.random.randn(n_features) * 0.1
            vectors.append(vec)
    
    vectors = np.array(vectors)
    
    # Build LSH index
    print("\nBuilding LSH index...")
    lsh = LocalitySensitiveHash(n_hyperplanes=10, n_tables=5)
    lsh.fit(vectors)
    
    # Query
    query = vectors[50] + np.random.randn(n_features) * 0.05  # Near vector 50
    
    print("\nQuerying for nearest neighbors...")
    neighbors = lsh.query(query, k=10)
    
    print(f"Found {len(neighbors)} candidates")
    print("Top 5 neighbors:")
    for vec_id, dist in neighbors[:5]:
        print(f"  Vector {vec_id}: distance = {dist:.3f}")
    
    # Compare with brute force
    print("\nComparing with brute force search...")
    brute_force = []
    for i in range(len(vectors)):
        dist = np.linalg.norm(query - vectors[i])
        brute_force.append((i, dist))
    brute_force.sort(key=lambda x: x[1])
    
    print("True top 5 neighbors:")
    for vec_id, dist in brute_force[:5]:
        print(f"  Vector {vec_id}: distance = {dist:.3f}")
    
    # Calculate recall
    lsh_ids = set([n[0] for n in neighbors[:10]])
    true_ids = set([n[0] for n in brute_force[:10]])
    recall = len(lsh_ids & true_ids) / len(true_ids)
    print(f"\nRecall@10: {recall:.2%}")
    
    return lsh


def demonstrate_enhanced_translator():
    """
    Demonstrate translation with LSH
    """
    print("\n" + "="*60)
    print("ENHANCED TRANSLATOR WITH LSH")
    print("="*60)
    
    # Create enhanced translator
    translator = EnhancedTranslator(embedding_dim=50)
    
    # Generate larger vocabulary for realistic scenario
    np.random.seed(42)
    
    # Source language
    source_words = [f"word_{i}" for i in range(1000)]
    source_embeddings = {}
    for word in source_words:
        source_embeddings[word] = np.random.randn(50)
    
    # Target language (related embeddings)
    target_words = [f"mot_{i}" for i in range(1000)]
    target_embeddings = {}
    for i, word in enumerate(target_words):
        # Make somewhat related to source
        base = source_embeddings[source_words[i]]
        target_embeddings[word] = base + np.random.randn(50) * 0.3
    
    # Setup
    translator.setup(source_embeddings, target_embeddings)
    
    # Train on subset
    train_pairs = [(source_words[i], target_words[i]) for i in range(100)]
    
    print("\nTraining on 100 word pairs...")
    translator.train(train_pairs, epochs=200, verbose=False)
    
    # Compare translation speed
    import time
    
    test_word = source_words[500]
    
    # Fast translation with LSH
    start = time.time()
    fast_results = translator.translate_fast(test_word, k=5)
    fast_time = time.time() - start
    
    # Slow translation (brute force)
    start = time.time()
    slow_results = translator.translator.translate_word(test_word, k=5)
    slow_time = time.time() - start
    
    print(f"\nTranslating '{test_word}':")
    print(f"LSH time: {fast_time*1000:.2f}ms")
    print(f"Brute force time: {slow_time*1000:.2f}ms")
    print(f"Speedup: {slow_time/fast_time:.1f}x")
    
    print("\nLSH results:")
    for word, dist in fast_results:
        print(f"  {word}: {dist:.3f}")
    
    return translator


# Run all demonstrations
if __name__ == "__main__":
    # Basic translation
    translator = demonstrate_translation()
    
    # LSH demonstration
    lsh = demonstrate_lsh()
    
    # Enhanced translator with LSH
    enhanced = demonstrate_enhanced_translator()
```



## Practice Questions & Answers

### Conceptual Questions

**Q1: Why can we use a linear transformation (matrix R) to translate between languages?**

**A1:** Word embeddings in different languages often form similar geometric structures for related concepts. The assumption is that these structures are approximately linearly related - a rotation, scaling, or shearing can align them. This works because:
- Semantic relationships are preserved across languages
- Similar concepts cluster similarly in vector space
- Linear transformations preserve relative distances

**Q2: What is the Frobenius norm and why use it for the loss function?**

**A2:** The Frobenius norm is the matrix equivalent of vector magnitude:
- **Formula**: ||A||_F = √(Σᵢⱼ aᵢⱼ²)
- **Why use it**: Measures overall difference between XR and Y
- **Squared version**: Easier to differentiate (no square root)
- **Interpretation**: Total "error" across all word translations

**Q3: How does Locality Sensitive Hashing achieve O(1) search time?**

**A3:** LSH uses hash functions that map similar items to the same bucket:
1. **Hashing**: Each vector gets a hash value based on hyperplane sides
2. **Bucketing**: Vectors with same hash go in same bucket
3. **Query**: Only search within relevant buckets (not entire space)
4. **Trade-off**: Approximate results but dramatic speedup

### Mathematical Questions

**Q4: Given matrices X=[[1,2],[3,4]] and Y=[[2,3],[4,5]], and R=[[1,0],[0,1]], calculate the loss.**

**A4:**
```
XR = [[1,2],[3,4]] × [[1,0],[0,1]] = [[1,2],[3,4]]
XR - Y = [[1,2],[3,4]] - [[2,3],[4,5]] = [[-1,-1],[-1,-1]]
||XR - Y||²_F = (-1)² + (-1)² + (-1)² + (-1)² = 4
Loss = 4/2 = 2
```

**Q5: Calculate the gradient for the above example.**

**A5:**
```
g = (2/m) × X^T(XR - Y)
X^T = [[1,3],[2,4]]
XR - Y = [[-1,-1],[-1,-1]]
X^T(XR - Y) = [[1,3],[2,4]] × [[-1,-1],[-1,-1]]
             = [[-4,-4],[-6,-6]]
g = (2/2) × [[-4,-4],[-6,-6]] = [[-4,-4],[-6,-6]]
```

**Q6: If a hyperplane has normal vector n=[1,0] and point p=[0.5, 0.7], which side is p on?**

**A6:**
```
Dot product: p · n = 0.5×1 + 0.7×0 = 0.5
Since 0.5 > 0, p is on the positive side
Hash bit = 1
```

### Implementation Questions

**Q7: How do you handle words that appear in training but not in pre-trained embeddings?**

**A7:** Several strategies:
1. **Skip**: Ignore these pairs (reduces training data)
2. **Random initialization**: Create random embeddings
3. **Subword embeddings**: Use character n-grams
4. **Average similar words**: Find similar words and average
5. **Pre-filter**: Only use words with embeddings

**Q8: How many hyperplanes should you use in LSH?**

**A8:** It's a trade-off:
- **Fewer hyperplanes** (5-10):
  - Larger buckets, more false positives
  - Higher recall, lower precision
  - Faster but less accurate
- **More hyperplanes** (15-20):
  - Smaller buckets, fewer candidates
  - Lower recall, higher precision
  - Slower but more accurate
- **Multiple tables**: Increases recall without reducing precision

**Q9: What if the translation matrix R overfits to training pairs?**

**A9:** Regularization strategies:
1. **L2 regularization**: Add λ||R||²_F to loss
2. **Orthogonal constraint**: Enforce R^T R ≈ I
3. **More training data**: Use more word pairs
4. **Early stopping**: Monitor validation loss
5. **Dropout**: Randomly zero elements during training

### Debugging Questions

**Q10: Your translation matrix gives random results. What could be wrong?**

**A10:** Common issues:
1. **Learning rate**: Too high (divergence) or too low (no learning)
2. **Initialization**: Bad random initialization
3. **Data alignment**: Training pairs not properly aligned
4. **Embedding quality**: Poor quality pre-trained embeddings
5. **Insufficient training**: Need more epochs
6. **Gradient calculation**: Bug in gradient formula

**Q11: LSH returns very few or no candidates. What's the issue?**

**A11:** Possible problems:
1. **Too many hyperplanes**: Buckets too specific
2. **High dimensions**: Curse of dimensionality
3. **Data distribution**: Vectors too spread out
4. **Single table**: Need multiple hash tables
5. **Hash implementation**: Bug in hash function



## External Resources

### Foundational Papers
1. [Mikolov et al. (2013) - Exploiting Similarities among Languages for MT](https://arxiv.org/abs/1309.4168) - Word vector translation
2. [Conneau et al. (2017) - Word Translation Without Parallel Data](https://arxiv.org/abs/1710.04087) - Unsupervised alignment
3. [Artetxe et al. (2018) - Robust Self-Learning Method](https://arxiv.org/abs/1805.06297)
4. https://aman.ai/coursera-nlp/machine-translation/

### LSH Resources
1. [Indyk & Motwani (1998) - Original LSH Paper](http://www.cs.princeton.edu/courses/archive/spring04/cos598B/bib/IndykM-98.pdf)
2. [Andoni & Indyk (2008) - LSH Survey](http://people.csail.mit.edu/indyk/p117-andoni.pdf)
3. [LSH Forest Paper](https://arxiv.org/abs/1404.7808) - Improved LSH method

### Books & Courses
1. [Neural Machine Translation - Koehn](https://www.cambridge.org/core/books/neural-machine-translation/7AAA628F88ADD2B86B2666896A7091DE)
2. [Stanford CS224N - Machine Translation](http://web.stanford.edu/class/cs224n/slides/cs224n-2023-lecture07-nmt.pdf)
3. [Graham Neubig's NMT Tutorial](https://arxiv.org/abs/1703.01619)

### Video Lectures
1. [Stanford CS224N - Machine Translation](https://www.youtube.com/watch?v=7VjQRpOOHw0)
2. [LSH Explained - Mining Massive Datasets](https://www.youtube.com/watch?v=e_V84aSrHqQ)
3. [Cross-lingual Word Embeddings](https://www.youtube.com/watch?v=L3T7XzdCKX0)

### Implementation Libraries
1. [FAISS](https://github.com/facebookresearch/faiss) - Facebook's similarity search library
2. [Annoy](https://github.com/spotify/annoy) - Spotify's approximate NN library
3. [MUSE](https://github.com/facebookresearch/MUSE) - Multilingual embeddings
4. [fastText](https://fasttext.cc/docs/en/aligned-vectors.html) - Aligned word vectors

### Tools & Frameworks
1. [Gensim](https://radimrehurek.com/gensim/models/translation_matrix.html) - Translation matrix training
2. [vecmap](https://github.com/artetxem/vecmap) - Cross-lingual embeddings
3. [LASER](https://github.com/facebookresearch/LASER) - Language-agnostic embeddings

### Modern Approaches
1. [MarianMT](https://huggingface.co/Helsinki-NLP) - Neural machine translation
2. [mBART](https://arxiv.org/abs/2001.08210) - Multilingual denoising pretraining
3. [XLM-R](https://arxiv.org/abs/1911.02116) - Cross-lingual understanding

### Datasets
1. [MUSE Dictionaries](https://github.com/facebookresearch/MUSE#ground-truth-bilingual-dictionaries) - Bilingual dictionaries
2. [WMT Translation Tasks](http://www.statmt.org/) - Translation benchmarks
3. [OPUS](https://opus.nlpl.eu/) - Parallel corpora

### Practice & Evaluation
1. [BLI (Bilingual Lexicon Induction)](https://aclanthology.org/W17-2506/) - Evaluation methods
2. [XNLI](https://arxiv.org/abs/1809.05053) - Cross-lingual NLI benchmark
3. [Tatoeba](https://tatoeba.org/) - Multilingual sentence pairs



## Summary

### Key Concepts Mastered

1. **Linear Alignment**: Languages' vector spaces can be aligned with linear transformations
2. **Optimization**: Gradient descent minimizes translation error
3. **Nearest Neighbors**: Finding closest vectors for translation candidates
4. **LSH**: Trading exactness for speed in high-dimensional search
5. **Practical Implementation**: Building working translation systems

### The Translation Pipeline

```
1. Load embeddings for both languages
2. Create training pairs (aligned translations)
3. Learn transformation matrix R via gradient descent
4. For new words: Transform → Search → Return candidates
5. Use LSH for production-scale efficiency
```

### Trade-offs

| Approach | Accuracy | Speed | Memory | Use Case |
|----------|----------|-------|--------|----------|
| Brute Force KNN | Perfect | O(n×d) | O(n×d) | Small vocab |
| LSH | Approximate | O(1) avg | O(n×d×L) | Large vocab |
| Learned Index | Good | O(log n) | O(n) | Medium vocab |
| Inverted Index | Good | O(k) | O(n×m) | Sparse data |

### Evolution of Machine Translation

```
Rule-based → Statistical (SMT) → Neural (NMT) → Pre-trained (mBART/mT5)
                     ↑
            Our approach: Embedding alignment
```

### Next Steps
1. Experiment with different embedding types (Word2Vec, GloVe, FastText)
2. Try unsupervised alignment methods
3. Implement cross-lingual retrieval
4. Explore neural machine translation
5. Build multilingual applications
