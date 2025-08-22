# Naive Bayes for Sentiment Analysis - Complete Learning Guide

## Table of Contents
1. [Probability Foundations](#probability-foundations)
2. [Naive Bayes Theory](#naive-bayes-theory)
3. [Laplacian Smoothing](#laplacian-smoothing)
4. [Log Likelihood & Numerical Stability](#log-likelihood--numerical-stability)
5. [Complete Implementation](#complete-implementation)
6. [Practice Questions & Answers](#practice-questions--answers)
7. [External Resources](#external-resources)



## Probability Foundations

### Basic Probability Concepts

**Probability of an Event**
- P(positive) = Number of positive tweets / Total tweets
- P(negative) = Number of negative tweets / Total tweets
- P(positive) + P(negative) = 1 (for binary classification)

**Example**: In a corpus of 20 tweets with 13 positive and 7 negative:
- P(positive) = 13/20 = 0.65
- P(negative) = 7/20 = 0.35

### Conditional Probability

**Definition**: Probability of event A given that event B has occurred.

```
P(A|B) = P(A ∩ B) / P(B)
```

**In Sentiment Analysis Context**:
- P(positive | "happy") = Probability tweet is positive given it contains "happy"
- P("happy" | positive) = Probability tweet contains "happy" given it's positive

### Bayes' Rule

The foundation of Naive Bayes classifier:

```
P(A|B) = P(B|A) × P(A) / P(B)
```

**For Sentiment Analysis**:
```
P(positive|tweet) = P(tweet|positive) × P(positive) / P(tweet)
```



## Naive Bayes Theory

### The "Naive" Assumption
Naive Bayes assumes **conditional independence** between features (words):
- Each word's presence is independent of other words
- Simplifies computation but isn't always realistic
- Despite this assumption, works surprisingly well in practice

### Naive Bayes Formula for Sentiment Classification

```
P(class|document) ∝ P(class) × ∏ P(word_i|class)
```

**Binary Classification Decision**:
```
Score = P(positive) × ∏ P(w_i|positive) / (P(negative) × ∏ P(w_i|negative))

If Score > 1: Positive
If Score < 1: Negative
```

### Computing Conditional Probabilities

For each word in vocabulary:
```
P(word|positive) = Count(word in positive tweets) / Total words in positive tweets
P(word|negative) = Count(word in negative tweets) / Total words in negative tweets
```



## Laplacian Smoothing

### The Zero Probability Problem
- If a word never appears in positive (or negative) tweets, P(word|class) = 0
- This causes the entire product to become 0 or undefined
- Solution: Add-one (Laplacian) smoothing

### Smoothing Formula

```
P(w_i|class) = (freq(w_i, class) + 1) / (N_class + V)
```

Where:
- `freq(w_i, class)` = frequency of word i in class
- `N_class` = total words in class
- `V` = vocabulary size (unique words across all classes)
- `+1` in numerator prevents zero probabilities
- `+V` in denominator ensures probabilities sum to 1



## Log Likelihood & Numerical Stability

### Why Use Logarithms?

1. **Prevent Numerical Underflow**: Multiplying many small probabilities (0 < p < 1) can result in numbers too small for computers to represent
2. **Computational Efficiency**: Addition is faster than multiplication
3. **Mathematical Convenience**: log(a×b) = log(a) + log(b)

### Log Likelihood Formula

Instead of:
```
Score = ∏ P(w_i|positive) / P(w_i|negative)
```

We use:
```
Log Score = Σ log(P(w_i|positive) / P(w_i|negative))
          = Σ [log P(w_i|positive) - log P(w_i|negative)]
          = Σ λ(w_i)
```

Where λ(w_i) is the log likelihood ratio for word i.

### Decision Rule with Log Likelihood

```
Log Score = log_prior + Σ λ(w_i)

If Log Score > 0: Positive
If Log Score < 0: Negative
```

---

## Complete Implementation

### Full Python Implementation with Detailed Comments

```python
import numpy as np
import re
from collections import defaultdict
from math import log
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
nltk.download('stopwords', quiet=True)

class NaiveBayesSentimentAnalyzer:
    """
    Naive Bayes classifier for sentiment analysis with Laplacian smoothing
    """
    def __init__(self):
        self.logprior = 0
        self.loglikelihood = {}
        self.vocab = set()
        self.word_freqs = {}
        self.stemmer = PorterStemmer()
        self.stopwords = set(stopwords.words('english'))
        
    def preprocess_tweet(self, tweet):
        """
        Clean and preprocess a tweet
        
        Args:
            tweet: Raw tweet text
            
        Returns:
            List of processed tokens
        """
        # Remove URLs
        tweet = re.sub(r'http[s]?://\S+', '', tweet)
        # Remove mentions and hashtags symbols (keep the text)
        tweet = re.sub(r'[@#]', '', tweet)
        # Remove punctuation and special characters
        tweet = re.sub(r'[^\w\s]', '', tweet)
        # Convert to lowercase
        tweet = tweet.lower()
        
        # Tokenize
        tokens = tweet.split()
        
        # Remove stopwords and apply stemming
        processed_tokens = []
        for token in tokens:
            if token not in self.stopwords and len(token) > 0:
                stemmed = self.stemmer.stem(token)
                processed_tokens.append(stemmed)
                
        return processed_tokens
    
    def build_freqs(self, tweets, labels):
        """
        Build frequency dictionary for words in positive and negative classes
        
        Args:
            tweets: List of tweet texts
            labels: List of labels (0 or 1)
            
        Returns:
            Dictionary mapping (word, label) to frequency
        """
        freqs = defaultdict(int)
        
        for tweet, label in zip(tweets, labels):
            processed = self.preprocess_tweet(tweet)
            for word in processed:
                # Add to vocabulary
                self.vocab.add(word)
                # Count frequency for (word, label) pair
                freqs[(word, label)] += 1
                
        return freqs
    
    def train(self, tweets, labels):
        """
        Train the Naive Bayes classifier
        
        Args:
            tweets: List of training tweets
            labels: List of labels (0 or 1)
        """
        # Build frequency dictionary
        self.word_freqs = self.build_freqs(tweets, labels)
        
        # Calculate log prior
        n_pos = sum(1 for label in labels if label == 1)
        n_neg = len(labels) - n_pos
        self.logprior = log(n_pos / n_neg) if n_neg > 0 else 0
        
        # Calculate log likelihood for each word
        self.calculate_log_likelihood()
        
        print(f"Training complete!")
        print(f"Vocabulary size: {len(self.vocab)}")
        print(f"Log prior: {self.logprior:.4f}")
        
    def calculate_log_likelihood(self):
        """
        Calculate log likelihood for each word in vocabulary
        Uses Laplacian smoothing to handle zero probabilities
        """
        # Count total words in each class
        n_pos = sum(freq for (word, label), freq in self.word_freqs.items() if label == 1)
        n_neg = sum(freq for (word, label), freq in self.word_freqs.items() if label == 0)
        
        # Vocabulary size for Laplacian smoothing
        V = len(self.vocab)
        
        # Calculate log likelihood for each word
        for word in self.vocab:
            # Get frequencies with Laplacian smoothing
            freq_pos = self.word_freqs.get((word, 1), 0) + 1
            freq_neg = self.word_freqs.get((word, 0), 0) + 1
            
            # Calculate probabilities with smoothing
            p_pos = freq_pos / (n_pos + V)
            p_neg = freq_neg / (n_neg + V)
            
            # Store log likelihood ratio
            self.loglikelihood[word] = log(p_pos / p_neg)
    
    def predict(self, tweet):
        """
        Predict sentiment of a single tweet
        
        Args:
            tweet: Raw tweet text
            
        Returns:
            Tuple of (prediction, probability score)
        """
        # Preprocess the tweet
        processed = self.preprocess_tweet(tweet)
        
        # Initialize with log prior
        score = self.logprior
        
        # Add log likelihood for each word
        for word in processed:
            if word in self.loglikelihood:
                score += self.loglikelihood[word]
            # Words not in vocabulary are ignored (neutral)
        
        # Make prediction
        prediction = 1 if score > 0 else 0
        
        # Convert score to probability (sigmoid of score)
        probability = 1 / (1 + np.exp(-score))
        
        return prediction, probability, score
    
    def evaluate(self, tweets, labels):
        """
        Evaluate model accuracy on test set
        
        Args:
            tweets: List of test tweets
            labels: List of true labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        correct = 0
        true_pos = 0
        false_pos = 0
        true_neg = 0
        false_neg = 0
        
        predictions = []
        for tweet, true_label in zip(tweets, labels):
            pred, _, _ = self.predict(tweet)
            predictions.append(pred)
            
            if pred == true_label:
                correct += 1
                if pred == 1:
                    true_pos += 1
                else:
                    true_neg += 1
            else:
                if pred == 1:
                    false_pos += 1
                else:
                    false_neg += 1
        
        accuracy = correct / len(labels) if len(labels) > 0 else 0
        precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
        recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': {
                'true_pos': true_pos,
                'false_pos': false_pos,
                'true_neg': true_neg,
                'false_neg': false_neg
            }
        }
    
    def get_most_informative_features(self, n=10):
        """
        Get the most informative features (words with highest |log likelihood|)
        
        Args:
            n: Number of features to return
            
        Returns:
            List of (word, log_likelihood) tuples
        """
        # Sort by absolute value of log likelihood
        sorted_words = sorted(self.loglikelihood.items(), 
                            key=lambda x: abs(x[1]), 
                            reverse=True)
        
        positive_words = [(w, ll) for w, ll in sorted_words if ll > 0][:n]
        negative_words = [(w, ll) for w, ll in sorted_words if ll < 0][:n]
        
        return positive_words, negative_words

# Demonstration with example usage
def demonstrate_naive_bayes():
    """
    Demonstrate Naive Bayes classifier with sample data
    """
    # Sample training data
    train_tweets = [
        "I love this movie, absolutely fantastic!",
        "This film is terrible, waste of time",
        "Amazing experience, highly recommend!",
        "Horrible acting, very disappointed",
        "Best movie I've seen this year!",
        "Worst film ever, completely boring",
        "Excellent plot and great characters",
        "Poor storyline, not worth watching",
        "Brilliant performance by the actors!",
        "Awful dialogue and bad direction"
    ]
    train_labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    
    # Initialize and train classifier
    nb_classifier = NaiveBayesSentimentAnalyzer()
    nb_classifier.train(train_tweets, train_labels)
    
    # Test tweets
    test_tweets = [
        "This is wonderful and amazing!",
        "I hate this so much, terrible",
        "Not bad, quite enjoyable actually",
        "Disappointing and boring experience"
    ]
    
    print("\n" + "="*50)
    print("PREDICTIONS ON NEW TWEETS:")
    print("="*50)
    
    for tweet in test_tweets:
        pred, prob, score = nb_classifier.predict(tweet)
        sentiment = "POSITIVE" if pred == 1 else "NEGATIVE"
        print(f"\nTweet: '{tweet}'")
        print(f"Sentiment: {sentiment}")
        print(f"Probability: {prob:.3f}")
        print(f"Log Score: {score:.3f}")
    
    # Show most informative features
    print("\n" + "="*50)
    print("MOST INFORMATIVE FEATURES:")
    print("="*50)
    
    pos_words, neg_words = nb_classifier.get_most_informative_features(5)
    
    print("\nTop Positive Words:")
    for word, score in pos_words:
        print(f"  {word:15} λ = {score:+.3f}")
    
    print("\nTop Negative Words:")
    for word, score in neg_words:
        print(f"  {word:15} λ = {score:+.3f}")
    
    # Evaluate on training set (just for demonstration)
    print("\n" + "="*50)
    print("EVALUATION METRICS (on training set):")
    print("="*50)
    
    metrics = nb_classifier.evaluate(train_tweets, train_labels)
    print(f"Accuracy:  {metrics['accuracy']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall:    {metrics['recall']:.3f}")
    print(f"F1 Score:  {metrics['f1_score']:.3f}")
    
    return nb_classifier

# Run demonstration
if __name__ == "__main__":
    classifier = demonstrate_naive_bayes()
```



## Practice Questions & Answers

### Conceptual Questions

**Q1: Why is Naive Bayes called "naive"?**

**A1:** It's called "naive" because it makes the simplifying assumption that all features (words) are conditionally independent given the class. In reality, words in language are highly dependent on each other (e.g., "not good" vs "good"), but Naive Bayes ignores these dependencies. Despite this naive assumption, it often works well in practice.

**Q2: What is Laplacian smoothing and why is it necessary?**

**A2:** Laplacian smoothing adds 1 to all frequency counts to prevent zero probabilities. Without it:
- Words that never appear in a class would have P(word|class) = 0
- This would make the entire product 0 or cause division by zero
- The formula becomes: P(word|class) = (count + 1) / (total_words + vocabulary_size)

**Q3: Why use log likelihood instead of raw probabilities?**

**A3:** Three main reasons:
1. **Numerical stability**: Multiplying many small probabilities (< 1) can cause underflow
2. **Computational efficiency**: Addition is faster than multiplication
3. **Mathematical convenience**: log(a×b) = log(a) + log(b) simplifies calculations

### Mathematical Questions

**Q4: Given the following frequencies, calculate P("happy"|positive) with Laplacian smoothing:**
- Word "happy" appears 5 times in positive tweets
- Total positive words: 100
- Vocabulary size: 500

**A4:**
```
P("happy"|positive) = (5 + 1) / (100 + 500)
                    = 6 / 600
                    = 0.01
```

**Q5: Calculate the log likelihood ratio λ for a word with:**
- P(word|positive) = 0.03
- P(word|negative) = 0.01

**A5:**
```
λ = log(P(word|positive) / P(word|negative))
  = log(0.03 / 0.01)
  = log(3)
  ≈ 1.099

Since λ > 0, this word indicates positive sentiment.
```

**Q6: If log_prior = 0.5 and a tweet has words with λ values [0.2, -0.3, 0.8, -0.1], what's the prediction?**

**A6:**
```
Log Score = log_prior + Σλ
          = 0.5 + 0.2 + (-0.3) + 0.8 + (-0.1)
          = 0.5 + 0.6
          = 1.1

Since 1.1 > 0, prediction is POSITIVE
```

### Implementation Questions

**Q7: How does Naive Bayes handle words not seen during training?**

**A7:** Words not in the training vocabulary are ignored (treated as neutral). They don't contribute to the score because:
- We can't calculate P(unseen_word|class) meaningfully
- They're assumed to be equally likely in both classes
- This is a limitation - the model can only score words it has seen

**Q8: What's the difference between Naive Bayes and Logistic Regression for text classification?**

**A8:**
| Aspect | Naive Bayes | Logistic Regression |
|--------|-------------|-------------------|
| Model Type | Generative | Discriminative |
| Training | Count frequencies | Iterative optimization |
| Assumptions | Feature independence | Linear decision boundary |
| Speed | Very fast training | Slower (gradient descent) |
| Interpretability | Probabilistic interpretation | Weight interpretation |
| Data Requirements | Works well with less data | Needs more data |

**Q9: How would you handle class imbalance in Naive Bayes?**

**A9:** Several approaches:
1. **Use the prior**: Naive Bayes naturally handles imbalance through P(class)
2. **Adjust the threshold**: Instead of 0, use a different decision threshold
3. **Weighted sampling**: Oversample minority or undersample majority class
4. **Cost-sensitive learning**: Apply different misclassification costs
5. **Use appropriate metrics**: Focus on F1, precision, recall rather than accuracy

### Debugging Questions

**Q10: Your Naive Bayes classifier predicts everything as positive. What could be wrong?**

**A10:** Possible issues:
1. **Severe class imbalance**: Check the log prior value
2. **Vocabulary issues**: Most words only appear in positive class
3. **Smoothing problem**: Incorrect implementation of Laplacian smoothing
4. **Preprocessing errors**: Stop words or stemming removing negative indicators
5. **Bug in calculation**: Check log likelihood calculations

**Q11: Why might "not good" be classified as positive by Naive Bayes?**

**A11:** This is due to the independence assumption:
- Naive Bayes treats "not" and "good" as independent words
- "good" has strong positive association
- "not" might be neutral or weakly negative
- The model misses that "not" negates "good"
- Solution: Use bigrams (word pairs) as features or more sophisticated models



## External Resources

### Foundational Theory
1. [Stanford CS124 - Naive Bayes](https://web.stanford.edu/~jurafsky/slp3/4.pdf) - Dan Jurafsky's comprehensive NLP textbook chapter
2. [Andrew Ng's CS229 Notes - Naive Bayes](http://cs229.stanford.edu/notes2020fall/notes2020fall/cs229-notes2.pdf)
3. [Tom Mitchell - Machine Learning Book](http://www.cs.cmu.edu/~tom/mlbook.html) - Classic ML textbook with Naive Bayes coverage
4. https://aman.ai/coursera-nlp/naive-bayes/

### Video Lectures
1. [StatQuest - Naive Bayes](https://www.youtube.com/watch?v=O2L2Uv9pdDA) - Intuitive visual explanation
2. [Andrew Ng's Coursera - Naive Bayes](https://www.coursera.org/learn/machine-learning)
3. [3Blue1Brown - Bayes Theorem](https://www.youtube.com/watch?v=HZGCoVF3YvM) - Beautiful visual explanation

### Interactive Tutorials
1. [Scikit-learn Naive Bayes Tutorial](https://scikit-learn.org/stable/modules/naive_bayes.html)
2. [Google's Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course)
3. [Kaggle Learn - Text Classification](https://www.kaggle.com/learn/natural-language-processing)

### Research Papers
1. [McCallum & Nigam (1998) - A Comparison of Event Models for Naive Bayes](http://www.cs.cmu.edu/~knigam/papers/multinomial-aaaiws98.pdf)
2. [Rennie et al. (2003) - Tackling the Poor Assumptions of Naive Bayes](https://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf)
3. [Manning et al. (2008) - Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/)

### Implementation Resources
1. [NLTK Book - Chapter 6: Learning to Classify Text](https://www.nltk.org/book/ch06.html)
2. [TextBlob - Naive Bayes Implementation](https://textblob.readthedocs.io/en/dev/classifiers.html)
3. [Scikit-learn Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html)

### Advanced Topics
1. [Complement Naive Bayes](https://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf) - Handles imbalanced data better
2. [Bernoulli vs Multinomial Naive Bayes](https://nlp.stanford.edu/IR-book/html/htmledition/the-bernoulli-model-1.html)
3. [Semi-supervised Naive Bayes](https://www.cs.cmu.edu/~tom/pubs/NigamEtAl-bookChapter.pdf)

### Practice Platforms
1. [Kaggle - Sentiment Analysis Competitions](https://www.kaggle.com/competitions?search=sentiment)
2. [UCI ML Repository - Text Datasets](https://archive.ics.uci.edu/ml/datasets.php)
3. [Papers with Code - Text Classification](https://paperswithcode.com/task/text-classification)

### Comparison with Other Methods
1. [Naive Bayes vs Logistic Regression](https://www.cs.cmu.edu/~tom/mlbook/NBayesLogReg.pdf)
2. [When to Use Naive Bayes](https://machinelearningmastery.com/naive-bayes-for-machine-learning/)
3. [Text Classification Algorithms Comparison](https://arxiv.org/abs/1901.02318)

---

## Summary

### Key Takeaways

1. **Probability Foundation**: Naive Bayes is grounded in Bayes' theorem and conditional probability
2. **Independence Assumption**: Assumes features are independent (naive but often effective)
3. **Laplacian Smoothing**: Essential to handle unseen words and prevent zero probabilities
4. **Log Likelihood**: Prevents numerical underflow and simplifies computation
5. **Fast Training**: Just counting frequencies - no iterative optimization needed
6. **Interpretability**: Each word's contribution is transparent through λ values

### Strengths of Naive Bayes
- ✅ Fast training and prediction
- ✅ Works well with small datasets
- ✅ Handles high-dimensional data well
- ✅ Naturally probabilistic
- ✅ Easy to implement and interpret

### Limitations
- ❌ Independence assumption often violated
- ❌ Can't capture word interactions (e.g., "not good")
- ❌ Sensitive to input representation
- ❌ Zero probability problem without smoothing
- ❌ Can't learn feature interactions

### When to Use Naive Bayes
- **Good for**: Text classification, spam filtering, sentiment analysis baseline
- **Not ideal for**: Tasks requiring understanding of word order or complex dependencies
- **As a baseline**: Always try Naive Bayes first - it's simple and often surprisingly effective

### Next Steps
1. Implement the code and experiment with different preprocessing
2. Try different smoothing parameters
3. Experiment with bigrams/n-grams as features
4. Compare with Logistic Regression performance
5. Explore Complement Naive Bayes for imbalanced data
6. Learn about more sophisticated models (SVM, Neural Networks)
