# Sentiment Analysis with Logistic Regression - Complete Learning Guide

## Table of Contents
1. [Core Concepts](#core-concepts)
2. [Feature Extraction](#feature-extraction)
3. [Preprocessing Pipeline](#preprocessing-pipeline)
4. [Logistic Regression Theory](#logistic-regression-theory)
5. [Implementation Code](#implementation-code)
6. [Practice Questions & Answers](#practice-questions--answers)
7. [External Resources](#external-resources)


## Core Concepts

### What is Sentiment Analysis?
Sentiment analysis is a Natural Language Processing (NLP) task that aims to determine the emotional tone or opinion expressed in text. In binary sentiment analysis, we classify text (like tweets or reviews) as either:
- **Positive sentiment** (label = 1)
- **Negative sentiment** (label = 0)

### Why Logistic Regression?
Logistic regression is ideal for binary classification because:
- It outputs probabilities between 0 and 1 using the sigmoid function
- It's computationally efficient and interpretable
- It works well with sparse, high-dimensional text data
- It provides a strong baseline for sentiment classification tasks

### The Supervised Learning Pipeline
```
Raw Text → Preprocessing → Feature Extraction → Logistic Regression → Prediction
```


## Feature Extraction

### 1. Sparse Representation (Traditional Approach)
- Creates a vector of size |V| (vocabulary size)
- Each dimension represents presence (1) or absence (0) of a word
- **Problem**: Results in very high-dimensional, sparse vectors
- **Example**: For vocabulary ["happy", "sad", "good", "bad"], tweet "I am happy" = [1, 0, 0, 0, ...]

### 2. Frequency-Based Features (Optimized Approach)
Instead of sparse vectors, we use 3 features:
1. **Bias term**: Always 1
2. **Positive frequency sum**: Total count of words appearing in positive tweets
3. **Negative frequency sum**: Total count of words appearing in negative tweets

**Example Feature Vector**:
```
Tweet: "I love this movie"
Features: [1, sum_positive_freq, sum_negative_freq]
         = [1, 8, 3]  # hypothetical frequencies
```



## Preprocessing Pipeline

### Step 1: Tokenization
Split text into individual words/tokens.

### Step 2: Remove Stop Words & Punctuation
Filter out non-informative words like "the", "is", "at" and punctuation marks.

### Step 3: Stemming
Reduce words to their root form:
- "learning", "learned", "learns" → "learn"
- "happiness", "happy", "happier" → "happi"

### Step 4: Lower-casing
Convert all text to lowercase to treat "Great" and "great" as the same word.

### Step 5: Handle Twitter-Specific Elements
Remove or process:
- URLs
- @mentions
- Hashtags (can extract text from them)
- Retweet markers (RT)


## Logistic Regression Theory

### The Mathematical Foundation

#### 1. Linear Combination (Logit)
```
z = θ₀ + θ₁x₁ + θ₂x₂ = θᵀx
```

#### 2. Sigmoid Function (Hypothesis)
```
h(z) = σ(z) = 1 / (1 + e^(-z))
```
- Maps any real value to [0, 1]
- Interpretable as probability

#### 3. Prediction Rule
```
ŷ = {1 if h(z) ≥ 0.5
     {0 if h(z) < 0.5
```

#### 4. Cost Function (Binary Cross-Entropy)
```
J(θ) = -1/m Σ[y⁽ⁱ⁾log(ŷ⁽ⁱ⁾) + (1-y⁽ⁱ⁾)log(1-ŷ⁽ⁱ⁾)]
```

#### 5. Gradient Descent Update Rule
```
θⱼ := θⱼ - α × ∂J/∂θⱼ
where ∂J/∂θⱼ = 1/m Σ(ŷ⁽ⁱ⁾ - y⁽ⁱ⁾)xⱼ⁽ⁱ⁾
```


## Implementation Code

### Complete Python Implementation

```python
import numpy as np
import re
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
nltk.download('stopwords')

class SentimentAnalyzer:
    def __init__(self, learning_rate=0.01, num_epochs=1000):
        self.lr = learning_rate
        self.epochs = num_epochs
        self.theta = None
        self.freqs = {}
        self.stemmer = PorterStemmer()
        self.stopwords = set(stopwords.words('english'))
    
    def preprocess_tweet(self, tweet):
        """Clean and preprocess a single tweet"""
        # Remove URLs
        tweet = re.sub(r'http[s]?://\S+', '', tweet)
        # Remove mentions and hashtags
        tweet = re.sub(r'[@#]\w+', '', tweet)
        # Remove punctuation
        tweet = re.sub(r'[^\w\s]', '', tweet)
        # Tokenize and lowercase
        tokens = tweet.lower().split()
        # Remove stopwords and stem
        processed = []
        for token in tokens:
            if token not in self.stopwords and len(token) > 0:
                processed.append(self.stemmer.stem(token))
        return processed
    
    def build_freqs(self, tweets, labels):
        """Build frequency dictionary from training data"""
        self.freqs = defaultdict(lambda: [0, 0])
        
        for tweet, label in zip(tweets, labels):
            processed = self.preprocess_tweet(tweet)
            for word in processed:
                if label == 1:
                    self.freqs[word][0] += 1  # positive frequency
                else:
                    self.freqs[word][1] += 1  # negative frequency
        
        return self.freqs
    
    def extract_features(self, tweet):
        """Extract features for a single tweet"""
        processed = self.preprocess_tweet(tweet)
        
        # Initialize features: [bias, pos_sum, neg_sum]
        features = np.array([1.0, 0.0, 0.0])
        
        for word in processed:
            if word in self.freqs:
                features[1] += self.freqs[word][0]  # positive frequency
                features[2] += self.freqs[word][1]  # negative frequency
        
        return features
    
    def sigmoid(self, z):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-z))
    
    def compute_cost(self, X, y):
        """Compute binary cross-entropy cost"""
        m = len(y)
        z = np.dot(X, self.theta)
        h = self.sigmoid(z)
        
        # Add small epsilon to avoid log(0)
        epsilon = 1e-7
        h = np.clip(h, epsilon, 1 - epsilon)
        
        cost = -1/m * (np.dot(y, np.log(h)) + np.dot(1-y, np.log(1-h)))
        return cost
    
    def gradient_descent(self, X, y):
        """Perform gradient descent to optimize parameters"""
        m = len(y)
        costs = []
        
        for epoch in range(self.epochs):
            # Forward propagation
            z = np.dot(X, self.theta)
            h = self.sigmoid(z)
            
            # Compute gradients
            gradients = 1/m * np.dot(X.T, (h - y))
            
            # Update parameters
            self.theta = self.theta - self.lr * gradients
            
            # Track cost
            if epoch % 100 == 0:
                cost = self.compute_cost(X, y)
                costs.append(cost)
                print(f"Epoch {epoch}: Cost = {cost:.6f}")
        
        return costs
    
    def train(self, tweets, labels):
        """Train the logistic regression model"""
        # Build frequency dictionary
        self.build_freqs(tweets, labels)
        
        # Extract features for all tweets
        X = np.array([self.extract_features(tweet) for tweet in tweets])
        y = np.array(labels)
        
        # Initialize parameters
        self.theta = np.zeros(3)
        
        # Train model
        costs = self.gradient_descent(X, y)
        
        return costs
    
    def predict(self, tweet):
        """Predict sentiment for a single tweet"""
        features = self.extract_features(tweet)
        z = np.dot(features, self.theta)
        probability = self.sigmoid(z)
        
        prediction = 1 if probability >= 0.5 else 0
        return prediction, probability
    
    def evaluate(self, tweets, labels):
        """Evaluate model accuracy"""
        correct = 0
        predictions = []
        
        for tweet, label in zip(tweets, labels):
            pred, _ = self.predict(tweet)
            predictions.append(pred)
            if pred == label:
                correct += 1
        
        accuracy = correct / len(labels)
        return accuracy, predictions

# Example usage
if __name__ == "__main__":
    # Sample training data
    train_tweets = [
        "I love this movie, it's amazing!",
        "This film is terrible and boring",
        "Best experience ever, highly recommend!",
        "Waste of time, very disappointed",
        "Absolutely fantastic performance!",
        "Horrible acting and poor story"
    ]
    train_labels = [1, 0, 1, 0, 1, 0]  # 1=positive, 0=negative
    
    # Initialize and train model
    analyzer = SentimentAnalyzer(learning_rate=0.1, num_epochs=500)
    costs = analyzer.train(train_tweets, train_labels)
    
    # Test predictions
    test_tweets = [
        "This is wonderful!",
        "I hate this so much",
        "Not bad, quite enjoyable"
    ]
    
    for tweet in test_tweets:
        pred, prob = analyzer.predict(tweet)
        sentiment = "Positive" if pred == 1 else "Negative"
        print(f"Tweet: '{tweet}'")
        print(f"Sentiment: {sentiment} (Probability: {prob:.3f})\n")
```



## Practice Questions & Answers

### Conceptual Questions

**Q1: Why do we use the sigmoid function in logistic regression instead of a linear function?**

**A1:** The sigmoid function maps any real-valued input to a value between 0 and 1, making it perfect for probability estimation. Unlike a linear function which could output any value, sigmoid ensures our predictions are interpretable as probabilities and bounded appropriately for binary classification.

**Q2: What is the difference between sparse representation and frequency-based features?**

**A2:** 
- **Sparse representation**: Creates a vector of size |V| (vocabulary size), with 1s and 0s indicating word presence. Results in high-dimensional, mostly zero vectors.
- **Frequency-based features**: Uses only 3 features (bias, positive frequency sum, negative frequency sum), making training much more efficient while capturing sentiment information.

**Q3: Why is preprocessing important in NLP tasks?**

**A3:** Preprocessing reduces noise and dimensionality by:
- Removing non-informative words (stop words)
- Normalizing word variations (stemming, lowercasing)
- Eliminating irrelevant content (URLs, punctuation)
- This leads to better generalization and faster training

### Mathematical Questions

**Q4: Given θ = [0.5, 0.3, -0.2] and features x = [1, 5, 3], calculate the prediction.**

**A4:**
```
z = θᵀx = 0.5(1) + 0.3(5) + (-0.2)(3) = 0.5 + 1.5 - 0.6 = 1.4
h(z) = 1/(1 + e^(-1.4)) ≈ 0.802
Prediction = 1 (positive) since 0.802 > 0.5
```

**Q5: Derive the gradient for a single training example.**

**A5:**
```
Cost for single example: J = -[y·log(ŷ) + (1-y)·log(1-ŷ)]
Gradient: ∂J/∂θⱼ = (ŷ - y)·xⱼ
Where ŷ = sigmoid(θᵀx)
```

### Implementation Questions

**Q6: How would you handle class imbalance in sentiment analysis?**

**A6:** Several approaches:
1. **Weighted loss function**: Give more weight to minority class
2. **Resampling**: Oversample minority or undersample majority class
3. **Adjust threshold**: Instead of 0.5, find optimal threshold using validation set
4. **Use metrics beyond accuracy**: Precision, recall, F1-score

**Q7: How can you prevent overfitting in logistic regression?**

**A7:**
1. **Regularization**: Add L1 or L2 penalty term to cost function
2. **Feature selection**: Remove redundant or noisy features
3. **Cross-validation**: Use k-fold CV to tune hyperparameters
4. **Early stopping**: Monitor validation loss during training
5. **More training data**: Collect more diverse examples

### Debugging Questions

**Q8: Your model always predicts the majority class. What could be wrong?**

**A8:** Possible issues:
- **Learning rate too high**: Causing divergence
- **Feature scaling**: Features have very different scales
- **Initialization**: Poor parameter initialization
- **Class imbalance**: Severe imbalance not handled
- **Bug in code**: Check gradient computation and update rule

**Q9: The training loss decreases but validation accuracy doesn't improve. What's happening?**

**A9:** This indicates **overfitting**. The model memorizes training data but doesn't generalize. Solutions:
- Add regularization
- Reduce model complexity
- Use dropout (for neural networks)
- Get more training data
- Implement early stopping



## External Resources

### Video Tutorials
1. [Andrew Ng's Machine Learning Course - Logistic Regression](https://www.coursera.org/learn/machine-learning)
2. [3Blue1Brown - Neural Networks Series](https://www.youtube.com/watch?v=aircAruvnKk)
3. [StatQuest - Logistic Regression](https://www.youtube.com/watch?v=yIYKR4sgzI8)
4. https://aman.ai/coursera-nlp/logistic-regression/ 

### Reading Materials
1. [Stanford CS229 Notes on Logistic Regression](http://cs229.stanford.edu/notes2020fall/notes2020fall/cs229-notes1.pdf)
2. [Pattern Recognition and Machine Learning - Bishop](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf)
3. [The Elements of Statistical Learning](https://web.stanford.edu/~hastie/ElemStatLearn/)

### Interactive Resources
1. [Google's Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course)
2. [Kaggle Learn - Intro to Machine Learning](https://www.kaggle.com/learn/intro-to-machine-learning)
3. [Fast.ai Practical Deep Learning Course](https://course.fast.ai/)

### NLP-Specific Resources
1. [NLTK Book - Natural Language Processing with Python](https://www.nltk.org/book/)
2. [spaCy 101 - NLP Concepts](https://spacy.io/usage/spacy-101)
3. [Hugging Face NLP Course](https://huggingface.co/learn/nlp-course)

### Research Papers
1. [Pang & Lee (2008) - Opinion Mining and Sentiment Analysis](https://www.cs.cornell.edu/home/llee/omsa/omsa.pdf)
2. [Liu (2012) - Sentiment Analysis and Opinion Mining](https://www.cs.uic.edu/~liub/FBS/SentimentAnalysis-and-OpinionMining.pdf)

### Practice Platforms
1. [Kaggle - Sentiment Analysis Competitions](https://www.kaggle.com/competitions?searchQuery=sentiment+analysis)
2. [Google Colab - Free GPU for Training](https://colab.research.google.com/)
3. [Papers with Code - Sentiment Analysis Benchmarks](https://paperswithcode.com/task/sentiment-analysis)

### Libraries and Tools
1. [NLTK - Natural Language Toolkit](https://www.nltk.org/)
2. [scikit-learn - Machine Learning in Python](https://scikit-learn.org/)
3. [TextBlob - Simplified Text Processing](https://textblob.readthedocs.io/)
4. [spaCy - Industrial-Strength NLP](https://spacy.io/)

---

## Summary

Logistic regression for sentiment analysis combines:
1. **Text preprocessing** to clean and normalize data
2. **Feature engineering** to represent text numerically
3. **Mathematical optimization** using gradient descent
4. **Probabilistic classification** via the sigmoid function

This approach provides an interpretable, efficient baseline for sentiment classification that often performs surprisingly well compared to more complex models.

### Key Takeaways
- Feature quality matters more than model complexity
- Preprocessing significantly impacts performance
- Frequency-based features are efficient and effective
- Logistic regression provides probabilistic interpretations
- Always validate on unseen data to assess generalization

### Next Steps
1. Implement the code and experiment with different preprocessing steps
2. Try regularization techniques (L1/L2)
3. Explore advanced features (n-grams, TF-IDF)
4. Compare with other algorithms (Naive Bayes, SVM)
5. Scale up to larger datasets and real-world applications
