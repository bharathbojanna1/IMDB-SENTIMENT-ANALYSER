
# 🎬 IMDb Sentiment Analysis Pipeline

A comprehensive machine learning pipeline for analyzing sentiment in movie reviews using natural language processing and multiple classification algorithms. This project provides end-to-end functionality from data preprocessing to model evaluation and prediction.

## 🌟 Features

- **Advanced Text Preprocessing**: HTML tag removal, URL cleaning, tokenization, lemmatization
- **Feature Engineering**: TF-IDF vectorization with n-grams, numerical feature extraction
- **Multiple ML Models**: Logistic Regression, Naive Bayes, Random Forest, SVM
- **Comprehensive Evaluation**: Cross-validation, ROC-AUC, confusion matrices
- **Data Visualization**: Distribution plots, correlation matrices, word clouds
- **Interactive Predictions**: Real-time sentiment prediction for new reviews
- **Performance Comparison**: Automated model selection based on accuracy

## 📋 Requirements

### Python Version
- Python 3.7 or higher

### Dependencies

```bash
# Core data science libraries
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
scipy>=1.7.0

# Visualization
matplotlib>=3.4.0
seaborn>=0.11.0
wordcloud>=1.8.0

# Natural Language Processing
nltk>=3.6.0

# Utilities
warnings
re
```

### Installation

```bash
pip install pandas numpy scikit-learn matplotlib seaborn nltk wordcloud scipy
```

## 🚀 Getting Started

### 1. Data Preparation

Ensure your IMDb dataset is in CSV format with the following structure:
```
review,sentiment
"Great movie, loved it!",positive
"Terrible acting, boring plot.",negative
```

Update the file path in the code:
```python
file_path = r"path/to/your/Imdb_data.csv"
```

### 2. Running the Analysis

```bash
python sentiment_analysis.py
```

The pipeline will automatically:
1. Load and explore your dataset
2. Generate visualizations
3. Preprocess text data
4. Train multiple models
5. Evaluate and compare performance
6. Create word clouds
7. Demonstrate predictions

## 📊 Pipeline Overview

### 1. Data Loading & Exploration
- Dataset shape and structure analysis
- Missing value detection
- Class distribution visualization
- Sample data inspection

### 2. Data Visualization
- **Sentiment Distribution**: Bar chart of positive/negative reviews
- **Review Length Analysis**: Histogram of word counts
- **Character Count Distribution**: Distribution of review lengths
- **Feature Correlations**: Heatmap of numerical features

### 3. Text Preprocessing
```python
def preprocess_text(text):
    # Convert to lowercase
    # Remove HTML tags and URLs
    # Remove punctuation and numbers
    # Tokenization and lemmatization
    # Remove stopwords
    return cleaned_text
```

### 4. Feature Engineering
- **TF-IDF Vectorization**: 5000 features, unigrams + bigrams
- **Numerical Features**: Character count, word count, average word length
- **Feature Scaling**: MinMaxScaler for compatibility with all models
- **Feature Combination**: Sparse matrix concatenation

### 5. Model Training & Evaluation

#### Models Implemented:
1. **Logistic Regression**: Linear classifier with regularization
2. **Naive Bayes**: Multinomial NB optimized for text
3. **Random Forest**: Ensemble method with 100 trees
4. **Support Vector Machine**: Linear SVM classifier

#### Evaluation Metrics:
- **Accuracy Score**: Overall classification accuracy
- **ROC-AUC Score**: Area under the ROC curve  
- **Cross-Validation**: 5-fold CV with mean and standard deviation
- **Classification Report**: Precision, recall, F1-score
- **Confusion Matrix**: Visual representation of predictions

## 🔧 Configuration Options

### TF-IDF Parameters
```python
TfidfVectorizer(
    max_features=5000,      # Maximum number of features
    ngram_range=(1, 2),     # Include unigrams and bigrams
    min_df=2,               # Minimum document frequency
    max_df=0.95             # Maximum document frequency
)
```

### Model Parameters
```python
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": MultinomialNB(alpha=0.1),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "SVM": LinearSVC(max_iter=1000)
}
```

## 📈 Expected Output

### Console Output
```
🎬 IMDb SENTIMENT ANALYSIS PIPELINE
=====================================
🔹 Dataset Shape: (50000, 2)
🔹 Columns: ['review', 'sentiment']

🔧 FEATURE ENGINEERING
---------------------
📝 Preprocessing text...
📊 Creating TF-IDF features...
✅ Final feature matrix shape: (50000, 5003)

🤖 MODEL TRAINING & EVALUATION
------------------------------
🔄 Training Logistic Regression...
✅ Logistic Regression:
   Accuracy: 0.8924
   ROC-AUC: 0.9542
   CV Score: 0.8901 (±0.0045)
```

### Visualizations Generated
1. **Data Exploration Dashboard**: 6-panel visualization
2. **Model Comparison Chart**: Performance comparison
3. **Confusion Matrix**: Best model evaluation
4. **Word Clouds**: Positive and negative sentiment visualizations

## 🎯 Usage Examples

### Predicting New Reviews
```python
# After running the main pipeline
sample_review = "This movie was absolutely fantastic!"
sentiment, confidence = predict_sentiment(
    sample_review, best_model, tfidf_vectorizer, scaler
)
print(f"Sentiment: {sentiment} (Confidence: {confidence:.3f})")
```

### Custom Predictions
```python
# Interactive usage
new_reviews = [
    "Amazing cinematography and stellar performances!",
    "Boring and predictable storyline.",
    "Mixed feelings about this one."
]

for review in new_reviews:
    sentiment, confidence = predict_sentiment(review, model, tfidf, scaler)
    print(f"'{review}' → {sentiment}")
```

## 🔍 Troubleshooting

### Common Issues

1. **File Not Found Error**:
   ```
   ❌ Error: CSV file not found
   ```
   - Verify the file path in the `load_and_explore_data()` function
   - Ensure the CSV file exists and is accessible

2. **NLTK Data Missing**:
   ```python
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```

3. **Memory Issues with Large Datasets**:
   - Reduce `max_features` in TfidfVectorizer
   - Use smaller data samples for testing
   - Consider using sparse matrices efficiently

4. **Model Training Failures**:
   - Check for empty reviews after preprocessing
   - Verify feature matrix dimensions
   - Ensure balanced class distribution

### Performance Optimization

- **For Large Datasets**: Use `n_jobs=-1` in sklearn models
- **Memory Optimization**: Process data in chunks
- **Speed Improvement**: Reduce cross-validation folds
- **Feature Selection**: Use SelectKBest for dimensionality reduction

## 📁 File Structure

```
sentiment_analysis/
├── sentiment_analysis.py      # Main pipeline script
├── README.md                  # This documentation
├── requirements.txt           # Python dependencies
├── data/
│   └── Imdb_data.csv         # IMDb dataset
└── outputs/
    ├── visualizations/        # Generated plots
    └── models/               # Saved model objects
```

## 🔬 Technical Details

### Text Preprocessing Pipeline
1. Convert to lowercase
2. Remove HTML tags using regex
3. Remove URLs and special characters
4. Tokenization with NLTK
5. Remove stopwords and short words
6. Lemmatization for word normalization

### Feature Engineering Strategy
- **TF-IDF**: Captures word importance across documents
- **N-grams**: Includes bigrams for contextual information
- **Numerical Features**: Review statistics for additional signals
- **Scaling**: MinMax scaling for algorithm compatibility

### Model Selection Criteria
- **Primary**: Accuracy on test set
- **Secondary**: Cross-validation stability
- **Tertiary**: ROC-AUC score for probability calibration

## 📊 Performance Benchmarks

Typical performance on IMDb dataset:
- **Logistic Regression**: ~89% accuracy
- **SVM**: ~88% accuracy  
- **Random Forest**: ~85% accuracy
- **Naive Bayes**: ~83% accuracy

## 🤝 Contributing

To contribute to this project:
1. Fork the repository
2. Create a feature branch
3. Add comprehensive tests
4. Update documentation
5. Submit a pull request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add docstrings to all functions
- Include error handling
- Test with different dataset sizes

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **IMDb Dataset**: Large Movie Review Dataset
- **Scikit-learn**: Machine learning library
- **NLTK**: Natural language processing toolkit
- **Pandas**: Data manipulation and analysis
- **Matplotlib/Seaborn**: Data visualization

## 📞 Support

For questions or issues:
1. Check the troubleshooting section
2. Review the error messages carefully
3. Ensure all dependencies are installed
4. Verify data format and file paths

---

**Note**: This pipeline is designed for educational and research purposes. For production use, consider additional optimizations such as model serialization, API endpoints, and real-time processing capabilities.
