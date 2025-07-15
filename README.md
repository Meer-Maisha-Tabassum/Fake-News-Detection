# Fake News Detection using Deep Learning

## Project Overview

This project implements a deep learning model to classify news articles as either "Fake" or "True" using a Bidirectional LSTM neural network. The model achieves 98% accuracy in distinguishing between genuine and fabricated news stories by analyzing textual content.

## Key Features

- **Data Preprocessing Pipeline**: Comprehensive text cleaning including lowercasing, URL removal, special character removal, tokenization, stopword removal, and lemmatization
- **Deep Learning Architecture**: Bidirectional LSTM model with Embedding, Spatial Dropout, and Dense layers
- **High Performance**: Achieves 98% accuracy on test data
- **Confusion Matrix Analysis**: Detailed evaluation metrics including precision, recall, and F1-score
- **Sample Predictions**: Demonstration of the model's ability to classify new, unseen text samples

## Dataset

The model was trained on a dataset containing:
- 23,481 fake news articles
- 21,417 true news articles

Dataset columns:
- Title: The headline of the news article
- Text: The main content of the article
- Subject: The topic/category of the article
- Date: Publication date

## Model Architecture

```python
model = Sequential()
model.add(Embedding(max_words, 256, input_length=max_len))
model.add(SpatialDropout1D(0.5))
model.add(Bidirectional(LSTM(128)))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
```

## Performance Metrics

| Metric    | Fake News | True News | Weighted Avg |
|-----------|-----------|-----------|--------------|
| Precision | 0.99      | 0.98      | 0.98         |
| Recall    | 0.98      | 0.99      | 0.98         |
| F1-Score  | 0.98      | 0.98      | 0.98         |
| Accuracy  |           |           | 0.98         |

## Confusion Matrix

```
        Predicted
        Fake  True
Actual
Fake    4439    91
True     43   4222
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/fake-news-detection.git
cd fake-news-detection
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model
```python
# Load and preprocess data
fake_df = pd.read_csv('Fake.csv')
true_df = pd.read_csv('True.csv')

# Preprocess text data
fake_df['text'] = fake_df['text'].apply(preprocess_text)
true_df['text'] = true_df['text'].apply(preprocess_text)

# Train the model
model.fit(X_train_pad, y_train, epochs=5, batch_size=64, validation_split=0.2)
```

### Making Predictions
```python
sample_text = "(Reuters) - Important news event reported by reliable source"
sample_sequence = tokenizer.texts_to_sequences([sample_text])
sample_pad = pad_sequences(sample_sequence, maxlen=max_len)
prediction = model.predict(sample_pad)
label = "True" if prediction >= 0.5 else "Fake"
print(f"Prediction: {label}")
```

## Sample Predictions

Input Text | Prediction
---------- | ----------
"Local School Holds Fundraiser for Children's Hospital." | Fake
"(Reuters) - Alabama officials certified election results..." | True
"Scientists discover a new breakthrough in technology." | Fake
"TEGUCIGALPA (Reuters) - Honduras political crisis..." | True

## Dependencies

- Python 3.7+
- TensorFlow 2.x
- Keras
- Pandas
- Numpy
- NLTK
- Scikit-learn

## File Structure

```
fake-news-detection/
├── data/
│   ├── Fake.csv
│   └── True.csv
├── models/
│   └── fake_news_model.h5
├── notebooks/
│   └── Fake_News_Detection.ipynb
├── utils/
│   └── preprocessing.py
├── requirements.txt
└── README.md
```

## Future Improvements

- Implement real-time news classification
- Add multi-language support
- Develop browser extension for online news verification
- Incorporate additional metadata features (source credibility, author reputation)
- Deploy as a web service using Flask/Django

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your improvements.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
