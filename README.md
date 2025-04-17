# 📰 Fake News Detection using TensorFlow (LSTM)

This project demonstrates how to classify news articles as real or fake using Natural Language Processing (NLP) and LSTM-based deep learning models in TensorFlow.

## 📌 Overview

- Preprocesses news text using tokenization and padding
- Uses LSTM for sequence learning
- Trained on a fake news dataset with binary classification

## 📦 Dependencies

```bash
pip install tensorflow pandas scikit-learn nltk
```

## 📁 Dataset

Use a dataset like [Fake and Real News Dataset](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset).

Expected CSV format:
- `text`: full article text
- `label`: "FAKE" or "REAL"

## 🚀 How to Run

1. Download and extract dataset into a `data/` folder.
2. Run the script:
```bash
python fake_news_detection.py
```

## 📂 Files

- `fake_news_detection.py`: Main script
- `README.md`: Project overview and usage

## 📊 Output

- Training and validation accuracy/loss
- Final evaluation score on test set