import pandas as pd
import numpy as np
import tensorflow as tf
import nltk
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# Load dataset
df = pd.read_csv("data/news.csv")  # Ensure you have a CSV file with 'text' and 'label' columns
df.dropna(inplace=True)

# Preprocess text
def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    tokens = text.lower().split()
    filtered = [word for word in tokens if word not in stop_words]
    return " ".join(filtered)

df['clean_text'] = df['text'].apply(clean_text)

# Encode labels
label_enc = LabelEncoder()
y = label_enc.fit_transform(df['label'])

# Tokenize and pad sequences
max_words = 5000
max_len = 500

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(df['clean_text'])
sequences = tokenizer.texts_to_sequences(df['clean_text'])
X = pad_sequences(sequences, maxlen=max_len)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=max_words, output_dim=64, input_length=max_len),
    tf.keras.layers.LSTM(64, return_sequences=False),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))

# Evaluate
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.2f}")