import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from transformers import BertTokenizer, TFBertForSequenceClassification

import sys
def pp(*p): print(*p);sys.exit()

# Load the training and test dataset
train_data = pd.read_csv("Subtask-1/subtask1_train.csv", sep=',')
test_data = pd.read_csv("Subtask-1/subtask1_test.csv", sep=',')

# Define input data and labels for training
train_sentences = train_data['sentence'].values
train_labels = train_data['gold_label'].values
# Define input data for testing
test_sentences = test_data['sentence'].values

# Tokenize the input sentences
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

train_tokenized_inputs = tokenizer(train_sentences.tolist(), padding=True, truncation=True, return_tensors="np")
test_tokenized_inputs = tokenizer(test_sentences.tolist(), padding=True, truncation=True, return_tensors="np")

# Create train/test split
train_inputs, _, train_labels, _ = train_test_split(
    train_tokenized_inputs['input_ids'],
    train_labels,
    test_size=0.2,
    random_state=42
)

# Model architecture
model = Sequential()
model.add(TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2))
model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_inputs, train_labels, epochs=5, batch_size=16)

# Save the model
model.save('models/transformer_model.h5')

# Make predictions on the test set
test_predictions = model.predict(test_tokenized_inputs['input_ids'])
predicted_labels = np.argmax(test_predictions.logits, axis=1)

# Save predictions to CSV
results_df = pd.DataFrame({'sentenceID': test_data['sentenceID'], 'predicted_label': predicted_labels})
results_df.to_csv('results/test_predictions.csv', index=False)