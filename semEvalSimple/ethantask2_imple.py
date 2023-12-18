"""
Paper:
ETHAN at SemEval-2020 Task 5: Modelling Causal Reasoning in
Language using neuro-symbolic cloud computing
Training Data is like : ID Sentence Antecedent
the goal of subtask 2 is to predict the presence
of antecedent by combining neural network architecture
and neurosymbolism.

TODO: implement consistuency parser
NOTE: 	use this https://demo.allennlp.org/constituency-parsing
		and this https://demo.allennlp.org/dependency-parsing

Example data:
sentenceID	sentence	antecedent
200000	I don't think any of us---even economic gurus like Paul Krugman---really, truly understand just how bad it could've gotten "on Main Street" if the stimulus bill had become hamstrung by a filibuster threat or recalcitrant conservadems, the way so much of our legislation has since.	if the stimulus bill had become hamstrung by a filibuster threat or recalcitrant conservadems
------
https://chat.openai.com/c/7cdc1342-381a-42c4-8b78-54f453fe13ce
"""
import spacy
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import pandas as pd

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Assuming you have the training data in a CSV file named 'training_data.csv'
file_path = 'Subtask-2/task2_train.csv'

# Read the CSV file into a pandas DataFrame
df = pd.read_csv(file_path)

# Define a function to extract syntactic information using spaCy
def extract_syntactic_info(sentence):
    doc = nlp(sentence)
    syntactic_info = []
    for token in doc:
        syntactic_info.append({
            'text': token.text,
            'lemma': token.lemma_,
            'pos': token.pos_,
            'dep': token.dep_,
        })
    return syntactic_info

# Apply the function to the dataset
df['syntactic_info'] = df['sentence'].apply(extract_syntactic_info)

# Tokenize the text data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['sentence'])

# Convert sentences to sequences
X = tokenizer.texts_to_sequences(df['sentence'])

# Pad sequences to ensure they have the same length
X = pad_sequences(X)

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=100, input_length=X.shape[1]),
    tf.keras.layers.LSTM(100, return_sequences=True),
    tf.keras.layers.Attention(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification for antecedent detection
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Define target variable
y_antecedent = df['antecedent_label']  # Assuming antecedent_label is 0 or 1

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y_antecedent, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=5, batch_size=32)

# Predict antecedent probabilities
y_pred_antecedent = model.predict(X_val)

# Create a DataFrame with results
results_df = pd.DataFrame({
    'sentenceID': df['sentenceID'],
    'clabel': [1 if p > 0.5 else 0 for p in y_pred_antecedent],  # Assuming a threshold of 0.5
    'cprob': y_pred_antecedent.flatten(),
    'consequent': df['consequent'],
    'gold_consequent': df['gold_consequent'],
    'alabel': y_val.tolist(),
    'aprob': y_pred_antecedent.flatten(),
    'antecedent': df['antecedent'],
    'gold_antecedent': df['gold_antecedent']
})

# Display the results DataFrame
print(results_df)

