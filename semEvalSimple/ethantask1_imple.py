import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from gensim.models import Word2Vec
import numpy as np
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm  
"""
convert to notebook, verify that you can follow training progress
increase model size, check out the actual mode arch
make sure model will be saved
try understand subtask 2, prototype, and train

"""
class BinaryClassificationModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BinaryClassificationModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

class BinaryClassificationDataset(Dataset):
    def __init__(self, dataframe, word2vec_model):
        self.data = dataframe
        self.word2vec_model = word2vec_model

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        sentence = self.vectorize_text(row['sentence'])
        actual_sentence = row['sentence']
        label = row['gold_label']
        return {'sentence': sentence,'actual_sentence': actual_sentence, 'label': label}

    def vectorize_text(self, text):
        if pd.isnull(text):
            return np.zeros(self.word2vec_model.vector_size)

        text = str(text)
        words = text.split()
        vectors = [self.word2vec_model.wv[word] for word in words if word in self.word2vec_model.wv]
        if not vectors:
            return np.zeros(self.word2vec_model.vector_size)
        return np.mean(vectors, axis=0)

def preprocess_data(data_path):
    df = pd.read_csv(data_path)
    return df

def train_word2vec_model(sentences):
    model = Word2Vec(sentences, size=100, window=5, min_count=1, workers=4)
    return model

def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    train_losses = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        # Wrap the train_loader with tqdm for a progress bar
        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
            sentences = torch.FloatTensor(batch['sentence'].float())  # Convert to float
            labels = torch.FloatTensor(batch['label'].float())  # Convert to float

            outputs = model(sentences)
            outputs = outputs.view(-1)  # Flatten
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        train_losses.append(total_loss / len(train_loader))
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_loader)}')

    # Plot the training loss
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.show()

def evaluate_model(model, test_loader):
    model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for batch in test_loader:
            sentences = torch.FloatTensor(batch['sentence'].float())
            labels = batch['label'].float().numpy()

            outputs = model(sentences).numpy()
            predictions = (outputs > 0.5).astype(int)

            all_labels.extend(labels)
            all_predictions.extend(predictions)

    if not all_labels or not all_predictions:
        print("No evaluation data.")
        return 0, 0, 0, 0

    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions)
    recall = recall_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions)

    return accuracy, precision, recall, f1

def display_examples(dataset, model, test_loader, num_examples=5, batch_size=32):
    model.eval()
    examples_displayed = 0

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            sentences = torch.FloatTensor(batch['sentence'].float())
            labels = batch['label'].float().numpy()

            outputs = model(sentences).numpy()
            predictions = (outputs > 0.5).astype(int)

            for j in range(len(labels)):
                print(f'\nExample {i * batch_size + j + 1}:')
                print(f'Sentence: {dataset.data.iloc[i * batch_size + j]["sentence"]}')
                print(f'True Label: {labels[j]}')
                print(f'Predicted Label: {predictions[j]}')

                examples_displayed += 1

                if examples_displayed >= num_examples:
                    return

def save_predictions_to_csv(predictions, true_labels, sentences, file_path='results/results.csv'):
    result_df = pd.DataFrame({
        'True Label': true_labels,
        'Predicted Label': predictions,
        'Sentence': sentences
    })
    result_df.to_csv(file_path, index=False)
    print(f"Predictions saved to {file_path}")


# ======================================
# ================ MAIN ================
# ======================================
def main():
    parser = argparse.ArgumentParser(description='Train or evaluate the binary classification model.')
    parser.add_argument('--retrain', action='store_true', help='Retrain the model if provided.')
    parser.add_argument('--examples', type=int, default=0, help='Number of examples to display.')
    parser.add_argument('--eval', action='store_true', help='show evaluations.')
    args = parser.parse_args()

    # ====== preprocessing ====== 
    data_path = 'Subtask-1/subtask1_train.csv'
    df = preprocess_data(data_path)

    sentences = [text.split() for text in df['sentence'] if not pd.isnull(text)]
    word2vec_model = train_word2vec_model(sentences)

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    input_size = word2vec_model.vector_size
    hidden_size = 32
    output_size = 1
    learning_rate = 0.001
    batch_size = 32
    num_epochs = 100

    model = BinaryClassificationModel(input_size=input_size, hidden_size=hidden_size, output_size=output_size)

    # ====== training or loading ======
    model_path = 'models/binary_model.pth'
    if args.retrain:
        print("Retraining the model.")
        dataset = BinaryClassificationDataset(train_df, word2vec_model)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        train_model(model, train_loader, criterion, optimizer, num_epochs=num_epochs)
    else:
        model.load_state_dict(torch.load(model_path))
        print("Model loaded from:", model_path)

    # ====== evaluation ======    
    dataset_train = BinaryClassificationDataset(train_df, word2vec_model)
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=False)
    accuracy_train, precision_train, recall_train, f1_train = evaluate_model(model, loader_train)

    dataset_test = BinaryClassificationDataset(test_df, word2vec_model)
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
    accuracy_test, precision_test, recall_test, f1_test = evaluate_model(model, loader_test)

    # ====== eval results ======
    print("\nEvaluation on Train Data:")
    print(f'Accuracy: {accuracy_train:.4f}, Precision: {precision_train:.4f}, Recall: {recall_train:.4f}, F1 Score: {f1_train:.4f}')

    print("\nEvaluation on Test Data:")
    print(f'Accuracy: {accuracy_test:.4f}, Precision: {precision_test:.4f}, Recall: {recall_test:.4f}, F1 Score: {f1_test:.4f}\n')

    # ====== Display examples ======
    if args.examples > 0:
        print(f'\nDisplaying {args.examples} examples:')
        display_examples(dataset_test, model, loader_test, num_examples=args.examples)
    
    
    # Save the model if retraining or not provided
    if args.retrain or not os.path.exists(model_path):
        print("Saving the model.")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(model.state_dict(), model_path)

    # Save predictions to CSV
    test_predictions = []
    test_true_labels = []
    test_sentences = []

    with torch.no_grad():
        for batch in loader_test:
            sentences = batch['actual_sentence']
            labels = batch['label'].float().numpy()

            outputs = model(torch.FloatTensor(batch['sentence'].float())).numpy()
            predictions = (outputs > 0.5).astype(int)

            test_predictions.extend(predictions)
            test_true_labels.extend(labels)
            test_sentences.extend(sentences)

    save_predictions_to_csv(test_predictions, test_true_labels, test_sentences)

if __name__ == "__main__":
    main()
