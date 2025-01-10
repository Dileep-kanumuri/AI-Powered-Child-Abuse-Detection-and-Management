import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import torch

# Load dataset
dataset = pd.read_csv("data/Final_Dataset_CAPS.csv")
dataset = dataset.dropna(subset=["Case Description"])

# Encode labels
label_encoder = LabelEncoder()
dataset["Abuse Type Label"] = label_encoder.fit_transform(dataset["Abuse Type"])

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Tokenization function
def tokenize_function(texts, tokenizer, max_length=128):
    return tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

# Split data into training and test sets
train_texts, test_texts, train_labels, test_labels = train_test_split(
    dataset["Case Description"],
    dataset["Abuse Type Label"],
    test_size=0.2,
    random_state=42,
)

# Tokenize datasets
train_encodings = tokenize_function(train_texts.tolist(), tokenizer)
test_encodings = tokenize_function(test_texts.tolist(), tokenizer)

# Create PyTorch datasets
train_dataset = TensorDataset(
    train_encodings["input_ids"],
    train_encodings["attention_mask"],
    torch.tensor(train_labels.values, dtype=torch.long),
)
test_dataset = TensorDataset(
    test_encodings["input_ids"],
    test_encodings["attention_mask"],
    torch.tensor(test_labels.values, dtype=torch.long),
)

# Split training data into train and validation sets
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_subset, val_subset = random_split(train_dataset, [train_size, val_size])

# Hyperparameters
learning_rate = 3e-5  # Reduced learning rate for fine-tuning
batch_size = 16
dropout_rate = 0.1  # Default dropout, can increase for regularization
epochs = 10

# DataLoaders
train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Initialize model with modified dropout rate
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", 
    num_labels=len(label_encoder.classes_),
    hidden_dropout_prob=dropout_rate,
)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Fine-tune by unfreezing more layers
for name, param in model.bert.named_parameters():
    if name.startswith("encoder.layer.11") or name.startswith("encoder.layer.10"):
        param.requires_grad = True
    else:
        param.requires_grad = False

# Class weights for imbalance
class_counts = np.bincount(train_labels)
class_weights = torch.tensor(1.0 / class_counts, dtype=torch.float32).to(device)
loss_fn = CrossEntropyLoss(weight=class_weights)

# Optimizer
optimizer = AdamW(model.parameters(), lr=learning_rate)

# Training and validation loop with early stopping
best_val_loss = float("inf")
patience = 3
epochs_no_improve = 0

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    
    # Training
    model.train()
    train_loss = 0
    for batch in tqdm(train_loader):
        input_ids, attention_mask, labels = [x.to(device) for x in batch]
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)
    print(f"Training Loss: {train_loss:.4f}")
    
    # Validation
    model.eval()
    val_loss = 0
    val_predictions, val_true_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids, attention_mask, labels = [x.to(device) for x in batch]
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            val_loss += outputs.loss.item()
            logits = outputs.logits
            val_predictions.extend(torch.argmax(logits, axis=1).cpu().numpy())
            val_true_labels.extend(labels.cpu().numpy())

    val_loss /= len(val_loader)
    val_accuracy = accuracy_score(val_true_labels, val_predictions)
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        save_directory = "bert_abuse_model_v2"  # Save updated folder
        model.save_pretrained(save_directory)
        tokenizer.save_pretrained(save_directory)
        print(f"Best model saved to: {save_directory}")
    else:
        epochs_no_improve += 1
        if epochs_no_improve == patience:
            print("Early stopping triggered!")
            break

# Load the best model for testing
model = BertForSequenceClassification.from_pretrained(save_directory)
tokenizer = BertTokenizer.from_pretrained(save_directory)
model = model.to(device)

# Testing
model.eval()
test_predictions, test_true_labels = [], []
with torch.no_grad():
    for batch in test_loader:
        input_ids, attention_mask, labels = [x.to(device) for x in batch]
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        test_predictions.extend(torch.argmax(logits, axis=1).cpu().numpy())
        test_true_labels.extend(labels.cpu().numpy())

# Test metrics
test_accuracy = accuracy_score(test_true_labels, test_predictions)
test_precision = precision_score(test_true_labels, test_predictions, average="weighted", zero_division=0)
test_recall = recall_score(test_true_labels, test_predictions, average="weighted", zero_division=0)
test_f1 = f1_score(test_true_labels, test_predictions, average="weighted", zero_division=0)
test_roc_auc = roc_auc_score(
    pd.get_dummies(test_true_labels).reindex(columns=range(len(label_encoder.classes_)), fill_value=0),
    pd.get_dummies(test_predictions).reindex(columns=range(len(label_encoder.classes_)), fill_value=0),
    average="weighted"
)

print("\nTest Metrics:")
print(f"Accuracy: {test_accuracy:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall: {test_recall:.4f}")
print(f"F1 Score: {test_f1:.4f}")
print(f"AUC-ROC: {test_roc_auc:.4f}")
