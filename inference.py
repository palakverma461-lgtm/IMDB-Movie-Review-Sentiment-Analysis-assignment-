import os
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt

MAX_LEN = 250
BATCH_SIZE = 16
MODEL_DIR = "bert_sentiment_model"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

if not os.path.exists(MODEL_DIR):
    raise FileNotFoundError(f"'{MODEL_DIR}' folder nahi mila. Pehle 'train.py' chalayein.")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(DEVICE)
model.eval()

class IMDBEvalDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_len
        )
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.encodings["input_ids"][idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.encodings["attention_mask"][idx], dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.float)
        }

with open("test_data.pkl", "rb") as f:
    test_texts, test_labels = pickle.load(f)

test_dataset = IMDBEvalDataset(test_texts, test_labels, tokenizer, MAX_LEN)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
criterion = nn.BCEWithLogitsLoss()

def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits.squeeze(-1)
            loss = criterion(logits, labels)

            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item()

    return total_loss / len(dataloader), correct / total

test_loss, test_acc = evaluate(model, test_loader, criterion)
print("\n--- Final Test Evaluation (DistilBERT) ---")
print(f"Test Accuracy: {test_acc*100:.2f}% | Test Loss: {test_loss:.4f}")

if os.path.exists("metrics.pkl"):
    with open("metrics.pkl", "rb") as f:
        metrics = pickle.load(f)

    epochs = range(1, len(metrics["losses"]) + 1)
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, metrics["losses"], marker="o")
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(epochs, [a * 100 for a in metrics["accs"]], marker="o")
    plt.title("Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")

    plt.tight_layout()
    plt.savefig("training_metrics.png")
    print("Training performance plots saved as 'training_metrics.png'")

def predict_review(text):
    encoded = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=MAX_LEN,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**encoded)
        prob = torch.sigmoid(outputs.logits.squeeze(-1)).item()

    label = "Positive" if prob >= 0.5 else "Negative"
    return label, prob

custom_reviews = [
    "The movie was not bad at all, I really enjoyed it.",
    "An absolute masterpiece with stunning visuals and brilliant storytelling.",
    "A complete mess of a script with zero chemistry between the main characters.",
    "I cannot say I enjoyed this painfully slow piece of cinema.",
    "It starts off a bit slow, but the second half turns it into an absolute triumph."
]

print("\n--- Predictions on Custom Movie Reviews ---")
for review in custom_reviews:
    label, prob = predict_review(review)
    print(f'Review: "{review}"')
    print(f"Prediction: {label} (Positive Score: {prob:.4f})\n")

print("\n--- Live Sentiment Predictor ---")
print("Type any movie review below to test it (type 'exit' to quit):")

while True:
    user_input = input("\nEnter your review: ").strip()
    if user_input.lower() == "exit":
        break
    if not user_input:
        continue
    label, prob = predict_review(user_input)
    print(f"Prediction: {label} (Positive Score: {prob:.4f})")