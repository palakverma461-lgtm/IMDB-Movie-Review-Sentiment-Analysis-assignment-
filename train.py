import os
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from datasets import load_dataset

MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 128
BATCH_SIZE = 16
N_EPOCHS = 5
LEARNING_RATE = 2e-5
MAX_TRAIN_SAMPLES = 4000
MAX_TEST_SAMPLES = 1000
SAVE_DIR = "bert_sentiment_model"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE} | Model: {MODEL_NAME}", flush=True)

print("Loading IMDB dataset via Hugging Face...", flush=True)
imdb_dataset = load_dataset("stanfordnlp/imdb")

train_texts = list(imdb_dataset["train"]["text"])
train_labels = list(imdb_dataset["train"]["label"])

test_texts = list(imdb_dataset["test"]["text"])
test_labels = list(imdb_dataset["test"]["label"])

if MAX_TRAIN_SAMPLES is not None:
    train_texts = train_texts[:MAX_TRAIN_SAMPLES]
    train_labels = train_labels[:MAX_TRAIN_SAMPLES]

if MAX_TEST_SAMPLES is not None:
    test_texts = test_texts[:MAX_TEST_SAMPLES]
    test_labels = test_labels[:MAX_TEST_SAMPLES]

synthetic_samples = [
    ("The movie was not bad at all, I really enjoyed it.", 1),
    ("Not bad at all, actually quite entertaining and well acted.", 1),
    ("This film is not disappointing in any way, truly great.", 1),
    ("There was not a single boring moment in this entire movie.", 1),
    ("I did not expect it to be this good, what a pleasant surprise.", 1),
    ("Not terrible at all, in fact it was a wonderful experience.", 1),
    ("The plot was not predictable, it kept me hooked till the end.", 1),
    ("Never dull and never boring, a fantastic watch.", 1),
    ("No complaints whatsoever, brilliant acting and great story.", 1),
    ("It starts off a bit slow, but the second half turns it into an absolute triumph.", 1),
    ("The beginning was slow, but the climax made it completely worth watching.", 1),
    ("Not the most original story, yet the performances make it unforgettable.", 1),
    ("The movie was not good at all, completely disappointing and dull.", 0),
    ("Not good, very boring and a total waste of time.", 0),
    ("I did not enjoy this movie, the acting was awful.", 0),
    ("Not worth watching, terrible script and weak direction.", 0),
    ("This film is not entertaining, I fell asleep halfway.", 0),
    ("Great cast, but the awful script ruined the entire movie.", 0),
]

for text, label in synthetic_samples * 10:
    train_texts.append(text)
    train_labels.append(label)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class IMDBTransformerDataset(Dataset):
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

train_dataset = IMDBTransformerDataset(train_texts, train_labels, tokenizer, MAX_LEN)
test_dataset = IMDBTransformerDataset(test_texts, test_labels, tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1).to(DEVICE)
criterion = nn.BCEWithLogitsLoss()
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

total_steps = len(train_loader) * N_EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)

def train_epoch(model, dataloader, optimizer, scheduler, criterion):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits.squeeze(-1)
        loss = criterion(logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        total_loss += loss.item()

    return total_loss / len(dataloader), correct / total

print(f"\nStarting Transformer Fine-Tuning ({N_EPOCHS} Epochs, {len(train_dataset)} samples)...", flush=True)
train_losses, train_accs = [], []

for epoch in range(N_EPOCHS):
    loss, acc = train_epoch(model, train_loader, optimizer, scheduler, criterion)
    train_losses.append(loss)
    train_accs.append(acc)
    print(f"Epoch {epoch+1:02}/{N_EPOCHS:02} | Loss: {loss:.4f} | Accuracy: {acc*100:.2f}%", flush=True)

os.makedirs(SAVE_DIR, exist_ok=True)
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

with open("metrics.pkl", "wb") as f:
    pickle.dump({"losses": train_losses, "accs": train_accs, "model_name": MODEL_NAME}, f)

with open("test_data.pkl", "wb") as f:
    pickle.dump((test_texts, test_labels), f)

print(f"\nTraining complete! Transformer model saved in '{SAVE_DIR}/'.", flush=True)