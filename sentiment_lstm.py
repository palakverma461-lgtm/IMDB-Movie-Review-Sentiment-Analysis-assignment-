import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from torch.nn.utils.rnn import pad_sequence

# Configuration
MAX_VOCAB_SIZE = 10000
MAX_LEN = 250
BATCH_SIZE = 64

# 1. Load IMDB Dataset (Using torchtext or fallback sample data)
try:
    from torchtext.datasets import IMDB
    print("Loading IMDB dataset via torchtext...")
    train_iter = IMDB(split='train')
    test_iter = IMDB(split='test')
    
    train_data = list(train_iter)
    test_data = list(test_iter)
    
    train_texts = [item[1] for item in train_data]
    train_labels = [1 if item[0] == 'pos' else 0 for item in train_data]
    
    test_texts = [item[1] for item in test_data]
    test_labels = [1 if item[0] == 'pos' else 0 for item in test_data]
except Exception as e:
    print(f"Notice: torchtext download skipped ({e}). Using sample data structure.")
    train_texts = [
        "This movie was an absolute masterpiece. The acting and cinematography were breathtaking.",
        "Terrible waste of time. The plot made no sense and the acting was atrocious.",
        "Brilliant direction and phenomenal performances. Loved every single minute.",
        "Not good at all. I expected much more from this director, total disappointment."
    ] * 200
    train_labels = [1, 0, 1, 0] * 200
    test_texts = ["A stunning film with brilliant visual storytelling.", "Worst movie I have seen this year."] * 50
    test_labels = [1, 0] * 50

# 2. Tokenizer and Vocabulary Setup
tokenizer = lambda x: x.lower().split()

print("Building vocabulary...")
counter = Counter()
for text in train_texts:
    counter.update(tokenizer(text))

# Reserve index 0 for <PAD> and index 1 for <UNK>
vocab = {word: idx + 2 for idx, (word, freq) in enumerate(counter.most_common(MAX_VOCAB_SIZE - 2))}
vocab['<PAD>'] = 0
vocab['<UNK>'] = 1

def text_to_tensor(text):
    tokens = tokenizer(text)
    return [vocab.get(token, vocab['<UNK>']) for token in tokens]

# 3. Custom PyTorch Dataset Class
class IMDBDataset(Dataset):
    def __init__(self, texts, labels):
        self.data = []
        for text, label in zip(texts, labels):
            seq = text_to_tensor(text)[:MAX_LEN]  # Truncate if longer than MAX_LEN
            self.data.append((
                torch.tensor(seq, dtype=torch.long), 
                torch.tensor(label, dtype=torch.float)
            ))
            
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx]

# 4. Collate function for Padding variable-length sequences in batches
def collate_batch(batch):
    text_list, label_list = zip(*batch)
    # Pad sequences to the max length in this specific batch using vocab['<PAD>'] (0)
    padded_texts = pad_sequence(text_list, batch_first=True, padding_value=vocab['<PAD>'])
    labels = torch.stack(label_list)
    return padded_texts, labels

# Initialize Datasets and DataLoaders
train_dataset = IMDBDataset(train_texts, train_labels)
test_dataset = IMDBDataset(test_texts, test_labels)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)

# Verification check
print(f"\nPhase 1 Complete!")
print(f"Total Training Samples: {len(train_dataset)}")
print(f"Total Test Samples: {len(test_dataset)}")
print(f"Vocabulary Size: {len(vocab)}")

# Verify batch shapes
sample_texts, sample_labels = next(iter(train_loader))
print(f"Batch text tensor shape: {sample_texts.shape}") # [Batch Size, Sequence Length]
print(f"Batch labels tensor shape: {sample_labels.shape}")