from datasets import load_from_disk
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
import torch
from torch.utils.data import Dataset, DataLoader
import os
import math

# Step 1: Load the Wikitext-2 Dataset
def load_and_tokenize_wikitext(dataset_dir, vocab_size, context_length):
    # Load the dataset saved earlier
    dataset = load_from_disk(dataset_dir)
    
    # Combine all text from the train, validation, and test sets
    text = " ".join(dataset['train']['text']) + " " + \
           " ".join(dataset['validation']['text']) + " " + \
           " ".join(dataset['test']['text'])

    # Initialize the tokenizer
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    # Train the tokenizer
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])
    tokenizer.train_from_iterator([text], trainer)

    # Tokenize the entire text
    tokens = tokenizer.encode(text).ids

    return tokens, tokenizer

# Step 2: Create a Dataset Class for Shifted Context Window
class WikitextShiftedDataset(Dataset):
    def __init__(self, tokenized_data, context_length):
        self.data = tokenized_data
        self.context_length = context_length
    
    def __len__(self):
        # Subtract context_length and 1 (for the target)
        return len(self.data) - self.context_length
    
    def __getitem__(self, idx):
        # Context window is from idx to idx + context_length
        x = self.data[idx : idx + self.context_length]
        # Target is the token immediately after the context window
        y = self.data[idx + self.context_length]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

# Step 3: Create the DataLoader
def create_dataloader(tokenized_data, context_length, batch_size):
    dataset = WikitextShiftedDataset(tokenized_data, context_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader


def generate_equal_spaced_points(num_examples, num_points):
    # Ensure at least two points to avoid errors
    if num_points < 2:
        raise ValueError("num_points must be at least 2")
    
    # Calculate the spacing
    step = num_examples / (num_points - 1)
    
    # Generate the points
    points = [int(round(step * i)) for i in range(num_points)]
    
    # Ensure num_examples is included and the list is unique
    if points[-1] != num_examples - 1:
        points[-1] = num_examples - 1
    
    # Remove duplicates and sort the list
    points = sorted(set(points))
    
    return points

# Example usage
dataset_dir = '/scratch/st-cthrampo-1/vaalaa/NTP_LLM_TokenDist/WikiText'  # Your saved dataset folder
vocab_size = 10000
context_length = 16
batch_size = 1000



# Step 4: Load and Tokenize the Wikitext-2 Dataset
tokenized_data, tokenizer = load_and_tokenize_wikitext(dataset_dir, vocab_size, context_length)

# Step 5: Create the DataLoader
dataloader = create_dataloader(tokenized_data, context_length, batch_size)

num_examples = len(tokenized_data) - context_length
num_batches = math.ceil(num_examples / batch_size)
print("=" * 100)
print("num_examples: " + str(num_examples))
print("num_batches: " + str(num_batches))
print("=" * 100)

milestones = generate_equal_spaced_points(num_examples, 100)
print(milestones)
input()

# Step 6: Iterate through the DataLoader and print samples
for batch in dataloader:
    x_batch, y_batch = batch
    print(f"Context: {x_batch}, Target: {y_batch}")
    input()
