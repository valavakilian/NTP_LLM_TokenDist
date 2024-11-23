import json
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import ByteLevelBPETokenizer
from tqdm import tqdm
import os
import random
from typing import List, Dict, Optional

class TinyStoriesProcessor:
    def __init__(
        self,
        data_dir: str,
        vocab_size: int = 8000,
        min_frequency: int = 2,
        special_tokens: List[str] = ["<s>", "</s>", "<pad>", "<unk>"],
        percentage: float = 100.0,  # New parameter for data percentage
        seed: int = 42  # Seed for reproducibility
    ):
        if not (0.0 < percentage <= 100.0):
            raise ValueError("Percentage must be between 0 and 100")
            
        self.data_dir = data_dir
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.special_tokens = special_tokens
        self.tokenizer = None
        self.percentage = percentage
        self.seed = seed
        random.seed(seed)
        
    def load_raw_data(self) -> List[str]:
        """Load stories from json files in the data directory."""
        all_stories = []
        json_files = glob.glob(os.path.join(self.data_dir, "data*.json"))
        
        # If using a subset, randomly sample the files first
        if self.percentage < 100.0:
            random.shuffle(json_files)
            num_files = max(1, int(len(json_files) * self.percentage / 100.0))
            json_files = json_files[:num_files]
        
        for file_path in tqdm(json_files, desc="Loading files"):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # If percentage < 100, sample stories from each file
                if self.percentage < 100.0:
                    num_stories = int(len(data) * self.percentage / 100.0)
                    data = random.sample(data, num_stories)
                stories = [item['story'] for item in data]
                all_stories.extend(stories)
        
        print(f"Loaded {len(all_stories)} stories ({self.percentage}% of dataset)")
        return all_stories
    
    def load_tokenizer(self, tokenizer_dir: str) -> None:
        """Load a pre-trained tokenizer."""
        if not os.path.exists(os.path.join(tokenizer_dir, "vocab.json")):
            raise FileNotFoundError(f"No tokenizer found in {tokenizer_dir}")
        self.tokenizer = ByteLevelBPETokenizer.from_file(
            vocab_filename=os.path.join(tokenizer_dir, "vocab.json"),
            merges_filename=os.path.join(tokenizer_dir, "merges.txt")
        )
    
    def train_tokenizer(self, output_dir: str) -> None:
        """Train a ByteLevelBPE tokenizer on the dataset."""
        self.tokenizer = ByteLevelBPETokenizer()
        
        json_files = glob.glob(os.path.join(self.data_dir, "data*.json"))
        
        # If using a subset, sample the files first
        if self.percentage < 100.0:
            random.shuffle(json_files)
            num_files = max(1, int(len(json_files) * self.percentage / 100.0))
            json_files = json_files[:num_files]
        
        temp_file = output_dir + "/temp_training_file.txt"
        with open(temp_file, 'w', encoding='utf-8') as f:
            for json_file in tqdm(json_files, desc="Preparing tokenizer training data"):
                with open(json_file, 'r', encoding='utf-8') as jf:
                    data = json.load(jf)
                    # If percentage < 100, sample stories from each file
                    if self.percentage < 100.0:
                        num_stories = int(len(data) * self.percentage / 100.0)
                        data = random.sample(data, num_stories)
                    for item in data:
                        f.write(item['story'] + '\n')
        
        self.tokenizer.train(
            files=[temp_file],
            vocab_size=self.vocab_size,
            min_frequency=self.min_frequency,
            special_tokens=self.special_tokens
        )
        
        os.makedirs(output_dir, exist_ok=True)
        self.tokenizer.save_model(output_dir)
        os.remove(temp_file)
    
    def process(self, tokenizer_dir: str, output_file: str) -> None:
        """Process the dataset, training tokenizer if needed and tokenizing if needed."""
        # Modify output filename to reflect the percentage if not using full dataset
        if self.percentage < 100.0:
            # base, ext = os.path.splitext(output_file)
            # output_file = f"{base}_{int(self.percentage)}percent{ext}"
            print(f"Using {self.percentage}% of data, output will be saved to: {output_file}")
        
        # Check if tokenized file already exists
        if os.path.exists(output_file):
            print(f"Tokenized file already exists at {output_file}")
            return
            
        # Check if tokenizer exists
        tokenizer_vocab = os.path.join(tokenizer_dir, f"vocab_V{self.vocab_size}_{self.percentage}%TS.json")
        tokenizer_merges = os.path.join(tokenizer_dir, f"merges_V{self.vocab_size}_{self.percentage}%TS.txt")
        
        if os.path.exists(tokenizer_vocab) and os.path.exists(tokenizer_merges):
            print("Loading existing tokenizer...")
            self.load_tokenizer(tokenizer_dir)
        else:
            print(f"Training new tokenizer on {self.percentage}% of data...")
            self.train_tokenizer(tokenizer_dir)
            
        print(f"Tokenizing {self.percentage}% of dataset...")
        self.tokenize_and_save(output_file)
        print("Processing complete!")
        
    def tokenize_and_save(self, output_file: str) -> None:
        """Tokenize stories and save them as separate sequences."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not trained! Call train_tokenizer first.")
        
        all_stories = self.load_raw_data()
        tokenized_stories = []
        
        for story in tqdm(all_stories, desc="Tokenizing stories"):
            encoded = self.tokenizer.encode(story)
            story_tokens = [self.tokenizer.token_to_id("<s>")] + encoded.ids + [self.tokenizer.token_to_id("</s>")]
            tokenized_stories.append(story_tokens)
        
        torch.save(tokenized_stories, output_file)


class TinyStoriesDataset(Dataset):
    def __init__(
        self,
        tokenized_file: str,
        context_size: int = 256,
        stride: int = None,  # If None, stride = context_size (no overlap)
        shuffle_chunks: bool = True,
        pad_token_id: int = 2,
        bos_token_id: int = 0,
        eos_token_id: int = 1,
    ):
        self.tokenized_stories = torch.load(tokenized_file)
        self.context_size = context_size
        self.stride = stride if stride is not None else context_size
        self.shuffle_chunks = shuffle_chunks
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        
        # Validate stride
        if self.stride <= 0:
            raise ValueError("Stride must be positive")
        if self.stride > self.context_size:
            raise ValueError("Stride cannot be larger than context_size")
        
        # Process stories into chunks
        self.chunks = self._prepare_chunks()
        
        if self.shuffle_chunks:
            self.shuffled_indices = torch.randperm(len(self.chunks))
            
        # Calculate and store dataset statistics
        self._calculate_statistics()
    
    def _prepare_chunks(self) -> List[torch.Tensor]:
        """Prepare story chunks with specified context size and stride."""
        chunks = []
        
        for story in self.tokenized_stories:
            # Skip stories that are too short (just BOS + EOS tokens)
            if len(story) <= 2:
                continue
                
            story_tensor = torch.tensor(story, dtype=torch.long)
            story_length = len(story)
            
            # For each story, generate chunks with the specified stride
            # We need context_size + 1 tokens for each chunk (input + next token as target)
            for start_idx in range(0, story_length - 1, self.stride):
                end_idx = start_idx + self.context_size + 1  # +1 for the target token
                
                if end_idx <= story_length:
                    # Full chunk
                    chunk = story_tensor[start_idx:end_idx]
                    chunks.append(chunk)
                else:
                    # Handle the last partial chunk if it's long enough
                    remaining_length = story_length - start_idx
                    if remaining_length > 1:  # Need at least 2 tokens for input/target
                        # Two options for handling the last chunk:
                        # 1. Pad it to full length
                        padded_chunk = torch.full((self.context_size + 1,), self.pad_token_id, dtype=torch.long)
                        padded_chunk[:remaining_length] = story_tensor[start_idx:]
                        chunks.append(padded_chunk)
                        
                        # 2. Or alternatively, take the last context_size + 1 tokens
                        # last_chunk = story_tensor[-self.context_size-1:]
                        # chunks.append(last_chunk)
                    break  # No more valid chunks in this story
        
        return chunks
    
    def _calculate_statistics(self):
        """Calculate various statistics about the dataset."""
        self.total_stories = len(self.tokenized_stories)
        self.total_chunks = len(self.chunks)
        self.total_tokens = sum(len(story) for story in self.tokenized_stories)
        
        # Calculate average story length
        self.avg_story_length = self.total_tokens / self.total_stories if self.total_stories > 0 else 0
        
        # Calculate effective tokens (accounting for stride overlap)
        if self.stride < self.context_size:
            effective_tokens = self.total_chunks * self.stride + (self.context_size - self.stride)
        else:
            effective_tokens = self.total_chunks * self.context_size
            
        self.effective_tokens = effective_tokens
    
    def get_statistics(self) -> Dict[str, int]:
        """Return statistics about the dataset."""
        return {
            "total_stories": self.total_stories,
            "total_chunks": self.total_chunks,
            "total_tokens": self.total_tokens,
            "effective_tokens": self.effective_tokens,
            "avg_story_length": self.avg_story_length,
            "context_size": self.context_size,
            "stride": self.stride
        }

    def __len__(self) -> int:
        return len(self.chunks)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.shuffle_chunks:
            idx = int(self.shuffled_indices[idx])
        
        chunk = self.chunks[idx]
        
        # Input is all but last token, target is all but first token
        # input_tensor = chunk[:-1]
        # target_tensor = chunk[1:]
        
        # Create attention mask (1 for real tokens, 0 for padding)
        # attention_mask = (input_tensor != self.pad_token_id).long()
        
        # return {
        #     "input_ids": input_tensor,
        #     "labels": target_tensor,
        #     # "attention_mask": attention_mask
        # }

        return chunk

def get_dataloader(
    dataset: TinyStoriesDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4
) -> DataLoader:
    """Create a DataLoader for the dataset."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )




def print_dataset_stats(stats: Dict[str, float], batch_size: Optional[int] = None) -> None:
    """
    Print dataset statistics in a nicely formatted way.
    
    Args:
        stats: Dictionary containing dataset statistics
        batch_size: Optional batch size to calculate training steps
    """
    # Define units for human-readable numbers
    def format_number(n: float) -> str:
        if n >= 1e6:
            return f"{n/1e6:.2f}M"
        elif n >= 1e3:
            return f"{n/1e3:.2f}K"
        else:
            return f"{n:.2f}"
    
    # Print header
    print("\n" + "="*50)
    print("Dataset Statistics".center(50))
    print("="*50)
    
    # Format core statistics
    print("\nData Points:")
    print(f"├── Stories: {format_number(stats['total_stories'])} ")
    print(f"├── Training Chunks: {format_number(stats['total_chunks'])} ")
    print(f"└── Total Tokens: {format_number(stats['total_tokens'])} ")
    
    print("\nConfiguration:")
    print(f"├── Context Size: {stats['context_size']} tokens")
    print(f"├── Stride: {stats['stride']} tokens")
    print(f"└── Avg Story Length: {stats['avg_story_length']:.1f} tokens")
    
    # If batch size provided, calculate training steps
    if batch_size is not None:
        steps_per_epoch = stats['total_chunks'] // batch_size
        print("\nTraining Info:")
        print(f"├── Batch Size: {batch_size}")
        print(f"├── Steps per Epoch: {format_number(steps_per_epoch)}")
        
        # Calculate some example epoch scenarios
        for num_epochs in [1, 10, 100]:
            total_steps = steps_per_epoch * num_epochs
            print(f"├── Steps for {num_epochs} epochs: {format_number(total_steps)}")
    
    print("\nMemory Estimates:")
    # Rough estimate assuming 2 bytes per token for input_ids and labels
    memory_per_batch = (stats['context_size'] * 2 * 2) / (1024 * 1024)  # in MB
    if batch_size is not None:
        print(f"└── Approximate memory per batch: {memory_per_batch * batch_size:.2f} MB")
    
    print("\n" + "="*50 + "\n")


# # First process the dataset
# # Initialize processor with percentage of data to use
# processor = TinyStoriesProcessor(
#     data_dir="/scratch/st-cthrampo-1/vaalaa/NTP_LLM_TokenDist/TinyStories/TinyStories_all_data",
#     percentage=5.0,  # Use 10% of the data
#     seed=42  # For reproducibility
# )

# # Process the data
# processor.process(
#     tokenizer_dir="/scratch/st-cthrampo-1/vaalaa/NTP_LLM_TinyStories/data_saves/tokenizers/tokenizer_output_dir",
#     output_file="/scratch/st-cthrampo-1/vaalaa/NTP_LLM_TinyStories/data_saves/tokenized_data/tokenized_stories.pt"
# )
# # Create dataset
# dataset = TinyStoriesDataset(
#     tokenized_file="/scratch/st-cthrampo-1/vaalaa/NTP_LLM_TinyStories/data_saves/tokenized_data/tokenized_stories_10percent.pt",
#     context_size=32,
#     stride=1,
#     shuffle_chunks=True
# )


# # Create dataloader
# dataloader = get_dataloader(
#     dataset,
#     batch_size=32,
#     shuffle=True,  # This will shuffle batches of chunks
#     num_workers=4  # Adjust based on your system
# )

# Training loop example
# for batch in dataloader:
    # inputs = batch["input_ids"]          # Shape: [batch_size, context_size]
    # labels = batch["labels"]             # Shape: [batch_size, context_size]
    # attention_mask = batch["attention_mask"]  # Shape: [batch_size, context_size]
    # Your training code here

    # print(inputs[0,:])
    # print(labels[0,:])
    # print(batch[0,:])
    # input()