import json
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import ByteLevelBPETokenizer
from tqdm import tqdm
import os
import random
import numpy as np
from typing import List, Dict, Optional, Tuple

class TinyStoriesProcessor:
    def __init__(
        self,
        data_dir: str,
        vocab_size: int = 8000,
        min_frequency: int = 2,
        special_tokens: List[str] = ["<s>", "</s>", "<pad>", "<unk>"],
        percentage: float = 100.0,
        seed: int = 42
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

        self.tokenization_percentage = 10
        
    def load_raw_data(self) -> List[str]:
        """Load stories from json files in the data directory."""
        all_stories = []
        json_files = glob.glob(os.path.join(self.data_dir, "data*.json"))
        
        if self.percentage < 100.0:
            random.shuffle(json_files)
            num_files = max(1, int(len(json_files) * self.percentage / 100.0))
            json_files = json_files[:num_files]
        
        for file_path in tqdm(json_files, desc="Loading files"):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if self.percentage < 100.0:
                    num_stories = int(len(data) * self.percentage / 100.0)
                    data = random.sample(data, num_stories)
                stories = [item['story'] for item in data]
                all_stories.extend(stories)
        
        print(f"Loaded {len(all_stories)} stories ({self.percentage}% of dataset)")
        return all_stories
    
    def load_tokenizer(self, tokenizer_dir: str) -> None:
        """Load a pre-trained tokenizer."""
        tokenizer_dir = os.path.join(tokenizer_dir, f"_V{self.vocab_size}_{self.percentage}%TS")
        os.makedirs(tokenizer_dir, exist_ok=True)

        vocab_file = os.path.join(tokenizer_dir, f"vocab.json")
        merges_file = os.path.join(tokenizer_dir, f"merges.txt")
        
        if not os.path.exists(vocab_file) or not os.path.exists(merges_file):
            raise FileNotFoundError(f"No tokenizer found in {tokenizer_dir}")
            
        self.tokenizer = ByteLevelBPETokenizer.from_file(
            vocab_filename=vocab_file,
            merges_filename=merges_file
        )
    
    def train_tokenizer(self, output_dir: str) -> None:
        """Train a ByteLevelBPE tokenizer on the dataset."""
        self.tokenizer = ByteLevelBPETokenizer()
        
        # Create temp file for tokenizer training
        temp_file = os.path.join(output_dir, "temp_training_file.txt")
        with open(temp_file, 'w', encoding='utf-8') as f:
            json_files = glob.glob(os.path.join(self.data_dir, "data*.json"))
            
            random.shuffle(json_files)
            num_files = max(1, int(len(json_files)  * (self.percentage / 100.0) * (self.tokenization_percentage / 100.0)))
            json_files = json_files[:num_files]
        
            for json_file in tqdm(json_files, desc="Preparing tokenizer training data"):
                with open(json_file, 'r', encoding='utf-8') as jf:
                    data = json.load(jf)
                    num_stories = int(len(data)  * (self.percentage / 100.0) * (self.tokenization_percentage / 100.0))
                    data = random.sample(data, num_stories)
                    for item in data:
                        f.write(item['story'] + '\n')
        
        # Train tokenizer
        self.tokenizer.train(
            files=[temp_file],
            vocab_size=self.vocab_size,
            min_frequency=self.min_frequency,
            special_tokens=self.special_tokens
        )
        
        # Save with percentage in filename
        output_dir = os.path.join(output_dir, f"_V{self.vocab_size}_{self.percentage}%TS")
        os.makedirs(output_dir, exist_ok=True)
        self.tokenizer.save_model(output_dir)
        
        os.remove(temp_file)
    
    def _count_max_length(self, sample_size: int = 1000) -> int:
        """Estimate max length from a sample of stories."""
        json_files = glob.glob(os.path.join(self.data_dir, "data*.json"))
        max_length = 0
        stories_sampled = 0
        
        # Look through ALL files to find true max length
        for file_path in tqdm(json_files, desc="Finding max length"):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    encoded = self.tokenizer.encode(item['story'])
                    max_length = max(max_length, len(encoded.ids) + 2)  # +2 for special tokens
        
        print(f"Found max length: {max_length}")
        return max_length  # No need for padding since this is true max
    
    def process(self, tokenizer_dir: str, output_dir: str) -> None:
        """Process the dataset, training tokenizer if needed and tokenizing if needed."""
        os.makedirs(output_dir, exist_ok=True)
        
        output_dir = os.path.join(output_dir, f"_V{self.vocab_size}_{self.percentage}%TS")
        os.makedirs(output_dir, exist_ok=True)

        # Define output files
        stories_file = os.path.join(output_dir, f"stories.mmap")
        lengths_file = os.path.join(output_dir, f"lengths.mmap")
        metadata_file = os.path.join(output_dir, f"metadata.json")

        # Format tokenizer directory with vocab size and percentage
        tokenizer_dir = os.path.join(tokenizer_dir, f"_V{self.vocab_size}_{self.percentage}%TS")
        os.makedirs(tokenizer_dir, exist_ok=True)

        # Check if processed files exist
        if all(os.path.exists(f) for f in [stories_file, lengths_file, metadata_file]):
            print(f"Processed files already exist in {output_dir}")
            return
            
        # Check/train tokenizer
        try:
            self.load_tokenizer(tokenizer_dir)
            print("Loaded existing tokenizer")
        except FileNotFoundError:
            print(f"Training new tokenizer on {self.percentage}% of data...")
            self.train_tokenizer(tokenizer_dir)
        
        # Process the dataset
        print(f"Processing {self.percentage}% of dataset to memory mapped files...")
        
        # First pass: count stories and get max length
        max_length = self._count_max_length()
        all_stories = self.load_raw_data()
        
        # Initialize memory mapped files
        stories_mmap = np.memmap(stories_file, dtype=np.int32, mode='w+',
                            shape=(len(all_stories), max_length))
        lengths_mmap = np.memmap(lengths_file, dtype=np.int32, mode='w+',
                            shape=(len(all_stories),))
        
        # Process stories
        for idx, story in enumerate(tqdm(all_stories, desc="Tokenizing stories")):
            encoded = self.tokenizer.encode(story)
            tokens = [self.tokenizer.token_to_id("<s>")] + encoded.ids + [self.tokenizer.token_to_id("</s>")]
            length = len(tokens)
            
            if length > max_length:
                raise ValueError(f"Found story of length {length} > max_length {max_length}. This should not happen!")
                
            stories_mmap[idx, :length] = tokens
            stories_mmap[idx, length:] = self.tokenizer.token_to_id("<pad>")
            lengths_mmap[idx] = length
        
        # Save metadata
        metadata = {
            "num_stories": len(all_stories),
            "max_length": max_length,
            "percentage": self.percentage,
            "vocab_size": self.vocab_size,
            "pad_token_id": self.tokenizer.token_to_id("<pad>"),
            "bos_token_id": self.tokenizer.token_to_id("<s>"),
            "eos_token_id": self.tokenizer.token_to_id("</s>")
        }
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f)
        
        # Flush and close memmaps
        stories_mmap.flush()
        lengths_mmap.flush()
        del stories_mmap
        del lengths_mmap
        
        print("Processing complete!")

class TinyStoriesDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        vocab_size: int,
        token_pairs: List[Tuple[int, int]],  # New parameter for token pairs,
        is_root: bool = False,
        root_ctx_len: int = 2,
        context_size: int = 256,
        stride: int = None,
        shuffle_chunks: bool = True,
        percentage: float = 100.0
    ):
        self.vocab_size = vocab_size
        self.percentage = percentage
        self.token_pairs = token_pairs  # Store the token pairs to filter for
        self.is_root = is_root
        self.root_ctx_len = root_ctx_len

        data_dir = os.path.join(data_dir, f"_V{self.vocab_size}_{self.percentage}%TS")
        os.makedirs(data_dir, exist_ok=True)
        
        # Load metadata
        metadata_file = os.path.join(data_dir, f"metadata.json")
        with open(metadata_file, 'r') as f:
            self.metadata = json.load(f)
        
        # Open memory mapped files
        self.stories = np.memmap(
            os.path.join(data_dir, f"stories.mmap"),
            dtype=np.int32, mode='r',
            shape=(self.metadata['num_stories'], self.metadata['max_length'])
        )
        self.lengths = np.memmap(
            os.path.join(data_dir, f"lengths.mmap"),
            dtype=np.int32, mode='r',
            shape=(self.metadata['num_stories'],)
        )
        
        self.context_size = context_size
        self.stride = stride if stride is not None else context_size
        self.shuffle_chunks = shuffle_chunks
        
        print("Preparing chunk locations and filtering for token pairs...")
        self.chunk_locations = self._prepare_chunk_locations()
        if self.shuffle_chunks:
            self.shuffled_indices = torch.randperm(len(self.chunk_locations))
        print(f"Dataset has {len(self.chunk_locations)} contexts starting with specified token pairs")
    
    def _check_token_pair(self, story_idx: int, start_idx: int) -> bool:
        """Check if the tokens at the given position match any of the specified pairs."""
        if start_idx + 1 >= self.lengths[story_idx]:  # Ensure we have at least 2 tokens
            return False
        
        current_pair = (
            int(self.stories[story_idx, start_idx]),
            int(self.stories[story_idx, start_idx + 1])
        )
        
        return current_pair in self.token_pairs
    
    def _prepare_chunk_locations(self) -> List[Tuple[int, int, int]]:
        """Prepare list of (story_idx, start_idx, length) for each chunk that starts with specified token pairs."""
        locations = []
        chunk_size = self.context_size + 1
        
        for story_idx in range(len(self.lengths)):
            story_length = self.lengths[story_idx]
            
            if story_length <= 2:  # Skip too short stories
                continue
                
            # Only process complete chunks
            num_complete_chunks = (story_length - chunk_size) // self.stride + 1
            
            for i in range(num_complete_chunks):
                start_idx = i * self.stride
                end_idx = start_idx + chunk_size
                
                # Only include chunks that start with one of the specified token pairs
                if end_idx <= story_length :
                    if len(self.token_pairs) == 0:
                        locations.append((story_idx, start_idx, end_idx))
                    elif self._check_token_pair(story_idx, start_idx):
                        locations.append((story_idx, start_idx, end_idx))
    
        return locations
    
    def __len__(self) -> int:
        return len(self.chunk_locations)
    
    def __getitem__(self, idx: int) -> np.ndarray:
        if self.shuffle_chunks:
            idx = int(self.shuffled_indices[idx])
        
        story_idx, start_idx, end_idx = self.chunk_locations[idx]
        if self.is_root:
            chunk = self.stories[story_idx, start_idx:end_idx][0:self.root_ctx_len]
        else:
            chunk = self.stories[story_idx, start_idx:end_idx][self.root_ctx_len:]

        return chunk

def collate_fn(batch: List[np.ndarray]) -> torch.Tensor:
    """Convert list of numpy arrays to a single torch tensor."""
    return torch.from_numpy(np.array(batch)).long()
    
def get_dataloader(
    dataset: TinyStoriesDataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4
    ) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )



# Utility function to print token pair statistics
def print_token_pair_stats(dataset: TinyStoriesDataset) -> None:
    print("\n" + "="*50)
    print("Token Pair Statistics".center(50))
    print("="*50)
    
    print(f"\nNumber of token pairs specified: {len(dataset.token_pairs)}")
    print(f"Total matching contexts found: {len(dataset):,}")
    print(f"Token pairs being filtered for: {dataset.token_pairs}")
    
    if len(dataset) == 0:
        print("\nWARNING: No contexts found matching the specified token pairs!")
    
    print("\n" + "="*50)