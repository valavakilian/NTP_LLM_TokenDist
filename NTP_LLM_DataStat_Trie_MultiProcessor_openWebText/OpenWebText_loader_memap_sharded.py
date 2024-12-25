import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from collections import defaultdict



class TokenizedDataset(Dataset):
    def __init__(self, data_path, context_length, data_percentage=100, stride=None, token_pairs=None, is_root = False, root_ctx_len = 2):
        """
        Args:
            data_path: Path to the .bin file containing tokenized data
            context_length: Length of context window
            data_percentage: Percentage of data to use (1-100)
            stride: Number of tokens to skip between windows. If None, defaults to context_length
            token_pairs: Set of tuples of tokens to filter windows by. If None, use all windows
        """
        # Load the binary data
        self.data = np.memmap(data_path, dtype=np.uint16, mode='r')
        
        # Calculate how much data to use based on percentage
        total_tokens = len(self.data)
        use_tokens = int(total_tokens * (data_percentage / 100))
        self.data = self.data[:use_tokens]
        
        self.context_length = context_length
        self.stride = stride if stride is not None else context_length

        self.is_root = is_root
        self.root_ctx_len = root_ctx_len
        
        # Handle token pairs filtering
        if token_pairs is None:
            self.token_pairs = False
            self.valid_indices = None
            self.n_windows = max(0, (len(self.data) - context_length - 1) // self.stride + 1)
        else:
            self.token_pairs = set(tuple(pair) for pair in token_pairs)
            # Pre-calculate valid window indices
            self.valid_indices = []
            total_possible_windows = (len(self.data) - context_length - 1) // self.stride + 1
            
            print("Finding windows with matching token pairs...")
            for idx in tqdm(range(total_possible_windows)):
                start_idx = idx * self.stride
                first_tokens = tuple(self.data[start_idx:start_idx + len(next(iter(self.token_pairs)))])
                if first_tokens in self.token_pairs:
                    self.valid_indices.append(idx)
            
            self.n_windows = len(self.valid_indices)
            print(f"Found {self.n_windows:,} windows matching the token pairs")
        
        print(f"Loaded {use_tokens:,} tokens out of {total_tokens:,} total tokens")
        print(f"Total available windows: {self.n_windows:,}")
        
        # Get vocab size
        self.vocab_size = int(np.max(self.data)) + 1
        print(f"Vocabulary size: {self.vocab_size:,}")

    def __len__(self):
        return self.n_windows

    def __getitem__(self, idx):
        if self.valid_indices is not None:
            # If we're filtering by token pairs, use the pre-calculated indices
            window_idx = self.valid_indices[idx]
            start_idx = window_idx * self.stride
        else:
            # Normal operation without filtering
            start_idx = idx * self.stride
        
        if self.is_root:
            x = self.data[start_idx : start_idx + self.root_ctx_len + 1]
        else:
            x = self.data[start_idx : start_idx + self.context_length + 1]

        return torch.from_numpy(np.array(x, dtype=np.int64))
    
    def analyze_window_startPairs(self, prefix_length=2):
        """
        Analyzes the first N tokens of each window in the dataset.
        
        Args:
            prefix_length: Number of tokens to analyze at start of each window
            
        Returns:
            dict: A dictionary where keys are tuples of tokens and 
                 values are the count of how many times this sequence appears
        """
        token_sequence_counts = defaultdict(int)
        
        print("Analyzing window starts...")
        total_windows = len(self)
        for idx in tqdm(range(total_windows)):
            if self.valid_indices is not None:
                window_idx = self.valid_indices[idx]
                start_idx = window_idx * self.stride
            else:
                start_idx = idx * self.stride
                
            first_tokens = tuple(self.data[start_idx:start_idx + prefix_length])
            token_sequence_counts[first_tokens] += 1
            
        return dict(token_sequence_counts)

def create_dataloader(data_path, context_length, batch_size, data_percentage=100, 
                     stride=None, token_pairs=None, shuffle=True, is_root = False, root_ctx_len = 2):
    """
    Create a DataLoader for pre-tokenized data with optional token filtering.
    
    Args:
        data_path: Path to the .bin file
        context_length: Length of context window
        batch_size: Batch size for DataLoader
        data_percentage: Percentage of data to use (1-100)
        stride: Number of tokens to skip between windows
        token_pairs: Set of tuples of tokens to filter windows by
        shuffle: Whether to shuffle the data
    """
    dataset = TokenizedDataset(
        data_path, 
        context_length, 
        data_percentage,
        stride,
        token_pairs,
        is_root,
        root_ctx_len
    )
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        pin_memory=True,
        num_workers=4
    )
    return dataloader, dataset.vocab_size

# Example usage:
# train_loader, vocab_size = create_dataloader('train.bin', context_length=1024, batch_size=32, data_percentage=10)

# Load just 10% of training data
# train_loader, vocab_size = create_dataloader(
#     '/arc/project/st-cthrampo-1/vala/openwebtext_karpathy/nanoGPT/data/openwebtext/train.bin',
#     context_length=1024,
#     batch_size=32,
#     data_percentage=10
# )

# # Full validation set
# val_loader, _ = create_dataloader(
#     '/arc/project/st-cthrampo-1/vala/openwebtext_karpathy/nanoGPT/data/openwebtext/val.bin',
#     context_length=1024,
#     batch_size=32,
#     data_percentage=100
# )

# # Example training loop
# for batch in train_loader:
#     inputs = batch[:, :-1]  # all tokens except last
#     targets = batch[:, 1:]  # all tokens except first
#     # Your training code here...

#     print(inputs.shape)
#     print(targets.shape)
#     input()