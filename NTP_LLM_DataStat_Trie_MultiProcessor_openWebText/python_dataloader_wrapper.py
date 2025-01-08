import torch
from torch.utils.data import Dataset, DataLoader
from fast_tokenized_dataset import FastTokenizedDataset

class TokenizedDataset(Dataset):
    def __init__(self, data_path, context_length, data_percentage=100, 
                 stride=None, token_pairs=None, valid_indices=None, 
                 is_root=False, root_ctx_len=2):
        """
        Args:
            data_path: Path to the .bin file containing tokenized data
            context_length: Length of context window
            data_percentage: Percentage of data to use (1-100)
            stride: Number of tokens to skip between windows
            token_pairs: Set of tuples of tokens to filter windows by
            valid_indices: Pre-calculated valid indices for token pairs
            is_root: Whether this is a root context dataset
            root_ctx_len: Context length for root datasets
        """
        self.cpp_dataset = FastTokenizedDataset(
            data_path=data_path,
            context_length=context_length,
            data_percentage=data_percentage,
            stride=stride if stride is not None else context_length,
            is_root=is_root,
            root_ctx_len=root_ctx_len
        )
        
        self.vocab_size = self.cpp_dataset.get_vocab_size()
        print(f"Vocabulary size: {self.vocab_size:,}")

    def __len__(self):
        return len(self.cpp_dataset)

    def __getitem__(self, idx):
        return torch.from_numpy(self.cpp_dataset[idx])
    
    def analyze_window_startPairs(self, prefix_length=2):
        """
        Analyze token pairs and track their locations in the data
        Returns both counts and a dictionary mapping each pair to its locations
        """
        return self.cpp_dataset.analyze_window_startPairs(prefix_length)

def create_dataloader(data_path, context_length, batch_size, data_percentage=100,
                     stride=None, token_pairs=None, valid_indices=None, 
                     shuffle=True, is_root=False, root_ctx_len=2):
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
        is_root: Whether this is a root context dataset
        root_ctx_len: Context length for root datasets
    """
    dataset = TokenizedDataset(
        data_path=data_path,
        context_length=context_length,
        data_percentage=data_percentage,
        stride=stride,
        token_pairs=token_pairs,
        valid_indices=valid_indices,
        is_root=is_root,
        root_ctx_len=root_ctx_len
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=4
    )
    
    return dataloader, dataset.vocab_size