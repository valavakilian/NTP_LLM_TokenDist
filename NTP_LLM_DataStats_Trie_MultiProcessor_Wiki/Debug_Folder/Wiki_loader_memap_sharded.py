from datasets import load_from_disk
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
import torch
from torch.utils.data import Dataset, DataLoader
import os
import pickle
import json
from collections import defaultdict
from tqdm import tqdm


def load_and_tokenize_wikitext(dataset_dir, vocab_size, context_length, tokenizer_path=None, 
                              tokenized_data_path=None, data_percentage=100, force_retokenize=False):
    """
    Load and tokenize WikiText dataset with caching capabilities and data percentage selection.
    
    Args:
        dataset_dir: Directory containing the WikiText dataset
        vocab_size: Size of the vocabulary for the tokenizer
        context_length: Length of context window
        tokenizer_path: Path to save/load tokenizer
        tokenized_data_path: Path to save/load tokenized data
        data_percentage: Percentage of data to use (1-100)
        force_retokenize: Force recreation of tokenized data
    """
    dataset = load_from_disk(dataset_dir)

    file_name = f"_V{vocab_size}_{data_percentage}%WikiText/"

    tokenizer_path = tokenizer_path + file_name
    tokenized_data_path = tokenized_data_path + file_name
    
    # Load or create tokenizer
    if tokenizer_path and os.path.exists(tokenizer_path + "tokenizer.json") and not force_retokenize:
        print(f"Loading existing tokenizer from {tokenizer_path}")
        tokenizer = Tokenizer.from_file(tokenizer_path + "tokenizer.json")
    else:
        print("Creating and training new tokenizer...")
        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size, 
            special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        )
        
        # Merge text data and apply percentage
        merged_text = dataset['train']['text'] + dataset['validation']['text'] + dataset['test']['text']
        total_length = len(merged_text)
        use_length = int(total_length * (data_percentage / 100))
        merged_text = merged_text[:use_length]
        
        # Train tokenizer with batches
        def batch_iterator(dataset, batch_size=300000):
            for i in range(0, len(dataset), batch_size):
                yield dataset[i:i + batch_size]
        
        for batch in batch_iterator(merged_text):
            tokenizer.train_from_iterator(batch, trainer)
        

        os.makedirs(tokenizer_path, exist_ok=True)
        # Save tokenizer if path provided
        if tokenizer_path:
            print(f"Saving tokenizer to {tokenizer_path}")
            tokenizer.save(tokenizer_path + "tokenizer.json")
    
    # Load or create tokenized data
    if tokenized_data_path and os.path.exists(tokenized_data_path + "data.pkl") and not force_retokenize:
        print(f"Loading existing tokenized data from {tokenized_data_path}")
        with open(tokenized_data_path + "data.pkl", 'rb') as f:
            tokenized_data = pickle.load(f)
    else:
        print("Tokenizing data...")
        # Apply percentage to dataset before tokenizing
        merged_text = dataset['train']['text'] + dataset['validation']['text'] + dataset['test']['text']
        total_length = len(merged_text)
        use_length = int(total_length * (data_percentage / 100))
        merged_text = merged_text[:use_length]
        
        def tokenize_batch(texts):
            tokenized_data = []
            print("Tokenizing text in batches...")
            for i in tqdm(range(0, len(texts), 300000), desc="Processing batches"):
                batch = texts[i:i + 300000]
                for text in tqdm(batch, desc="Tokenizing texts", leave=False):
                    tokenized_data.extend(tokenizer.encode(text).ids)
            return tokenized_data
        
        tokenized_data = tokenize_batch(merged_text)
        
        # Save tokenized data if path provided
        os.makedirs(tokenized_data_path, exist_ok=True)
        if tokenized_data_path:
            print(f"Saving tokenized data to {tokenized_data_path}")
            with open(tokenized_data_path + "data.pkl", 'wb') as f:
                pickle.dump(tokenized_data, f)
    
    return tokenized_data, tokenizer


class WikitextShiftedDataset(Dataset):
    def __init__(self, tokenized_data, context_length, stride=None, token_pairs=None, is_root = False, root_ctx_len = 2):
        """
        Args:
            tokenized_data: List of tokenized integers
            context_length: Length of context window
            stride: Number of tokens to skip between windows. If None, defaults to context_length
            token_pairs: Set of tuples of (token1, token2) to filter windows. If None, use all windows
        """
        self.data = tokenized_data
        self.context_length = context_length
        self.stride = stride if stride is not None else context_length
        if token_pairs is None:
            self.token_pairs = False
        else:
            self.token_pairs = set(tuple(pair) for pair in token_pairs)
        self.is_root = is_root
        self.root_ctx_len = root_ctx_len
        
        # Pre-calculate valid window indices if token_pairs is provided
        if self.token_pairs:
            self.valid_indices = []
            total_possible_windows = (len(self.data) - context_length - 1) // self.stride + 1
            
            print("Finding windows with matching token pairs...")
            for idx in tqdm(range(total_possible_windows)):
                start_idx = idx * self.stride
                first_two_tokens = tuple(self.data[start_idx:start_idx + 2])
                if first_two_tokens in self.token_pairs:
                    self.valid_indices.append(idx)
            
            self.n_windows = len(self.valid_indices)
            print(f"Found {self.n_windows} windows matching the token pairs")
        else:
            self.valid_indices = None
            self.n_windows = max(0, (len(self.data) - context_length - 1) // self.stride + 1)
    
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
        
        return torch.tensor(x, dtype=torch.long)
    
    def analyze_window_startPairs(self):
        """
        Analyzes the first two tokens of each window in the dataset.
        
        Returns:
            dict: A dictionary where keys are tuples of (token1, token2) and 
                 values are the count of how many times this pair appears
        """
        token_pair_counts = defaultdict(int)
        
        print("Analyzing window starts...")
        total_windows = len(self)
        for idx in tqdm(range(total_windows)):
            if self.valid_indices is not None:
                window_idx = self.valid_indices[idx]
                start_idx = window_idx * self.stride
            else:
                start_idx = idx * self.stride
                
            first_two_tokens = tuple(self.data[start_idx:start_idx + 2])
            token_pair_counts[first_two_tokens] += 1
            
        return dict(token_pair_counts)


def create_dataloader(tokenized_data, context_length, batch_size, stride=None, token_pairs=None, is_root = False, root_ctx_len = 2):
    """
    Create a DataLoader with optional stride and token pair filtering.
    
    Args:
        tokenized_data: List of tokenized integers
        context_length: Length of context window
        batch_size: Batch size for DataLoader
        stride: Number of tokens to skip between windows. If None, defaults to context_length
        token_pairs: Set of tuples of (token1, token2) to filter windows. If None, use all windows
    """
    dataset = WikitextShiftedDataset(tokenized_data, context_length, stride, token_pairs, is_root = is_root, root_ctx_len = root_ctx_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader



def analyze_dataloader_windows_startPairs(dataloader):
    """
    Analyzes all windows in a dataloader to count occurrences of starting token pairs.
    """
    return dataloader.dataset.analyze_window_startPairs()