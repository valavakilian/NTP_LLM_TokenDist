import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool, cpu_count
from collections import Counter
import time


class TokenizedDataset(Dataset):
    def __init__(self, data_path, context_length, data_percentage=100, stride=None, token_pairs=None, valid_indices = None, is_root = False, root_ctx_len = 2):
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
        if token_pairs is None or valid_indices is None:
            self.token_pairs = False
            self.valid_indices = None
            self.n_windows = max(0, (len(self.data) - context_length - 1) // self.stride + 1)
        else:
            self.token_pairs = set(tuple(pair) for pair in token_pairs)
            # Pre-calculate valid window indices
            self.valid_indices = [pos for pos in valid_indices if pos + self.context_length + 1 <= len(self.data)]
            total_possible_windows = (len(self.data) - context_length - 1) // self.stride + 1
            
            # print("Finding windows with matching token pairs...")
            # for idx in tqdm(range(total_possible_windows)):
            #     start_idx = idx * self.stride
            #     first_tokens = tuple(self.data[start_idx:start_idx + len(next(iter(self.token_pairs)))])
            #     if first_tokens in self.token_pairs:
            #         self.valid_indices.append(idx)
            
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
    
    
    
    
    # def analyze_window_startPairs(self, prefix_length=2):
    #     """
    #     Analysis with detailed progress tracking
    #     """
    #     total_windows = (len(self.data) - prefix_length - 1) // self.stride + 1
        
    #     # Use smaller chunks for better progress tracking
    #     num_processes = min(cpu_count(), 8)
    #     chunk_size = min(1000000, total_windows // (num_processes * 8))
        
    #     # Prepare chunks
    #     chunks = []
    #     for chunk_start in range(0, total_windows, chunk_size):
    #         chunk_end = min(chunk_start + chunk_size, total_windows)
    #         chunks.append((chunk_start, chunk_end, self.data.filename, self.stride))
        
    #     print(f"Starting analysis of {total_windows:,} windows...")
    #     print(f"Using {num_processes} processes with chunk size: {chunk_size:,}")
    #     print(f"Total chunks to process: {len(chunks)}")
        
    #     final_counts = Counter()
    #     start_time = time.time()
    #     processed_tokens = 0
        
    #     with Pool(processes=num_processes) as pool:
    #         # Use tqdm for progress tracking
    #         with tqdm(total=len(chunks), desc="Processing chunks") as pbar:
    #             for chunk_counts in pool.imap(process_chunk, chunks):
    #                 final_counts.update(chunk_counts)
    #                 processed_tokens += chunk_size
    #                 pbar.update(1)
                    
    #                 # Print periodic statistics
    #                 if pbar.n % 100 == 0:
    #                     elapsed = time.time() - start_time
    #                     tokens_per_sec = processed_tokens / elapsed
    #                     print(f"\nProcessing speed: {tokens_per_sec:,.0f} tokens/sec")
    #                     print(f"Unique pairs found so far: {len(final_counts):,}")
        
    #     # Final statistics
    #     total_time = time.time() - start_time
    #     print(f"\nAnalysis complete!")
    #     print(f"Total time: {total_time:.2f} seconds")
    #     print(f"Average speed: {total_windows/total_time:,.0f} tokens/sec")
    #     print(f"Total unique pairs found: {len(final_counts):,}")
        
    #     return dict(final_counts)
    
    def analyze_window_startPairs(self, prefix_length=2):
        """
        Analyze token pairs and track their locations in the data
        Returns both counts and a dictionary mapping each pair to its locations
        """
        try:
            total_windows = (len(self.data) - prefix_length - 1) // self.stride + 1
            num_processes = min(cpu_count(), 8)
            chunk_size = min(1000000, total_windows // (num_processes * 8))
            
            chunks = []
            for chunk_start in range(0, total_windows, chunk_size):
                chunk_end = min(chunk_start + chunk_size, total_windows)
                chunks.append((chunk_start, chunk_end, self.data.filename, self.stride, prefix_length))
            
            print(f"Starting analysis of {total_windows:,} windows...")
            final_counts = Counter()
            final_locations = defaultdict(list)
            chunk_results = []
            
            # Stage 1: Process chunks
            print("Stage 1: Processing chunks...")
            with Pool(processes=num_processes) as pool:
                for chunk_counts, chunk_locs in tqdm(pool.imap(process_chunk_with_locations, chunks), 
                                                total=len(chunks), 
                                                desc="Processing chunks"):
                    chunk_results.append((chunk_counts, chunk_locs))
            
            # Stage 2: Combine results
            print("\nStage 2: Combining results...")
            for i, (chunk_dict, chunk_locs) in enumerate(tqdm(chunk_results, desc="Merging counts and locations")):
                final_counts.update(chunk_dict)
                # Merge locations
                for pair, locations in chunk_locs.items():
                    final_locations[pair].extend(locations)
                chunk_results[i] = None  # Free memory
            
            print(f"\nTotal unique pairs: {len(final_counts)}")
            print("Sample of pairs and their occurrences (first 5):")
            for i, (pair, count) in enumerate(final_counts.most_common(5)):
                num_locations = len(final_locations[pair])
                print(f"{pair}: count={count:,}, locations={num_locations:,}")
            
            return dict(final_counts), dict(final_locations)
                
        except Exception as e:
            print(f"\nERROR OCCURRED!")
            print(f"Error type: {type(e)}")
            print(f"Error message: {str(e)}")
            import traceback
            print("Traceback:")
            traceback.print_exc()
            raise


def process_chunk_with_locations(args):
    """Process a chunk and return both counts and locations of token pairs"""
    start_idx, end_idx, data_path, stride, prefix_length = args
    data = np.memmap(data_path, dtype=np.uint16, mode='r')
    
    chunk_counter = Counter()
    # Dictionary to store locations for each prefix
    chunk_locations = defaultdict(list)
    
    for idx in range(start_idx, end_idx):
        start_pos = idx * stride
        if start_pos + prefix_length <= len(data):
            prefix = tuple([data[start_pos + i] for i in range(0, prefix_length)])
            chunk_counter[prefix] += 1
            # Store the actual position in the data
            chunk_locations[prefix].append(start_pos)
    
    return dict(chunk_counter), dict(chunk_locations)


def create_dataloader(data_path, context_length, batch_size, data_percentage=100, 
                     stride=None, token_pairs=None, valid_indices=None, shuffle=True, is_root = False, root_ctx_len = 2):
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
        valid_indices,
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