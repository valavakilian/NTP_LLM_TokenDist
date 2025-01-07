# fast_token_analysis.py
import numpy as np
from token_analysis import TokenAnalyzer
from typing import Dict, List, Tuple
import os
import gc

class FastTokenAnalysis:
    def __init__(self, data_path: str, context_length: int, stride: int = 1, num_threads: int = 8):
        """
        Initialize the fast token analysis.
        
        Args:
            data_path: Path to the memory-mapped data file
            context_length: Length of context window
            stride: Stride length for window analysis
            num_threads: Number of threads to use for parallel processing
        """
        self.data_path = data_path
        self.stride = stride
        self.context_length = context_length
        self.analyzer = TokenAnalyzer(self.context_length)
        self.num_threads = num_threads
        
        # Get data size without loading entire file
        self.data_size = os.path.getsize(data_path) // 2  # uint16 = 2 bytes
        print(f"Data size: {self.data_size:,} tokens")
        
    def analyze_and_distribute(self, num_bins: int, data_percentage: float = 100.0) -> Tuple[List[List[Tuple[int, int]]], Dict]:
        """
        Analyze token pairs and distribute them into bins.
        
        Args:
            num_bins: Number of bins to distribute token pairs into
            data_percentage: Percentage of data to use (1-100)
            
        Returns:
            Tuple containing:
            - List of bins, where each bin contains token pairs
            - Dictionary mapping token pairs to their locations
        """
        print("Loading memory-mapped data...")
        data = np.memmap(self.data_path, dtype=np.uint16, mode='r')
        
        print("Starting pair analysis...")
        try:
            counts, locations = self.analyzer.analyze_pairs(data, self.stride, self.num_threads, data_percentage)
            
            print("Distributing pairs to bins...")
            bins = self.analyzer.distribute_to_bins(counts, num_bins)
            
            return bins, locations
            
        finally:
            # Clean up memmap
            del data
            gc.collect()
    
    def save_bin_data(self, output_dir: str, bins: List[List[Tuple[int, int]]], 
                      locations: Dict[Tuple[int, int], List[int]], batch_size: int = 1000000):
        """Save bin data in batches to manage memory"""
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Saving {len(bins)} bins...")
        for bin_idx, bin_pairs in enumerate(bins):
            bin_dir = os.path.join(output_dir, f"group{bin_idx}")
            os.makedirs(bin_dir, exist_ok=True)
            
            # Save indices
            np.save(os.path.join(bin_dir, 'indices.npy'), bin_pairs)
            
            # Process locations in batches
            all_locations = []
            for i in range(0, len(bin_pairs), batch_size):
                batch_pairs = bin_pairs[i:i + batch_size]
                batch_locations = []
                for pair in batch_pairs:
                    batch_locations.extend(locations[pair])
                all_locations.extend(batch_locations)
                
                # Free memory
                del batch_locations
                gc.collect()
            
            # Save shuffled locations
            shuffled_locations = np.random.permutation(all_locations)
            np.save(os.path.join(bin_dir, 'shuffled_indices_locations.npy'), 
                   shuffled_locations)
            
            # Free memory
            del all_locations
            del shuffled_locations
            gc.collect()
            
            print(f"Saved bin {bin_idx}/{len(bins)}")

# # Example usage:
# if __name__ == "__main__":
#     import argparse
    
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--data_path", required=True)
#     parser.add_argument("--output_dir", required=True)
#     parser.add_argument("--num_bins", type=int, default=100)
#     parser.add_argument("--stride", type=int, default=1)
#     parser.add_argument("--context_length", type=int, default=32)
#     parser.add_argument("--data_percentage", type=float, default=100.0)
#     parser.add_argument("--num_threads", type=int, default=8)
#     args = parser.parse_args()
    
#     analyzer = FastTokenAnalysis(
#         data_path=args.data_path,
#         context_length=args.context_length,
#         stride=args.stride,
#         num_threads=args.num_threads
#     )
    
#     bins, locations = analyzer.analyze_and_distribute(args.num_bins, args.data_percentage)
#     analyzer.save_bin_data(args.output_dir, bins, locations)