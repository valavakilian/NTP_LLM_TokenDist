"""
Download, preprocess and serve the WikiText dataset as a DataLoader.
"""
import argparse
import matplotlib.pylab as plt
import numpy as np
import os
from torch.utils.data import DataLoader
from Trie_dataloader import TokenizedDataset, create_dataloader
import torch 

os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("Importing Done")
# -----------------------------------------------------------------------------


def iterate_context_windows(dataset, batch_size=32):
    """
    Iterator that yields batches of context windows from the dataset.
    
    Args:
        dataset: TokenizedDataset instance
        batch_size: Number of context windows to yield at a time
        
    Yields:
        torch.Tensor: Batch of context windows of shape (batch_size, context_length)
    """
    num_windows = dataset.getNumWindows()
    
    for start_idx in range(0, num_windows, batch_size):
        # Get the end index for this batch
        end_idx = min(start_idx + batch_size, num_windows)
        current_batch_size = end_idx - start_idx
        
        # Create batch
        batch = torch.stack([dataset[i] for i in range(start_idx, end_idx)])
        
        # Slice off the last token (target) and keep only the context
        contexts = batch[:, :-1]
        
        yield contexts


def build_pairs_counts(dataset, batch_size=32):
    # Initialize counters using nested defaultdict
    pairs_counts = {}
    
    # Process all contexts
    for contexts in iterate_context_windows(dataset, batch_size):
        for ctx_index in range(contexts.shape[0]):
            
            # Get token pair and its next token
            pair = tuple(contexts[ctx_index][0:2].tolist())
            if pair in pairs_counts.keys():
                pairs_counts[pair] += 1
            else:
                pairs_counts[pair] = 1
    
    return pairs_counts


def compare_nested_dicts(dict1, dict2):
    """
    Compare two nested dictionaries. If a sub-dictionary differs for a key, print both versions.

    Args:
        dict1 (dict): First dictionary of dictionaries.
        dict2 (dict): Second dictionary of dictionaries.
    """
    all_keys = dict1.keys() | dict2.keys()  # Union of keys from both dictionaries


    sum_dict_1 = 0
    sum_dict_2 = 0
    for key in all_keys:
        sub_dict1 = dict1.get(key, None)
        sub_dict2 = dict2.get(key, None)

        sum_dict_1 += sub_dict1
        sum_dict_2 += sub_dict2

        if sub_dict1 != sub_dict2:  # Only print if they differ
            print(f"Key '{key}' differs:")
            print(f"  Root_softLabel_dict[{key}] = {sub_dict1}")
            print(f"  singles_pairs_pyDict[{key}] = {sub_dict2}")
            print("-" * 50)
    
    print("Sum Root_softLabel_dict: " + str(sum_dict_1))
    print("Sum singles_pairs_pyDict: " + str(sum_dict_2))

if __name__ == "__main__":
    """
    These stages are designed to be run in order.

    To tokenize data with the Llama 2 tokenizer:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, default=8196, help="pretokenization vocab size. 0 = use Llama 2 tokenizer.")
    parser.add_argument("--context_length", type=int, default=32, help="Context Length")
    parser.add_argument("--exp_case", type=int, default=0, help="Used for sockeye purposes")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument("--stride", type=int, default=1, help="Window stride size")
    parser.add_argument("--perc_stories", type=int, default=100, help="percentage of stories")
    parser.add_argument("--num_tokens_to_proc", type=int, default=0, help="Number of tokens to process, if zero we go with the precentage")
    parser.add_argument("--scheduler_type", type=str, default="cosine", help="lr-scheduling style")
    parser.add_argument("--num_epochs", type=int, default=90, help="Step size")
    parser.add_argument("--num_bins", type=int, default=4, help="Step size")
    parser.add_argument("--Trie_dir", type=str, default="/scratch/st-cthrampo-1/vaalaa/NTP_LLM_DataStats_Trie_TinyStories/Tries/", help="Save Trie File name")
    args = parser.parse_args()


    print("_" * 100)
    print("Creating Trie directory")
    if not os.path.exists(args.Trie_dir):
        os.mkdir(args.Trie_dir)
    print("_" * 100)

    # Step 4: Load and Tokenize the Wikitext-2 Dataset
    # Example usage with stride
    print("_" * 100)
    print("Creating dataloader ... ")
    train_loader, vocab_size = create_dataloader(
        f'/scratch/st-cthrampo-1/vaalaa/TinyStories_tokenizer_bin/synthetic_data/synthetic_tokens.bin',
        context_length=args.context_length,
        batch_size=args.batch_size,
        data_percentage=args.perc_stories,
        stride=args.stride,   
        is_root = False, 
        root_ctx_len = 2,
        num_bins = args.num_bins,
    )
    print("Complete!")
    print("_" * 100)
    args.vocab_size = vocab_size
    print("Running experiments for Vocab Size " + str(args.vocab_size) + " with Context Lenght " + str(args.context_length))
    

    print("_" * 100)
    print("Analyzing the toke pairs ... ")
    num_ctx = len(train_loader)
    # first_token_bins = {token_num:0 for token_num in range(0, args.vocab_size)}
    firstTwo_token_bins = train_loader.dataset.analyze_window_startPairs(args.num_bins)
    pairs_counts_py = build_pairs_counts(train_loader.dataset, batch_size=32)
    compare_nested_dicts(firstTwo_token_bins, pairs_counts_py)
    print("firstTwo_token_bins: " + str(firstTwo_token_bins))
    print("pairs_counts_py: " + str(pairs_counts_py))
    print("Complete!")
    print("_" * 100)
    # input()
    

    print("_" * 100)
    print("Graphing the token pair frequencies ... ")
    save_graph_folder = args.Trie_dir + "/stat_graphs/"
    if not os.path.exists(save_graph_folder):
        os.mkdir(save_graph_folder)
    graph_tokenFirstTwoDist_filename = f"{save_graph_folder}pairTokens_freq.jpg"
    print("Complete!")
    print("_" * 100)

    print("_" * 100)
    print("Separating context indices based on into first two token into bins ...")
    bins, bin_sums = train_loader.dataset.distribute_tuples()
    print("bins: " + str(bins))
    print("Bin loads:", bin_sums)
    print("Complete!")
    print("_" * 100)


    print("_" * 100)
    print("Creating shard folders and indices ...")
    for b in range(0, args.num_bins):
        bin_folder_path = args.Trie_dir + f"/shard{b}/"
        if not os.path.exists(bin_folder_path):
            os.mkdir(bin_folder_path)
        bin_assigned_indices = bins[b]
        np.save(bin_folder_path + 'indices.npy', bin_assigned_indices)
        print("bin_assigned_indices for b = " + str(b) + " is " + str(bin_assigned_indices))
    
    train_loader.dataset.save_pair_locations(args.Trie_dir)

    bin_folder_path = args.Trie_dir + "/root/"
    if not os.path.exists(bin_folder_path):
        os.mkdir(bin_folder_path)
    print("Complete!")
    print("_" * 100)
