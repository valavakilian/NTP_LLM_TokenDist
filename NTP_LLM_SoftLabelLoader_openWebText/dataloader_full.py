
import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from Trie_dataloader import create_dataloader
import time

os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("Importing Done")


# -----------------------------------------------------------------------------


if __name__ == "__main__":
    """
    These stages are designed to be run in order.

    To tokenize data with the Llama 2 tokenizer:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, default=8196, help="pretokenization vocab size. 0 = use Llama 2 tokenizer.")
    parser.add_argument("--context_length", type=int, default=32, help="Context Length")
    parser.add_argument("--group", type=int, default=0, help="Used for sockeye purposes")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument("--stride", type=int, default=1, help="Window stride size")
    parser.add_argument("--perc_stories", type=int, default=100, help="percentage of stories")
    parser.add_argument("--scheduler_type", type=str, default="cosine", help="lr-scheduling style")
    parser.add_argument("--num_epochs", type=int, default=90, help="Step size")
    parser.add_argument("--LoadTrieFromFile", type=bool, default=False, help="Load from existing file")
    parser.add_argument("--num_bins", type=int, default=4, help="Number of shard")
    parser.add_argument("--root_ctx_len", type=int, default=2, help="Size of the root context lenght, shards will each have context length of (context_length - root_ctx_len) for the Trie")
    parser.add_argument("--Trie_dir", type=str, default="/scratch/st-cthrampo-1/vaalaa/NTP_LLM_DataStats_Trie_MultiProcessor_Wiki/Tries/", help="Save Trie File name")
    args = parser.parse_args()

    bin_folder_path = args.Trie_dir + f"shard{args.group}/"
    
    local_bin_folder_path = "./Trie_info/"
    if not os.path.exists(local_bin_folder_path):
        os.mkdir(local_bin_folder_path)
    print(f"Directory exists: {os.path.exists(local_bin_folder_path)}")
    bin_assigned_indices = np.load(bin_folder_path + 'indices.npy')
    valid_indices = np.loadtxt(bin_folder_path + 'shuffled_indices_locations.txt', dtype=int)
    print(valid_indices[0:100])


    # Step 4: Load and Tokenize the Wikitext-2 Dataset
    # Example usage with stride
    print("_" * 100)
    print("Creating dataloader ... ")
    dataloader, vocab_size = create_dataloader(
        '/arc/project/st-cthrampo-1/vala/openwebtext_karpathy/nanoGPT/data/openwebtext/train.bin',
        context_length=args.context_length,
        batch_size=args.batch_size,
        data_percentage=args.perc_stories,
        token_pairs=bin_assigned_indices,
        valid_indices = valid_indices,
        stride=args.stride,   
        is_root = False, 
        root_ctx_len = 2
    )
    print("Complete!")
    print("_" * 100)
    args.vocab_size = vocab_size

    print("Running experiments for Vocab Size " + str(args.vocab_size) + " with Context Lenght " + str(args.context_length))
    

    print("_" * 100)
    print("Initiate shards ... ")
    # Get the dataset object
    dataloader.dataset.initialize_shard_system(
        base_trie_path = args.Trie_dir,
        mapping_dir = args.Trie_dir,
        transitions_dir = args.Trie_dir + "root",
        num_shards = args.num_bins
    )
    print("Complete!")
    print("_" * 100)

