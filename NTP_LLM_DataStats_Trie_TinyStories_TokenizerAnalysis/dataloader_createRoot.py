"""
Download, preprocess and serve the WikiText dataset as a DataLoader.
"""

import argparse
import matplotlib.pylab as plt
import numpy as np
import os
from torch.utils.data import DataLoader
from Trie_dataloader import create_dataloader
import time
import json

os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("Importing Done")

SIZE_NODE_BYTES = 56 

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
    parser.add_argument("--num_tokens_to_proc", type=int, default=0, help="Number of tokens to process, if zero we go with the precentage")
    parser.add_argument("--scheduler_type", type=str, default="cosine", help="lr-scheduling style")
    parser.add_argument("--num_epochs", type=int, default=90, help="Step size")
    parser.add_argument("--LoadTrieFromFile", type=bool, default=False, help="Load from existing file")
    parser.add_argument("--num_bins", type=int, default=4, help="Step size")
    parser.add_argument("--root_ctx_len", type=int, default=2, help="Size of the root context lenght, shards will each have context length of (context_length - root_ctx_len) for the Trie")
    parser.add_argument("--Trie_dir", type=str, default="/scratch/st-cthrampo-1/vaalaa/NTP_LLM_DataStats_Trie_MultiProcessor_Wiki/Tries/", help="Save Trie File name")
    parser.add_argument("--tokenizer", type=str, choices=["bpe", "unigram", "wordpiece", "sentencepiece"], required=False, help="Type of tokenizer to use.")
    args = parser.parse_args()


    
    # Step 4: Load and Tokenize the Wikitext-2 Dataset
    # Example usage with stride
    print("_" * 100)
    print("Creating dataloader ... ")
    dataloader, vocab_size = create_dataloader(
        f'/scratch/st-cthrampo-1/vaalaa/TinyStories_tokenizer_bin/data/processed/train_{args.tokenizer}_{args.vocab_size}.bin',
        context_length=args.context_length,
        batch_size=args.batch_size,
        data_percentage=args.perc_stories,
        stride=args.stride,   
        is_root = False, 
        root_ctx_len = 2,
        num_bins = args.num_bins
    )
    print("Complete!")
    print("_" * 100)
    args.vocab_size = vocab_size
    print("Running experiments for Vocab Size " + str(args.vocab_size) + " with Context Lenght " + str(args.context_length))
    

    local_bin_folder_path = "./Trie_info/"
    if not os.path.exists(local_bin_folder_path):
        os.mkdir(local_bin_folder_path)
    print(f"Directory exists: {os.path.exists(local_bin_folder_path)}")
    
    print("_" * 100)
    print("Creating the root transition ...")
    save_tree_folder = local_bin_folder_path
    Root_softLabel_dict = dataloader.dataset.analyze_token_transitions(save_tree_folder, False)
    print("Complete!")
    print("_" * 100)


    print("_" * 100)
    print("Get root stats")
    start_time = time.time()
    stats = dataloader.dataset.calculate_transition_statistics()
    stat_time = time.time() - start_time
    print("Complete!")
    print(f"Took: {stat_time} seconds")
    print("_" * 100)

    with open(args.Trie_dir + 'stat_graphs/stats_root.json', 'w') as f:
        json.dump(stats, f)
    with open(args.Trie_dir + 'stat_graphs/stats_root.json', 'r') as f:
        loaded_dict = json.load(f)
    print("loaded_dict:")
    print(loaded_dict)
