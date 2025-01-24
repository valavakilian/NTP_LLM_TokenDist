
import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from Trie_dataloader import create_dataloader
import time
import random

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
    parser.add_argument("--num_tokens_to_proc", type=int, default=0, help="Number of tokens to process, if zero we go with the precentage")
    parser.add_argument("--scheduler_type", type=str, default="cosine", help="lr-scheduling style")
    parser.add_argument("--num_epochs", type=int, default=90, help="Step size")
    parser.add_argument("--LoadTrieFromFile", type=bool, default=False, help="Load from existing file")
    parser.add_argument("--num_bins", type=int, default=4, help="Number of shard")
    parser.add_argument("--root_ctx_len", type=int, default=2, help="Size of the root context lenght, shards will each have context length of (context_length - root_ctx_len) for the Trie")
    parser.add_argument("--Trie_dir", type=str, default="/scratch/st-cthrampo-1/vaalaa/NTP_LLM_DataStats_Trie_MultiProcessor_Wiki/Tries/", help="Save Trie File name")
    args = parser.parse_args()

    
    if not os.path.exists(args.Trie_dir):
        os.mkdir(args.Trie_dir)
    print(f"Directory exists: {os.path.exists(args.Trie_dir)}")


    # Step 4: Load and Tokenize the Wikitext-2 Dataset
    # Example usage with stride
    print("_" * 100)
    print("Creating dataloader ... ")
    dataloader, vocab_size = create_dataloader(
        '/arc/project/st-cthrampo-1/vala/openwebtext_karpathy/nanoGPT/data/openwebtext/train.bin',
        context_length=args.context_length,
        batch_size=args.batch_size,
        data_percentage=args.perc_stories,
        stride=args.stride,   
        is_root = False, 
        root_ctx_len = 2,
        num_tokens_to_proc = args.num_tokens_to_proc
    )
    print("Complete!")
    print("_" * 100)
    args.vocab_size = vocab_size

    print("Running experiments for Vocab Size " + str(args.vocab_size) + " with Context Lenght " + str(args.context_length))
    

    print("_" * 100)
    print("Initiate shards ... ")
    # Get the dataset object
    # dataloader.dataset.initialize_shard_system(
    #     base_trie_path = args.Trie_dir,
    #     mapping_dir = args.Trie_dir,
    #     transitions_dir = args.Trie_dir + "root",
    #     num_shards = args.num_bins
    # )
    dataloader.dataset.initialize_shard_system(
        base_trie_path = "./",
        mapping_dir = ".",
        transitions_dir = "./" + "root",
        num_shards = args.num_bins
    )
    print("Complete!")
    print("_" * 100)

    thing = 0
    for X in dataloader:
        if thing == 1:
            break 
        thing += 1
    print("_" * 100)
    print(X)
    print("_" * 100)


    # Run several batches and time them
    num_batches = 10
    total_time = 0

    print(f"\nTesting {num_batches} batches with batch_size={args.batch_size}")
    print("-" * 50)

    for i in range(num_batches):
        start_idx = random.randint(0, 5000 - args.batch_size)
        print(start_idx)
        
        start_time = time.time()
        batch = dataloader.dataset.get_batch_with_labels_DEBUG(start_idx, args.batch_size)
        end_time = time.time()
        
        batch_time = end_time - start_time
        total_time += batch_time
        
        print(f"Batch {i+1}: {batch_time:.4f} seconds")
        print(f"  Sequences shape: {batch.sequences.shape}")
        print(f"  Hard targets shape: {batch.hard_targets.shape}")
        print(f"  Soft targets shape: {batch.soft_targets.shape}")

    avg_time = total_time / num_batches
    print("\nTiming Summary:")
    print(f"Average batch time: {avg_time:.4f} seconds")
    print(f"Throughput: {args.batch_size/avg_time:.2f} sequences/second")


    

