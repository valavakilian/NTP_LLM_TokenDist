"""
Download, preprocess and serve the WikiText dataset as a DataLoader.
"""
import argparse
import matplotlib.pylab as plt
import numpy as np
import os
from torch.utils.data import DataLoader
from Trie_dataloader import TokenizedDataset, create_dataloader

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
    parser.add_argument("--exp_case", type=int, default=0, help="Used for sockeye purposes")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument("--stride", type=int, default=1, help="Window stride size")
    parser.add_argument("--perc_stories", type=int, default=100, help="percentage of stories")
    parser.add_argument("--scheduler_type", type=str, default="cosine", help="lr-scheduling style")
    parser.add_argument("--num_epochs", type=int, default=90, help="Step size")
    parser.add_argument("--num_bins", type=int, default=4, help="Step size")
    parser.add_argument("--Trie_dir", type=str, default="/scratch/st-cthrampo-1/vaalaa/NTP_LLM_DataStats_Trie_MultiProcessor_Wiki/Tries/", help="Save Trie File name")
    args = parser.parse_args()

    dataset_dir = '/arc/project/st-cthrampo-1/vala/openwebtext_karpathy/nanoGPT/data/openwebtext/train.bin'  # Your saved dataset folder

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
        '/arc/project/st-cthrampo-1/vala/openwebtext_karpathy/nanoGPT/data/openwebtext/train.bin',
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
    

    print("_" * 100)
    print("Analyzing the toke pairs ... ")
    num_ctx = len(train_loader)
    # first_token_bins = {token_num:0 for token_num in range(0, args.vocab_size)}
    firstTwo_token_bins = train_loader.dataset.analyze_window_startPairs(args.num_bins)
    print("Complete!")
    print("_" * 100)
    

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
    
    train_loader.dataset.save_pair_locations(args.Trie_dir)

    bin_folder_path = args.Trie_dir + "/root/"
    if not os.path.exists(bin_folder_path):
        os.mkdir(bin_folder_path)
    print("Complete!")
    print("_" * 100)
