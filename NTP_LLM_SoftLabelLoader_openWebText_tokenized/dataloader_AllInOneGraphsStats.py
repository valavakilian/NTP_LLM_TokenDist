import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from Trie_dataloader import create_dataloader, Trie_module_protV1
import time
import json
import matplotlib.pyplot as plt

def plot_comparative_statistics(stats_list, token_counts, vocab_size, save_path):
    """
    Create comparative plots for branch statistics across different token counts.
    
    Args:
        stats_list (list): List of dictionaries containing merged statistics for different token counts
        token_counts (list): List of token counts corresponding to each stats dictionary
        vocab_size (int): Vocabulary size used in the analysis
        save_path (str): Directory path where plots should be saved
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Common plot settings
    plt.style.use('seaborn')
    figsize = (12, 7)
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']  # Different markers for different lines
    colors = plt.cm.viridis(np.linspace(0, 1, len(stats_list)))  # Color palette
    
    # Plot functions for different metrics
    plot_configs = [
        {
            'metric': 'avg_entropies',
            'title': 'Average Entropy per Level',
            'ylabel': 'Average Entropy',
            'log_scale': True,
            'filename': 'comparative_avg_entropy.png'
        },
        {
            'metric': 'avg_uniformity',
            'title': 'Average Uniformity per Level',
            'ylabel': 'Average Uniformity',
            'log_scale': True,
            'filename': 'comparative_avg_uniformity.png'
        },
        {
            'metric': 'avg_support_sizes',
            'title': 'Average Support Sizes per Level',
            'ylabel': 'Average Support Size',
            'log_scale': True,
            'filename': 'comparative_avg_support_sizes.png'
        },
        {
            'metric': 'total_visits',
            'title': 'Total Visits per Level',
            'ylabel': 'Total Visits',
            'log_scale': True,
            'filename': 'comparative_total_visits.png'
        },
        {
            'metric': 'unique_nodes',
            'title': 'Unique Nodes per Level',
            'ylabel': 'Number of Unique Nodes',
            'log_scale': True,
            'filename': 'comparative_unique_nodes.png'
        }
    ]
    
    # Create each plot
    for config in plot_configs:
        plt.figure(figsize=figsize)
        
        for i, (stats, token_count) in enumerate(zip(stats_list, token_counts)):
            plt.plot(
                stats['levels'],
                stats[config['metric']],
                marker=markers[i % len(markers)],
                color=colors[i],
                label=f'{format_number(token_count)} tokens',
                linestyle='-',
                markersize=6
            )
        
        if config['log_scale']:
            plt.yscale('log')
            
        plt.title(f"{config['title']} (vocab size = {vocab_size:,})")
        plt.xlabel('Level')
        plt.ylabel(config['ylabel'])
        plt.grid(True)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, config['filename']), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Special plot for OneHots ratio
    plt.figure(figsize=figsize)
    for i, (stats, token_count) in enumerate(zip(stats_list, token_counts)):
        onehot_ratio = [oh/tv for oh, tv in zip(stats['total_num_oneHots'], stats['total_visits'])]
        plt.plot(
            stats['levels'],
            onehot_ratio,
            marker=markers[i % len(markers)],
            color=colors[i],
            label=f'{format_number(token_count)} tokens',
            linestyle='-',
            markersize=6
        )
    
    plt.title(f'Ratio of OneHots to Total Visits per Level (vocab size = {vocab_size:,})')
    plt.xlabel('Level')
    plt.ylabel('OneHots / Total Visits')
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'comparative_onehot_ratio.png'), dpi=300, bbox_inches='tight')
    plt.close()



def merge_branch_statistics(root_stats, shard_stats_list):
    """
    Merge root statistics with multiple branch shard statistics.
    
    Args:
        root_stats (dict): Statistics for the root levels (levels 1,2)
        shard_stats_list (list): List of dictionaries containing branch shard statistics
    
    Returns:
        dict: Merged statistics
    """
    # Initialize the merged stats with the root stats
    merged = {
        'levels': root_stats['levels'].copy(),
        'total_entropies': root_stats['total_entropies'].copy(),
        'avg_entropies': root_stats['avg_entropies'].copy(),
        'total_uniformity': root_stats['total_uniformity'].copy(),
        'avg_uniformity': root_stats['avg_uniformity'].copy(),
        'avg_support_sizes': root_stats['avg_support_sizes'].copy(),
        'total_visits': root_stats['total_visits'].copy(),
        'unique_nodes': root_stats['unique_nodes'].copy(),
        'total_num_oneHots': root_stats['total_num_oneHots'].copy()
    }
    
    # Get the maximum level across all shards
    max_level = max(max(shard['levels']) for shard in shard_stats_list)
    
    # For each level beyond root levels
    for level in range(3, max_level + 1):
        # Initialize accumulators for weighted averages
        total_entropy = 0
        total_uniformity = 0
        total_visits_level = 0
        total_unique_nodes = 0
        total_num_oneHots = 0
        weighted_support_sizes = 0
        
        # Process each shard
        for shard in shard_stats_list:
            if level in shard['levels']:
                level_idx = shard['levels'].index(level)
                visits = shard['total_visits'][level_idx]
                
                # Accumulate weighted entropy and uniformity
                total_entropy += shard['total_entropies'][level_idx]
                total_uniformity += shard['total_uniformity'][level_idx]
                
                # Accumulate other metrics
                total_visits_level += visits
                total_unique_nodes += shard['unique_nodes'][level_idx]
                total_num_oneHots += shard['total_num_oneHots'][level_idx]
                weighted_support_sizes += shard['avg_support_sizes'][level_idx] * visits
        
        # Append level
        merged['levels'].append(level)
        
        # Append total entropy and calculate average
        merged['total_entropies'].append(total_entropy)
        merged['avg_entropies'].append(total_entropy / total_visits_level if total_visits_level > 0 else 0)
        
        # Append total uniformity and calculate average
        merged['total_uniformity'].append(total_uniformity)
        merged['avg_uniformity'].append(total_uniformity / total_visits_level if total_visits_level > 0 else 0)
        
        # Calculate and append average support sizes
        avg_support_size = weighted_support_sizes / total_unique_nodes if total_unique_nodes > 0 else 0
        merged['avg_support_sizes'].append(avg_support_size)
        
        # Append other metrics
        merged['total_visits'].append(total_visits_level)
        merged['unique_nodes'].append(total_unique_nodes)
        merged['total_num_oneHots'].append(total_num_oneHots)
    
    return merged




def calculate_aggregate_metrics(stats):
    """
    Calculate aggregate metrics across all levels using weighted averages where appropriate.
    
    Args:
        stats (dict): Dictionary containing the merged statistics
        
    Returns:
        dict: Dictionary of aggregate metrics
    """
    total_visits_all = sum(stats['total_visits'])
    
    # Weighted average entropy
    total_entropy = sum(e * v for e, v in zip(stats['avg_entropies'], stats['total_visits']))
    weighted_avg_entropy = total_entropy / total_visits_all if total_visits_all > 0 else 0
    
    # Weighted average uniformity
    total_uniformity = sum(u * v for u, v in zip(stats['avg_uniformity'], stats['total_visits']))
    weighted_avg_uniformity = total_uniformity / total_visits_all if total_visits_all > 0 else 0
    
    # Weighted average support size
    total_support = sum(s * v for s, v in zip(stats['avg_support_sizes'], stats['total_visits']))
    weighted_avg_support = total_support / total_visits_all if total_visits_all > 0 else 0
    
    # Overall OneHot ratio
    total_onehots = sum(stats['total_num_oneHots'])
    onehot_ratio = total_onehots / total_visits_all if total_visits_all > 0 else 0
    
    # Average unique nodes per visit
    total_unique = sum(stats['unique_nodes'])
    unique_ratio = total_unique / total_visits_all if total_visits_all > 0 else 0
    
    return {
        'weighted_entropy': weighted_avg_entropy,
        'weighted_uniformity': weighted_avg_uniformity,
        'weighted_support_size': weighted_avg_support,
        'onehot_ratio': onehot_ratio,
        'unique_node_ratio': unique_ratio,
        'total_visits': total_visits_all,
        'total_unique_nodes': total_unique
    }

def plot_aggregate_statistics(stats_list, token_counts, vocab_size, save_path):
    """
    Create plots showing how aggregate metrics evolve with increasing token counts.
    
    Args:
        stats_list (list): List of dictionaries containing merged statistics for different token counts
        token_counts (list): List of token counts corresponding to each stats dictionary
        vocab_size (int): Vocabulary size used in the analysis
        save_path (str): Directory path where plots should be saved
    """
    # Calculate aggregate metrics for each token count
    aggregate_metrics = [calculate_aggregate_metrics(stats) for stats in stats_list]
    
    # Create save directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Common plot settings
    plt.style.use('seaborn')
    figsize = (10, 6)
    marker_style = 'o-'
    
    def format_number(num):
        if num >= 1_000_000_000:
            return f"{num // 1_000_000_000}B"
        elif num >= 1_000_000:
            return f"{num // 1_000_000}M"
        elif num >= 1_000:
            return f"{num // 1_000}K"
        else:
            return str(num)
    
    # Plot configurations
    plot_configs = [
        {
            'metric': 'weighted_entropy',
            'title': 'Weighted Average Entropy vs Tokens Processed',
            'ylabel': 'Weighted Average Entropy',
            'log_scale': True,
            'filename': 'aggregate_entropy.png'
        },
        {
            'metric': 'weighted_uniformity',
            'title': 'Weighted Average Uniformity vs Tokens Processed',
            'ylabel': 'Weighted Average Uniformity',
            'log_scale': True,
            'filename': 'aggregate_uniformity.png'
        },
        {
            'metric': 'weighted_support_size',
            'title': 'Weighted Average Support Size vs Tokens Processed',
            'ylabel': 'Weighted Average Support Size',
            'log_scale': False,
            'filename': 'aggregate_support_size.png'
        },
        {
            'metric': 'onehot_ratio',
            'title': 'Overall OneHot Ratio vs Tokens Processed',
            'ylabel': 'OneHot Ratio',
            'log_scale': False,
            'filename': 'aggregate_onehot_ratio.png'
        },
        {
            'metric': 'unique_node_ratio',
            'title': 'Unique Nodes per Visit vs Tokens Processed',
            'ylabel': 'Unique Nodes per Visit',
            'log_scale': False,
            'filename': 'aggregate_unique_ratio.png'
        },
        {
            'metric': 'total_unique_nodes',
            'title': 'Total Unique Nodes vs Tokens Processed',
            'ylabel': 'Total Unique Nodes',
            'log_scale': True,
            'filename': 'aggregate_total_unique.png'
        }
    ]
    
    # Create each plot
    for config in plot_configs:
        plt.figure(figsize=figsize)
        
        # Extract values for current metric
        values = [metrics[config['metric']] for metrics in aggregate_metrics]
        
        # Create x-axis labels
        x_labels = [format_number(tc) for tc in token_counts]
        
        plt.plot(range(len(token_counts)), values, marker_style)
        plt.xticks(range(len(token_counts)), x_labels, rotation=45)
        
        if config['log_scale']:
            plt.yscale('log')
            
        plt.title(f"{config['title']} (vocab size = {vocab_size:,})")
        plt.xlabel('Tokens Processed')
        plt.ylabel(config['ylabel'])
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, config['filename']), dpi=300, bbox_inches='tight')
        plt.close()


def format_number(num):
    """
    Format a number to abbreviated form (K for thousands, M for millions, B for billions)
    
    Args:
        num (int): Number to format
        
    Returns:
        str: Formatted number string
    """
    if num >= 1_000_000_000:  # Billions
        return f"{num // 1_000_000_000}B"
    elif num >= 1_000_000:     # Millions
        return f"{num // 1_000_000}M"
    elif num >= 1_000:         # Thousands
        return f"{num // 1_000}K"
    else:
        return str(num)


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
    args = parser.parse_args()


    args.vocab_size=50257
    args.context_length=16
    args.stride=1
    


    # Example usage:
    # Assuming you have multiple merged stats for different token counts:
    token_counts = [100000, 200000, 500000, 1000000, 2000000, 10000000, 20000000, 50000000, 100000000, 200000000, 500000000, 1000000000, 2000000000, 5000000000]
    stats_list = []
    tries_folder_path = "/scratch/st-cthrampo-1/vaalaa/NTP_LLM_DataStats_Trie_MultiProcessor_OpenWebText_New/Tries/"
    for tokens in token_counts:
        

        if tokens >= 100000000:
            num_bins = 100
        else:
            num_bins = 10
        
        tokens_processed_str = format_number(tokens)
        print(f"Processing {tokens_processed_str} tokens ...")

        merged_dict_path = tries_folder_path + f"voc{args.vocab_size}_ctxLen{args.context_length}_{tokens_processed_str}TokensOpWT_Stride{args.stride}_NumBins{num_bins}/stat_graphs/"

        
        with open(merged_dict_path + 'stats_root.json', 'r') as f:
            root_stats = json.load(f)
        
        shard_stats_list = []
        for shard in range(0, num_bins):
            with open(merged_dict_path + f'stats_shard{shard}.json', 'r') as f:
                shard_stats_list.append(json.load(f))


        merged_stats = merge_branch_statistics(root_stats, shard_stats_list)
        with open(merged_dict_path + 'stats_merged.json', 'w') as f:
            json.dump(merged_stats, f)
        
        stats_list.append(merged_stats)

    plot_comparative_statistics(
        stats_list,
        token_counts,
        args.vocab_size,
        '/scratch/st-cthrampo-1/vaalaa/NTP_LLM_DataStats_Trie_MultiProcessor_OpenWebText_New/Tries/FullStatPlots/'
    )

    plot_aggregate_statistics(
        stats_list,
        token_counts,
        args.vocab_size,
        '/scratch/st-cthrampo-1/vaalaa/NTP_LLM_DataStats_Trie_MultiProcessor_OpenWebText_New/Tries/FullStatPlots/'
    )
