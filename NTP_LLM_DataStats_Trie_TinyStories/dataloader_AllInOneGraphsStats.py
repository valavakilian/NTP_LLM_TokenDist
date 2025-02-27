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

def plot_comparative_statistics(stats_list, vocab_sizes, perc_stories, save_path):
    """
    Create comparative plots for branch statistics across different vocabulary sizes.
    
    Args:
        stats_list (list): List of dictionaries containing merged statistics for different vocab sizes
        vocab_sizes (list): List of vocabulary sizes corresponding to each stats dictionary
        perc_stories (int): Percentage of stories used in analysis
        save_path (str): Directory path where plots should be saved
    """
    os.makedirs(save_path, exist_ok=True)
    
    plt.style.use('seaborn')
    figsize = (12, 7)
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    colors = plt.cm.viridis(np.linspace(0, 1, len(stats_list)))
    
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
    
    for config in plot_configs:
        plt.figure(figsize=figsize)
        
        for i, (stats, vocab_size) in enumerate(zip(stats_list, vocab_sizes)):
            plt.plot(
                stats['levels'],
                stats[config['metric']],
                marker=markers[i % len(markers)],
                color=colors[i],
                label=f'Vocab size = {format_number(vocab_size)}',
                linestyle='-',
                markersize=6
            )
        
        if config['log_scale']:
            plt.yscale('log')
            
        plt.title(f"{config['title']} ({perc_stories}% stories)")
        plt.xlabel('Level')
        plt.ylabel(config['ylabel'])
        plt.grid(True)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, config['filename']), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Special plot for OneHots ratio
    plt.figure(figsize=figsize)
    for i, (stats, vocab_size) in enumerate(zip(stats_list, vocab_sizes)):
        onehot_ratio = [oh/tv for oh, tv in zip(stats['total_num_oneHots'], stats['total_visits'])]
        plt.plot(
            stats['levels'],
            onehot_ratio,
            marker=markers[i % len(markers)],
            color=colors[i],
            label=f'Vocab size = {format_number(vocab_size)}',
            linestyle='-',
            markersize=6
        )
    
    plt.title(f'Ratio of OneHots to Total Visits per Level ({perc_stories}% stories)')
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
        merge_branch_statistics
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



def plot_aggregate_statistics(stats_list, vocab_sizes, perc_stories, save_path):
    """
    Create plots showing how aggregate metrics evolve with increasing vocabulary sizes.
    
    Args:
        stats_list (list): List of dictionaries containing merged statistics for different vocab sizes
        vocab_sizes (list): List of vocabulary sizes corresponding to each stats dictionary
        perc_stories (int): Percentage of stories used in analysis
        save_path (str): Directory path where plots should be saved
    """
    aggregate_metrics = [calculate_aggregate_metrics(stats) for stats in stats_list]
    
    os.makedirs(save_path, exist_ok=True)
    
    plt.style.use('seaborn')
    figsize = (10, 6)
    marker_style = 'o-'
    
    plot_configs = [
        {
            'metric': 'weighted_entropy',
            'title': 'Weighted Average Entropy vs Vocabulary Size',
            'ylabel': 'Weighted Average Entropy',
            'log_scale': True,
            'filename': 'aggregate_entropy.png'
        },
        {
            'metric': 'weighted_uniformity',
            'title': 'Weighted Average Uniformity vs Vocabulary Size',
            'ylabel': 'Weighted Average Uniformity',
            'log_scale': True,
            'filename': 'aggregate_uniformity.png'
        },
        {
            'metric': 'weighted_support_size',
            'title': 'Weighted Average Support Size vs Vocabulary Size',
            'ylabel': 'Weighted Average Support Size',
            'log_scale': False,
            'filename': 'aggregate_support_size.png'
        },
        {
            'metric': 'onehot_ratio',
            'title': 'Overall OneHot Ratio vs Vocabulary Size',
            'ylabel': 'OneHot Ratio',
            'log_scale': False,
            'filename': 'aggregate_onehot_ratio.png'
        },
        {
            'metric': 'unique_node_ratio',
            'title': 'Unique Nodes per Visit vs Vocabulary Size',
            'ylabel': 'Unique Nodes per Visit',
            'log_scale': False,
            'filename': 'aggregate_unique_ratio.png'
        },
        {
            'metric': 'total_unique_nodes',
            'title': 'Total Unique Nodes vs Vocabulary Size',
            'ylabel': 'Total Unique Nodes',
            'log_scale': True,
            'filename': 'aggregate_total_unique.png'
        }
    ]
    
    for config in plot_configs:
        plt.figure(figsize=figsize)
        
        values = [metrics[config['metric']] for metrics in aggregate_metrics]
        x_labels = [format_number(vs) for vs in vocab_sizes]
        
        plt.plot(range(len(vocab_sizes)), values, marker_style)
        plt.xticks(range(len(vocab_sizes)), x_labels, rotation=45)
        
        if config['log_scale']:
            plt.yscale('log')
            
        plt.title(f"{config['title']} ({perc_stories}% stories)")
        plt.xlabel('Vocabulary Size')
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--context_length", type=int, default=32, help="Context Length")
    parser.add_argument("--stride", type=int, default=1, help="Window stride size")
    parser.add_argument("--perc_stories", type=int, default=100, help="percentage of stories")
    parser.add_argument("--root_ctx_len", type=int, default=2, help="Size of the root context length")
    parser.add_argument("--Trie_dir", type=str, default="/scratch/st-cthrampo-1/vaalaa/NTP_LLM_DataStats_Trie_TinyStories/Tries/", help="Save Trie File name")
    args = parser.parse_args()

    # List of vocabulary sizes to analyze
    # vocab_sizes = [128 256 512 1024 2048 4096 8192 16384]
    vocab_sizes = [128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    stats_list = []
    
    for vocab_size in vocab_sizes:
        print(f"Processing vocabulary size {vocab_size}...")
        
        # Determine number of bins based on vocabulary size
        num_bins = 100
        
        merged_dict_path = os.path.join(
            args.Trie_dir,
            f"voc{vocab_size}_ctxLen{args.context_length}_{args.perc_stories}%TS_Stride{args.stride}_NumBins{num_bins}/stat_graphs/"
        )
        
        # Load root stats
        with open(os.path.join(merged_dict_path, 'stats_root.json'), 'r') as f:
            root_stats = json.load(f)
        
        # Load shard stats
        shard_stats_list = []
        for shard in range(0, num_bins):
            with open(os.path.join(merged_dict_path, f'stats_shard{shard}.json'), 'r') as f:
                shard_stats_list.append(json.load(f))
        
        # Merge statistics
        merged_stats = merge_branch_statistics(root_stats, shard_stats_list)
        
        # Save merged stats
        with open(os.path.join(merged_dict_path, 'stats_merged.json'), 'w') as f:
            json.dump(merged_stats, f)
        
        stats_list.append(merged_stats)

    # Generate plots
    output_dir = os.path.join(args.Trie_dir, 'FullStatPlots')
    
    plot_comparative_statistics(
        stats_list,
        vocab_sizes,
        args.perc_stories,
        output_dir
    )

    plot_aggregate_statistics(
        stats_list,
        vocab_sizes,
        args.perc_stories,
        output_dir
    )
