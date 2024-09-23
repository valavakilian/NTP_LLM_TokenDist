"""
Download, preprocess and serve the Wikitext dataset as a DataLoader.
"""

import argparse
import glob
import json
import os
import random
from typing import List
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
import requests
import sentencepiece as spm
import torch
import torch.distributed as dist
from tqdm import tqdm

from tokenizer import Tokenizer
from random import shuffle
from collections import Counter

import matplotlib.pylab as plt
import time as time 
import psutil

import pickle 
import gc
import tracemalloc
import resource
from datasets import load_dataset

DATA_CACHE_DIR = "/scratch/st-cthrampo-1/vaalaa/NTP_LLM_Scaling/WikiText"
DATA_PROCESS_DIR = "/scratch/st-cthrampo-1/vaalaa/NTP_LLM_Scaling/WikiText/WikiText_processing_files"


def train_vocab(vocab_size, target_tokens, num_stories = 2000):
    """
    Trains a custom sentencepiece tokenizer on the Wikitext dataset.
    The custom tokenizer files will be saved in DATA_CACHE_DIR/tok{N} directories,
    where N is the vocab size. This is also where the pretok .bin files will go.
    """
    assert vocab_size > 0, "Vocab size must be positive"

    # output file prefix path for sentencepiece
    prefix = os.path.join(DATA_PROCESS_DIR, f"tok{vocab_size}")

    dataset = load_dataset("wikitext", cache_dir = "/arc/project/st-cthrampo-1/vala/WikiText")
    dataset = dataset["train"]

    wiki_file = os.path.join(DATA_CACHE_DIR, "wiki.txt")
    with open(wiki_file, "w", encoding="utf-8") as of:
        
        for text in tqdm(dataset[0:num_stories]['text']):
            text = text.strip()
            of.write(text + "\n")
    print(f"Size is: {os.path.getsize(wiki_file) / 1024 / 1024:.2f} MB")

    print("Will now train the vocab...")

    spm.SentencePieceTrainer.train(input=wiki_file,
                                   model_prefix=prefix,
                                   model_type="bpe",
                                   vocab_size=vocab_size, 
                                   user_defined_symbols=target_tokens)

    print(f"Trained tokenizer is in {prefix}.model")
    print("Done.")


def tokenize_one(vocab_size, num_stories_start, num_stories_end, Train = True):

    tokenizer_model = get_tokenizer_model_path(vocab_size)
    enc = Tokenizer(tokenizer_model)

    input("Got here")
    dataset = load_dataset("wikitext", cache_dir = "/arc/project/st-cthrampo-1/vala/WikiText")
    input("just passed")
    dataset = dataset["train"]

    all_tokens = []

    if Train:
        for text in tqdm(dataset[num_stories_start:num_stories_end]['text']):
            text = text.strip()  # get rid of leading/trailing whitespace
            tokens = enc.encode(text, bos=True, eos=False)  # encode the text, use BOS
            all_tokens.extend(tokens)
    else:
        for text in tqdm(dataset[-num_stories:]['text']):
            text = text.strip()  # get rid of leading/trailing whitespace
            tokens = enc.encode(text, bos=True, eos=False)  # encode the text, use BOS
            all_tokens.extend(tokens)
    
    # convert to uint16 nparray
    all_tokens = np.array(all_tokens, dtype=np.uint16)
    # calculate the output filename

    if Train:
        bin_basename = "Wikitext_voc" + str(vocab_size) + "_stor" + str(num_stories)+ ".bin" 
    else:
        bin_basename = "Wikitext_voc" + str(vocab_size) + "_stor" + str(num_stories)+ "_Test.bin" 
    
    tokenized_filename = os.path.join(DATA_PROCESS_DIR, bin_basename)
    # write the bytes
    with open(tokenized_filename, "wb") as f:
        f.write(all_tokens.tobytes())
    # calculate the average sequence length (they are separated by BOS=1)
    avg_seq_len = all_tokens.size / ((all_tokens == 1).sum())
    print(f"Saved {tokenized_filename}, average seqlen: {avg_seq_len:.2f}")


def pretokenize_target_stories(vocab_size, num_stories_start, num_stories_end):

    tokenize_one(vocab_size, num_stories_start, num_stories_end, Train = True)

    print("Done.")



class PretokDataset(torch.utils.data.IterableDataset):
    """Loads pretokenized examples from disk and yields them as PyTorch tensors."""

    def __init__(self, split, max_seq_len, vocab_size, vocab_source, num_stories, AR_training, Train = "True"):
        super().__init__()
        self.split = split
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.vocab_source = vocab_source
        self.iteration = 0
        self.num_batches = 0
        self.AR_training = AR_training
        self.Train = Train

        # get worker info within a DataLoader
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        # get DDP rank info
        rank = dist.get_rank() if dist.is_initialized() else 0
        # combine the worker_id and worker_rank to create a unique seed for rng
        seed = 42 + worker_id + 1337 * rank
        self.rng = random.Random(seed)
        print(f"Created a PretokDataset with rng seed {seed}")
        if self.vocab_source == "custom":
            # the .bin files are in tok{N} directory
            # bin_dir = os.path.join(DATA_PROCESS_DIR, f"tok{self.vocab_size}")
            bin_dir = "/scratch/st-cthrampo-1/vaalaa/NTP_LLM_Scaling/WikiText/WikiText_processing_files/"
            if Train:
                shard_filenames = sorted(glob.glob(os.path.join(bin_dir, "*_voc" + str(vocab_size) + "_stor" + str(num_stories) + ".bin")))
            else:
                shard_filenames = sorted(glob.glob(os.path.join(bin_dir, "*_voc" + str(vocab_size) + "_stor" + str(num_stories) + "_Test" + ".bin")))
            # shard_filenames = [bin_dir + "data00_voc" + str(vocab_size) + "_stor" + str(num_stories) + ".bin"]
        # train/test split. let's use only shard 0 for test split, rest train
        # shard_filenames = shard_filenames[1:] if self.split == "train" else shard_filenames[:1]
        assert len(shard_filenames)>0, f"No bin files found in {bin_dir}"
        
        self.rng.shuffle(shard_filenames)
        for shard in shard_filenames:
            # open the dataset for reading but keep it on disk with memmap
            self.m = np.memmap(shard, dtype=np.uint16, mode="r")

            index = 0
            self.indices_to_sample_from = []
            while index < len(self.m) - self.max_seq_len:
                self.indices_to_sample_from.append(index)
                index += (self.max_seq_len // 4)

            self.num_batches = len(self.indices_to_sample_from) // self.max_seq_len

            assert self.num_batches > 0, "this shard is way too small? investigate."

    def __iter__(self):
        print("Total number of training samples: " + str(len(self.indices_to_sample_from)))

        self.rng.shuffle(self.indices_to_sample_from)
        for ix in self.indices_to_sample_from:
            start = ix
            end = start + self.max_seq_len + 1
            chunk = torch.from_numpy((self.m[start:end]).astype(np.int64))

            if self.AR_training:
                x = chunk[:-1]
                y = chunk[1:]
            else:
                x = chunk[:-1]
                y = chunk[-1]
            yield x, y

    def __next__(self):
        if self.iteration < self.num_batches:
            sample = self.__iter__()
            self.iteration += 1
            return sample
        else:
            # Raise StopIteration when the iteration is complete
            self.rng.shuffle(self.indices_to_sample_from)
            self.iteration = 0
            raise StopIteration

                

# -----------------------------------------------------------------------------
# public interface functions

def get_tokenizer_model_path(vocab_size):
    """
    Returns path to the sentencepiece tokenizer model for a given vocab size
    vocab_size = 0 designates the default Llama 2 tokenizer, in that case
    None is returned.
    """
    if vocab_size == 0:
        return None
    else:
        return os.path.join(DATA_PROCESS_DIR, f"tok{vocab_size}.model")

class Task:

    @staticmethod
    def iter_batches(batch_size, device, num_workers=0, **dataset_kwargs):
        ds = PretokDataset(**dataset_kwargs)
        dl = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, pin_memory=True, num_workers=num_workers
        )
        return dl



def calc_data_context_info(context_length, vocab_size, vocab_source, AR_training, support_set_matrix, context_to_index_dict, context_count_dict, pos_context_count):

    ds = PretokDataset("train", context_length, vocab_size, vocab_source, num_stories, AR_training = AR_training)
    dl = torch.utils.data.DataLoader(
        ds, batch_size=1000, pin_memory=True, num_workers=0
    )

    entropy = 0
    entropy_pos = {}
    for pos in range(1, context_length + 1):
        entropy_pos[pos] = 0

    if len(context_to_index_dict.keys()) == 0:
        index_counter = 0
    else:   
        index_counter = max(context_to_index_dict.values())
        index_counter += 1

    num_datapoints = 0
    print("Creating context dictionaries: ")
    for batch_idx, (X, Y) in tqdm(enumerate(dl)):
        for index in range(0, Y.shape[0]):
            # Should this sum be here ? 
            # num_datapoints += 1
            x = X[index, :]
            y = Y[index, :]
        
            for i in range(1, context_length + 1):
                context = x[:i]
                token = y[i - 1]

                pos_context_count[len(context)] += 1

                # context = "_".join([str(n) for n in context.tolist()])
                context = tuple(context.tolist())
                token = token.item()

                num_datapoints += 1

                if context in context_to_index_dict.keys():
                    context_index = context_to_index_dict[context]
                    support_set_matrix[context_index, token] += 1
                    context_count_dict[context] += 1
                else:
                    context_to_index_dict[context] = index_counter
                    context_count_dict[context] = 1
                    if support_set_matrix.shape[0] <= index_counter:
                        print("Updating size")
                        support_set_matrix = np.concatenate((support_set_matrix, np.zeros((10000000,vocab_size))), axis=0)
                    support_set_matrix[index_counter, token] += 1
                    index_counter += 1

        del X
        del Y
        gc.collect()
    del dl
    del ds
    gc.collect()

    return_dict = {}

    set_unique_sup_set = set()
    dict_unique_sup_set_counts = {}
    dict_unique_sup_set_unique_context_sets = {}
    dict_unique_sup_set_unique_context_sets_counts = {}
    num_unique_sup_sets = 0
    sup_set_size_dist = {}
    
    entropy_calc_count = 0
    print("Calculating Entropy and SupSet info")
    for context in tqdm(context_to_index_dict.keys()):
        context_index = context_to_index_dict[context]
        entropy_context = 0
        # context_freq = sum(support_set_matrix[context_index,:].nonzero(as_tuple=True)[0])
        context_freq = context_count_dict[context]

        len_context = len(context)
        non_zero_tokens = tuple(support_set_matrix[context_index,:].nonzero()[0])
        # non_zero_tokens = [x.item() for x in non_zero_tokens]
        # print(non_zero_tokens)
        # input()

        # if len(non_zero_tokens) != 1:
        #     print(context)
        #     print(non_zero_tokens)
        #     print(context_freq)
        #     input()

        for token in non_zero_tokens:
            entropy_context -= support_set_matrix[context_index,token].item() / context_freq * np.log(support_set_matrix[context_index,token].item() / context_freq)

        entropy_pos[len_context] += (context_freq / pos_context_count[len_context]) * entropy_context

        entropy += (context_freq / num_datapoints) * entropy_context
        entropy_calc_count += 1

        # print(list(non_zero_tokens))
        # print(tuple(non_zero_tokens))
        # input()
        # if tuple(non_zero_tokens) in set_unique_sup_set:
        #     input("Found One")
        #     continue
        # else:
        # context_sup_set = tuple(support_set_matrix[context_index,:])
        
        if non_zero_tokens in set_unique_sup_set:
            dict_unique_sup_set_counts[non_zero_tokens] += 1
            dict_unique_sup_set_unique_context_sets[non_zero_tokens].add(context)

            if context not in dict_unique_sup_set_unique_context_sets[non_zero_tokens]:
                dict_unique_sup_set_unique_context_sets_counts[non_zero_tokens] += 1
            
            sup_set_size_dist[len(non_zero_tokens)] += 1
        else:
            set_unique_sup_set.add(non_zero_tokens)
            dict_unique_sup_set_counts[non_zero_tokens] = 1
            dict_unique_sup_set_unique_context_sets[non_zero_tokens] = set()
            dict_unique_sup_set_unique_context_sets[non_zero_tokens].add(context)

            if context not in dict_unique_sup_set_unique_context_sets[non_zero_tokens]:
                dict_unique_sup_set_unique_context_sets_counts[non_zero_tokens] += 1
            
            sup_set_size_dist[len(non_zero_tokens)] = 1

        # if context_sup_set in set_unique_sup_set:
        #     dict_unique_sup_set_counts[context_sup_set] += 1
        # else:
        #     dict_unique_sup_set_counts[context_sup_set] = 1
        #     set_unique_sup_set.add(context_sup_set)
        #     num_unique_sup_sets += 1
    
    print("entropy_calc_count: " + str(entropy_calc_count))

    num_full_contexts = num_datapoints // context_length
    total_num_contexts = num_datapoints

    num_unique_contexts = index_counter - 1
    num_unique_sup_sets = len(set_unique_sup_set)
    sup_set_context_count_dist = np.sort(list(dict_unique_sup_set_counts.values()))
    sup_set_unique_context_count_dist = np.sort(list(dict_unique_sup_set_unique_context_sets_counts.values()))

    return_dict = {"entropy": entropy,
                   "entropy_pos": entropy_pos,
                   "total_num_contexts": total_num_contexts,
                   "num_full_contexts": num_full_contexts,
                   "num_unique_contexts": num_unique_contexts,
                   "num_unique_sup_sets": num_unique_sup_sets,
                   "sup_set_context_count_dist": sup_set_context_count_dist,
                   "sup_set_unique_context_count_dist": sup_set_unique_context_count_dist,
                   "sup_set_size_dist": sup_set_size_dist
                   }

    return support_set_matrix, context_to_index_dict, context_count_dict, pos_context_count, return_dict

# -----------------------------------------------------------------------------
# CLI for constructing the dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, default=1024, help="pretokenization vocab size. 0 = use Llama 2 tokenizer.")
    parser.add_argument("--context_length", type=int, default=16, help="Number of stories to use")
    args = parser.parse_args()
    

    list_num_stories = [100, 250, 500, 750, 1000, 2500, 5000, 7500, 10000, 12500, 15000, 17500, 20000, 22500, 25000, 27500, 30000, 32500, 35000, 37500, 40000]
    # list_num_stories = [100, 250, 500, 550, 600, 650, 700]
    train_vocab(vocab_size=args.vocab_size, target_tokens = [], num_stories = 10000)
    start_stories = 0

    support_set_matrix = np.zeros((10000000,args.vocab_size))
    context_to_index_dict = {}
    context_count_dict = {}

    pos_context_count = {}
    for pos in range(1, args.context_length + 1):
        pos_context_count[pos] = 0
    
    vocab_source = "custom"
    AR_training = True

    for num_stories in list_num_stories:
        print("Num Stories being processed: " + str(num_stories))
        pretokenize_target_stories(vocab_size=args.vocab_size, num_stories_start = start_stories, num_stories_end = num_stories)
        start_stories = num_stories

        # tracemalloc.start()
        start_time = time.time()
        support_set_matrix, context_to_index_dict, context_count_dict, pos_context_count, return_dict = calc_data_context_info(args.context_length, args.vocab_size, vocab_source, AR_training, support_set_matrix, context_to_index_dict, context_count_dict, pos_context_count)
        total_time = time.time() - start_time

        with open("/scratch/st-cthrampo-1/vaalaa/NTP_LLM_Scaling/logs_wiki/Wiki_Voc_" + str(args.vocab_size) + "_ctxLen_" + str(args.context_length) + "_stories_" + str(num_stories)  + '.pkl', 'wb') as f:
            pickle.dump(return_dict, f)

        # snapshot = tracemalloc.take_snapshot()
        # top_stats = snapshot.statistics('lineno')

        # print("[ Top 10 ]")
        # for stat in top_stats[:10]:
        #     print(stat)

        # tracemalloc.stop()

        
        print("Entropy is " + str(return_dict["entropy"]))
        print("total_num_contexts: " + str(return_dict["total_num_contexts"]))
        print("num_unique_contexts: " + str(return_dict["num_unique_contexts"]))
        print("num_unique_sup_sets: " + str(return_dict["num_unique_sup_sets"]))
        print("Time for operation: " + str(total_time / 60) + " minutes.")
        print('RAM memory % used:', psutil.virtual_memory().percent)
        print('RAM memory % free:', psutil.virtual_memory().free/(1024**3))
        print('RAM Used (GB):', psutil.virtual_memory()[3]/(1024**3))
        process = psutil.Process(os.getpid())
        print('Physical RAM Used (GB):', process.memory_info().rss/(1024**3))
        print('Physical RAM % Used (GB):', process.memory_percent())
        print('MidPeak RAM Used (GB):', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2))
        print("____________________________________________")
        # input()

    