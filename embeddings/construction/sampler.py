import numpy as np
import random
import torch
from torch.utils.data import BatchSampler, Sampler
from tqdm import tqdm
import os
from os import path
import sys
import pickle 
import datetime

class pair_sampler(Sampler):

    def __init__(self, data_source, options_dict):
        self.labels = data_source.labels
        self.speakers = data_source.speakers
        self.batch_size = options_dict["batch_size"]
        
        # load = False

        if options_dict["pair_list_fn"] is None:
            # get_pair_list shuffles pairs given n_max_pairs
            self.pair_list = get_pair_list(
                data_source.labels, options_dict, both_directions=False,
                n_max_pairs=options_dict["n_max_pairs"]
                )
            if options_dict["n_max_pairs"] is None:
                random.seed(options_dict['rnd_seed'])
                # random.shuffle(self.pair_list)

            # print(data_source.labels)    
            # print(self.pair_list)
            print("No. Pairs:", len(self.pair_list))
        else:
            # pair_list_dir = path.join("pair_lists", options_dict["train_lang"])
            # pair_list_fn = os.path.join(pair_list_dir, "pair_list.npz")
            # pair_list_npz = np.load(pair_list_fn)
            # print(pair_list_npz["arr_0"])
            # print(len(pair_list_npz["arr_0"]))
            # self.pair_list = pair_list_npz["arr_0"].tolist()

            pair_list_fn = options_dict["pair_list_fn"]
            print("Reading: ", pair_list_fn)
            with open(pair_list_fn, "rb") as f:
                self.pair_list = pickle.load(f)


        self.n_minibatches = options_dict["n_minibatches"]
        
        # self.tuplet_list_idxs, self.tuplet_list_hot = get_tuplet_lists(self.pair_list, self.labels, 5)
        # self.tuplet_list_idxs, self.tuplet_list_hot = tuplets_shuffle(self.tuplet_list_idxs, self.tuplet_list_hot)
        original = self.pair_list.copy()
        self.batch = get_minibatch(original, self.labels, self.batch_size, self.n_minibatches)
        
        # self.batch = self.speaker_minibatch_const(self.speakers, self.labels, self.pair_list)
        # self.batch = [self.batch[x:x+self.batch_size] for x in range(0,len(self.batch), self.batch_size)]


    def minibatch_reconstruction(self):
        print("Start shuffle", datetime.datetime.now())
        random.shuffle(self.pair_list)
        print("End shuffle", datetime.datetime.now())
        original = self.pair_list.copy()
        self.batch = get_minibatch(original, self.labels, self.batch_size, self.n_minibatches)

    def speaker_minibatch_const(self, speakers, labels, pairs):
        speakers_unique = np.unique(speakers)
        # print(len(speakers_unique))
        # print(pairs[1000])
        # print(labels[1005:1007])
        # print(speakers[1005:1007])
        random.shuffle(pairs)
        batch = []

        for speaker in speakers_unique:
            minibatch_pairs = []
            minibatch_labels = []
            for pair in pairs:
                if speakers[pair[0]] == speaker and labels[pair[0]] not in minibatch_labels:
                # if speakers[pair[0]] != speakers[pair[1]] and speakers[pair[0]] == speaker and labels[pair[0]] not in minibatch_labels:

                    minibatch_pairs.append(pair)
                    minibatch_labels.extend(labels[pair[0]])
                    # if len(minibatch_pairs) > 99:
                    #     break
                        
            batch.extend(minibatch_pairs)

        batch = batch[:(len(batch)//self.batch_size)*self.batch_size]

        return batch


    def __iter__(self):
        """
        returns tuplet of idxs and 2-hot eg. [2, 45, 67, 102, 43], [0, 1, 0, 0, 1]
        """
        return iter(self.batch)        
        # print(self.tuplet_list_idxs, self.tuplet_list_hot)

        # return iter(zip(self.tuplet_list_idxs, self.tuplet_list_hot))




### UTILITY ###
# def get_pair_list(labels, options_dict, both_directions=True, n_max_pairs=None):
#     """Return a list of tuples giving indices of matching types."""
#     print(labels)
#     N = len(labels)
#     match_list = []
#     for n in range(N - 1):
#         cur_label = labels[n]
#         for cur_match_i in (n + 1 + np.where(np.asarray(labels[n + 1:]) ==
#                 cur_label)[0]):
#             match_list.append((n, cur_match_i))
#             # if both_directions:
#             #     match_list.append((cur_match_i, n))
#     if n_max_pairs is not None:
#         random.seed(1)  # use the same list across different models
#         random.shuffle(match_list)
#         match_list = match_list[:n_max_pairs]
#     if both_directions:
#         return match_list + [(i[1], i[0]) for i in match_list]
#     else:
#         return match_list
def get_pair_list(labels, options_dict, both_directions=True, n_max_pairs=None):
    """Return a list of tuples giving indices of matching types."""
    N = len(labels)
    match_list = []

    # Check if list with current restrictions exist (n_min_tokens_per_type, n_max_tokens_per_type, n_max_types)
    # pair_list_fn = path.join("pair_lists", options_dict["train_lang"], str(options_dict["n_min_tokens_per_type"]) + "." + str(options_dict["n_max_types"]) + ".npz")
    pair_list_fn = ""
    print("Pair list final:", pair_list_fn)

    if os.path.isfile(pair_list_fn):
        print("Use existing pair list")
        pair_list_npz = np.load(pair_list_fn)
#            print(pair_list_npz["arr_0"])
        print(len(pair_list_npz["arr_0"]))
        match_list = pair_list_npz["arr_0"]

    else:
        for n in tqdm(range(N - 1)):
            cur_label = labels[n]
            for cur_match_i in (n + 1 + np.where(np.asarray(labels[n + 1:]) ==
                    cur_label)[0]):
                match_list.append((n, cur_match_i))
                # if both_directions:
                #     match_list.append((cur_match_i, n))

        # Save match list (with current n_min_tokens_per_type, n_max_tokens_per_type, n_max_types)
        pair_list_dir = path.join("pair_lists", options_dict["train_lang"])
        if not os.path.isdir(pair_list_dir):
            os.makedirs(pair_list_dir)

        pair_list_fn = os.path.join(pair_list_dir, str(options_dict["n_min_tokens_per_type"]) + "."  + str(options_dict["n_max_types"]) + ".npz")
        print("Writing pair list to:", pair_list_fn)
                
        if both_directions:
            pair_list = match_list + [(i[1], i[0]) for i in match_list]
        else:
            pair_list = match_list

        # np.savez(pair_list_fn, pair_list)

    print("Done")
    if n_max_pairs is not None:
        random.seed(1)  # use the same list across different models
        random.shuffle(match_list)
        match_list = match_list[:n_max_pairs]
    if both_directions:
        return match_list + [(i[1], i[0]) for i in match_list]
    else:
        return match_list


def get_tuplet_lists(pair_list, labels, n_items):
    batch_idxs, batch_hot = list(), list()

    for pair in pair_list:
        mini_batch_idxs, mini_batch_hot = list(), list()
        mini_batch_idxs.extend(pair)
        mini_batch_hot.extend([1, 1])
        while len(mini_batch_idxs) < n_items:
            rand_idx = np.random.randint(0, len(labels))
            if pair[0] != labels[rand_idx] and rand_idx not in mini_batch_idxs:
                mini_batch_idxs.append(rand_idx) 
                mini_batch_hot.append(0)
        batch_idxs.append(mini_batch_idxs)
        batch_hot.append(mini_batch_hot)

    return batch_idxs, batch_hot

def tuplets_shuffle(tuplet_list_idxs, tuplet_list_hot):
    batch_idxs_shuffled, batch_hot_shuffle = list(), list() 
    for x, y in zip(tuplet_list_idxs, tuplet_list_hot):
        temp = list(zip(x, y))
        random.shuffle(temp)
        res1, res2 = zip(*temp)
        batch_idxs_shuffled.append(list(res1))
        batch_hot_shuffle.append(list(res2))

    # print("\nAfter shuffle:")
    # print(batch_idxs_shuffled[0])
    # print(batch_hot_shuffle[0])

    return batch_idxs_shuffled, batch_hot_shuffle

def get_minibatch(pairs, labels, N, n_minibatches):
    batch = []
    for i, positive_pair in tqdm(enumerate(pairs)):
        # print(i)
        # print("len pairs", len(pairs))
        minibatch = []
        minibatch.append(positive_pair)
        # print(positive_pair)
        positive_label = labels[positive_pair[0]]
        # print(positive_label)

        minibatch_labels = []
        for pair in pairs[i+1:]:
            # print(pair)
            pair_label = labels[pair[0]]
            # print(pair_label)
            if pair_label != positive_label and pair_label not in minibatch_labels:
                # print(pair)
                # print(pair_label)
                minibatch.append(pair)
                # print(minibatch)
                minibatch_labels.append(pair_label)
                # print(minibatch_labels)
                pairs.remove(pair)
                # del pairs[labels.index(pair)]
            if len(minibatch) == N:
                batch.append(minibatch)
                # print("break")
                break

        # if len(minibatch) == N:
            # batch.append(minibatch)

        if n_minibatches is not None:
            if len(batch) == n_minibatches:
                break

    print("No. minibatches:", len(batch))
    return batch

