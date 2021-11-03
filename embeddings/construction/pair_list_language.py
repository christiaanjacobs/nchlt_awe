"Create pair list per language"
import numpy as np
from dataset import GlobalPhone
import random
from sampler import pair_sampler
from os import path
import data_process
import pickle
import tqdm as tqdm
import argparse
import sys
import os

default_options_dict = {
        'train_lang': None,                 # language code

        'n_max_pairs': None,
        'n_min_tokens_per_type': None,         
        'n_max_types': None,
        'n_max_tokens': None,
        'rnd_seed': 1,

    }

def lang_pair_list(labels, offset, options_dict):

    pair_list = get_pair_list(
                    labels, offset, options_dict, both_directions=True,
                n_max_pairs=options_dict["n_max_pairs"]
                )
    if options_dict["n_max_pairs"] is None:
        random.seed(1)
        # random.shuffle(pair_list)

    print("No. Pairs:", len(pair_list)//2)
    # print(pair_list)
    # print(len(labels))
    
    return pair_list



def load_language(train_lang, options_dict):

        data_root = "../data"        
        data_dir = path.join(data_root, train_lang, "train." + "gt" + ".npz")
            
     
        x, labels, lengths, keys, speakers = data_process.load_data(data_dir)
        x, labels, lengths, keys, speakers = data_process.filter_data(x, labels, lengths, keys, speakers,
                                                                        n_min_tokens_per_type=options_dict["n_min_tokens_per_type"],
                                                                        n_max_types=options_dict["n_max_types"],
                                                                        n_max_tokens=options_dict["n_max_tokens"])
        return labels                                                 


def get_pair_list(labels, offset, options_dict, both_directions=True, n_max_pairs=None):
    """Return a list of tuples giving indices of matching types."""
    # print(labels)
    N = len(labels)
    match_list = []
    for n in range(N - 1):
        cur_label = labels[n]
        for cur_match_i in (n + 1 + np.where(np.asarray(labels[n + 1:]) ==
                cur_label)[0]):
            match_list.append(((n+offset), (cur_match_i+offset)))
            # if both_directions:
            #     match_list.append((cur_match_i, n))
    if n_max_pairs is not None:
        random.seed(1)  # use the same list across different models
        random.shuffle(match_list)
        match_list = match_list[:n_max_pairs]
    if both_directions:
        return match_list + [(i[1], i[0]) for i in match_list]
    else:
        return match_list


def check_argv():
    parser = argparse.ArgumentParser(
        description=__doc__.strip().split("\n")[0], add_help=False
        )
    parser.add_argument(
        "--train_lang", type=str,
        help="training set tag (default: %(default)s)",
        default=default_options_dict["train_lang"]
        )
    parser.add_argument(
        "--n_max_pairs", type=int,
        help="maximum number of same-word pairs to use (default: %(default)s)",
        default=default_options_dict["n_max_pairs"]
        )
    parser.add_argument(
        "--n_min_tokens_per_type", type=int,
        help="minimum number of tokens per type (default: %(default)s)",
        default=default_options_dict["n_min_tokens_per_type"]
        )
    parser.add_argument(
        "--n_max_types", type=int,
        help="maximum number of types per language (default: %(default)s)",
        default=default_options_dict["n_max_types"]
        )
    parser.add_argument(
        "--n_max_tokens", type=int,
        help="maximum number of tokens per language (default: %(default)s)",
        default=default_options_dict["n_max_tokens"]
        )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()

if __name__ == "__main__":

    args = check_argv()
    options_dict = default_options_dict
    options_dict["train_lang"] = args.train_lang
    options_dict["n_max_pairs"] = args.n_max_pairs
    options_dict["n_min_tokens_per_type"] = args.n_min_tokens_per_type
    options_dict["n_max_types"] = args.n_max_types
    options_dict["n_max_tokens"] = args.n_max_tokens

    random.seed(options_dict['rnd_seed'])
    np.random.seed(options_dict['rnd_seed'])

    list_dir = path.join("pair_lists", options_dict["train_lang"])
    if not os.path.isdir(list_dir):
        os.makedirs(list_dir)
    
    pair_list_final = []
    offset = 0 
    for lang in options_dict["train_lang"].split("+"):
        labels = load_language(lang, options_dict)
        pair_list = lang_pair_list(labels, offset, options_dict)
        offset = offset + len(labels)
        # print(pair_list)
        pair_list_final.extend(pair_list)

    pair_list_final_fn = path.join("pair_lists", options_dict["train_lang"], 
                                    str(options_dict['n_max_pairs']) + "." +
                                    str(options_dict['n_min_tokens_per_type']) + "." +
                                    str(options_dict['n_max_types']) + "." +
                                    str(options_dict['n_max_tokens']) + "." + "pkl")

    print("Writing: ", pair_list_final_fn)
    with open(pair_list_final_fn, "wb") as f:
        pickle.dump(pair_list_final, f)

    

