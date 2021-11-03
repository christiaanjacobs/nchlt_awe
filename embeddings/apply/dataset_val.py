import numpy as np
import torch
from torch.utils.data import Dataset
from random import randint
from os import path
from tqdm import tqdm
import sys

sys.path.append(path.join(".."))
import data_io



class GlobalPhone(Dataset):
    def __init__(self, options_dict):        
        data_root = '../data'
        data_dir = path.join(data_root, options_dict['val_lang'], 'test.npz')
        print('Loading validation data from ', data_dir)



          # if options_dict["remove_overlap"] is not None:
        # overlap_languages = "xho+zul+ssw+nbl+tsn+sot+afr+eng+tso+ven".split("+")
        overlap_languages = ["ssw", "eng"]
        overlap_labels = set()
        for lang in overlap_languages:
            lang_list_fn = path.join('..', '..', 'features', 'lists', lang, 'test.gt_words.list')
            with open(lang_list_fn, "r") as f:
                for line in f.readlines():
                    overlap_labels.add(line.strip().split("_")[0])


        self.x = []
        self.labels = []
        self.lengths = []
        self.keys = []
        self.speakers = []
        
        self.x, self.labels, self.lengths, self.keys, self.speakers = data_io.load_data_from_npz(data_dir, overlap_labels=overlap_labels)

        self.trunc_and_limit(options_dict)

    def trunc_and_limit(self, options_dict):
        # Truncate and limit dimensionality
        max_length = options_dict["max_length"]
        d_frame = 13  # None
        options_dict["n_input"] = d_frame
        print("Limiting dimensionality:", d_frame)
        print("Limiting length:", max_length)
        data_io.trunc_and_limit_dim(self.x, self.lengths, d_frame, max_length)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.labels[idx], self.lengths[idx], self.keys[idx], self.speakers[idx]
