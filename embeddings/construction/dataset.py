import numpy as np
import torch
from torch.utils.data import Dataset
from random import randint
from os import path
from tqdm import tqdm

import data_process



class GlobalPhone(Dataset):
    def __init__(self, mode, options_dict):        
        self.mode = mode
        data_root = '../data'

        data_dir = []
        if self.mode == 'train':
            tag = options_dict['train_tag']

            if '+' in options_dict['train_lang']:
                train_languages = options_dict['train_lang'].split('+')
                for train_lang in train_languages:
                    cur_dir = path.join(data_root, train_lang, 'train.' + tag + '.npz')
                    data_dir.append(cur_dir)        
            else:
                train_lang = options_dict['train_lang']
                data_dir.append(path.join(data_root, train_lang, 'train.' + tag + '.npz'))


            if options_dict["remove_overlap"] is not None:
                overlap_languages = options_dict["remove_overlap"].split("+")
                overlap_labels = set()
                for lang in overlap_languages:
                    lang_list_fn = path.join('..', '..', 'features', 'lists', lang, 'train.gt_words.list')
                    with open(lang_list_fn, "r") as f:
                        for line in f.readlines():
                            overlap_labels.add(line.strip().split("_")[0])
                # print(english_labels[:10])


        if self.mode == 'val':
            val_lang = options_dict['val_lang']
            data_dir.append(path.join(data_root, val_lang, 'val.npz'))
            
            if options_dict["remove_overlap"] is not None:
                # overlap_languages = options_dict["remove_overlap"].split("+")
                overlap_languages = options_dict["remove_overlap"].split("+")
                overlap_labels = set()
                for lang in overlap_languages:
                    lang_list_fn = path.join('..', '..', 'features', 'lists', lang, 'dev.gt_words.list')
                    with open(lang_list_fn, "r") as f:
                        for line in f.readlines():
                            overlap_labels.add(line.strip().split("_")[0])
            # overlap_labels = set()

        self.x = []
        self.labels = []
        self.lengths = []
        self.keys = []
        self.speakers = []
     
        for i, cur_dir in enumerate(data_dir):
            cur_x, cur_labels, cur_lengths, cur_keys, cur_speakers = data_process.load_data(cur_dir, overlap_labels=overlap_labels)
            if self.mode == 'train':
                cur_x, cur_labels, cur_lengths, cur_keys, cur_speakers = data_process.filter_data(cur_x, cur_labels, cur_lengths, cur_keys, cur_speakers,
                                                                            n_min_tokens_per_type=options_dict["n_min_tokens_per_type"],
                                                                            n_max_types=options_dict["n_max_types"],
                                                                            n_max_tokens=options_dict["n_max_tokens"])
                                                            
        
            self.x.extend(cur_x)
            self.labels.extend(cur_labels)
            self.lengths.extend(cur_lengths)
            self.keys.extend(cur_keys)
            self.speakers.extend(cur_speakers) # list ['GE034', 'GE045', ..., 'GE12', RU087', 'RU012', 'RU012', ..., 'RU020']
        

        # convert labels to int for train classifier rnn
        if self.mode == 'train':
            self.labels_int = self.convert_labels(options_dict)
        
        self.trunc_and_limit(options_dict)


    def convert_labels(self, options_dict):
        # Convert training labels to integers
        train_label_set = list(set(self.labels))
        label_to_id = {}
        for i, label in enumerate(sorted(train_label_set)):
            label_to_id[label] = i
        train_y = []
        for label in self.labels:
            train_y.append(label_to_id[label])
        train_y = np.array(train_y, dtype=np.int32)
        print(train_y)
        options_dict["n_classes"] = len(label_to_id)
        print("Total no. classes:", options_dict["n_classes"])
        return train_y

    def trunc_and_limit(self, options_dict):
        max_length = options_dict["max_length"]
        d_frame = 13  # None
        options_dict["n_input"] = d_frame
        print("Limiting dimensionality:", d_frame)
        print("Limiting length:", max_length)
        data_process.trunc_and_limit_dim(self.x, self.lengths, d_frame, max_length)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):

        if self.mode == 'train':
            idx_list = [] 
            for pair in idx:
                idx_list.extend(list(pair))
            x = []
            y = []
            for i in idx_list:
                x.append(self.x[i])
                y.append(self.labels_int[i])
            return x, y

        if self.mode == 'val':
            return self.x[idx], self.labels[idx], self.lengths[idx], self.keys[idx], self.speakers[idx]