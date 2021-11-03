from torch.utils.data import Sampler
import numpy as np
import random


class PairedSampler(Sampler):
    # TODO shuffle pairs
    """
    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source, options_dict):
        self.data_source = data_source

        # get_pair_list shuffles pairs given n_max_pairs
        self.pair_list = get_pair_list(
            data_source.labels, both_directions=True,
            n_max_pairs=options_dict["n_max_pairs"]
            )
        if options_dict["n_max_pairs"] is None:
            random.seed(options_dict['rnd_seed'])
            random.shuffle(self.pair_list)

        print("Total pairs: ", len(self.pair_list))
    
    def __len__(self):
        return len(self.pair_list)

    def __iter__(self):
        return iter(self.pair_list)




class SiameseSampler(Sampler):
    # TODO shuffle pairs
    """
    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source, options_dict):
        self.data_source = data_source

        # get_pair_list shuffles pairs given n_max_pairs
        self.pair_list = get_pair_list(
            data_source.labels, both_directions=True,
            n_max_pairs=options_dict["n_max_pairs"]
            )
        if options_dict["n_max_pairs"] is None:
            random.seed(options_dict['rnd_seed'])
            random.shuffle(self.pair_list)

        print(len(self.pair_list))
        self.flat_list = [item for sublist in self.pair_list for item in sublist]
        
        self.flat_list = self.flat_list[:len(self.flat_list)//options_dict['batch_size']*options_dict['batch_size']]
        self.flat_list = np.reshape(self.flat_list, (-1, options_dict['batch_size']))
        print(self.flat_list.shape)
            
    def __len__(self):
        return len(self.flat_list)

    def __iter__(self):
        return iter(self.flat_list)




#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#


def get_pair_list(labels, both_directions=True, n_max_pairs=None):
    """Return a list of tuples giving indices of matching types."""
    N = len(labels)
    match_list = []
    for n in range(N - 1):
        cur_label = labels[n]
        for cur_match_i in (n + 1 + np.where(np.asarray(labels[n + 1:]) ==
                cur_label)[0]):
            match_list.append((n, cur_match_i))
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
