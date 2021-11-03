#!/home/christiaan/anaconda3/envs/myenv/bin/python
"""
Encode the given NumPy archive using the specified model.
"""

from datetime import datetime
from os import path
import argparse
import pickle
import numpy as np
import os
import sys
from scipy.spatial.distance import pdist
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
# sys.path.append(path.join('..', '..' , '..', 'src', 'speech_dtw', 'utils'))
import samediff_hacked
# sys.path.append(path.join("../construction"))
# sys.path.append(os.getcwd())
# sys.path.append(os.path.join(os.path.dirname(__file__), "../data_io"))
sys.path.append(path.abspath(path.join(path.dirname(__file__), "..")))


# from apply_model import build_model
# from link_mfcc import sixteen_languages
# import batching
import data_io
sys.path.append(path.join("..", "train_ae_cae_rnn"))
import models

#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#

def check_argv():
    """Check the command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__.strip().split("\n")[0], add_help=False
        )
    parser.add_argument("model_fn", type=str, help="model checkpoint filename")
    parser.add_argument("npz_fn", type=str, help="the NumPy archive to encode")
    parser.add_argument(
        "--output_npz_fn", type=str,
        help="if provided, the output is written to this NumPy archive "
        "instead of the model directory"
        )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def collate_fn(batch):
    """
    batch = (data, lengths), data is shape (seq_len, feat_dim)
    """
    seqs = [x[0] for x in batch]
    batch_seqs_padded = pad_sequence(seqs, batch_first=True) # pad variable length sequences to max seq length
    lengths = [x[1] for x in batch]
    batch_lengths = torch.LongTensor(lengths)
    keys = [x[2] for x in batch]
    labels = [x[3] for x in batch]
    speakers = [x[4] for x in batch]
    return batch_seqs_padded, batch_lengths, keys, labels, speakers


#-----------------------------------------------------------------------------#
#                                   Dataset                                   #
#-----------------------------------------------------------------------------#
class npz_data(Dataset):
    def __init__(self, x_data, lengths, keys, labels, speakers):
        self.x_data = x_data
        self.lengths = lengths
        self.keys = keys
        self.labels = labels
        self.speakers = speakers

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return self.x_data[idx], self.lengths[idx], self.keys[idx], self.labels[idx], self.speakers[idx]


#-----------------------------------------------------------------------------#
#                            APPLY MODEL FUNCTIONS                            #
#-----------------------------------------------------------------------------#

def apply_model(model_fn, npz_fn):
    # Load model options
    model_dir = path.split(model_fn)[0]
    options_dict_fn = path.join(model_dir, "options_dict.pkl")
    print("Reading:", options_dict_fn)
    with open(options_dict_fn, "rb") as f:
        options_dict = pickle.load(f)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    print('Device:', device)  
    options_dict['device'] = device

    # Load data
    x_data, labels, lengths, keys, speakers = data_io.load_data_from_npz(npz_fn)
    print(keys)
    # Truncate and limit dimensionality
    data_io.trunc_and_limit_dim(x_data, lengths, options_dict["n_input"], options_dict["max_length"])

    # # Load model
    model = models.cae_rnn(options_dict)
    # model.load_state_dict(torch.load(model_fn)["state_dict"], strict=False)
    model.load_state_dict(torch.load(model_fn), strict=True)
    model.to(device)
    model.eval()

    # # Encapuslate data in Dataset and apply Dataloader
    dataset = npz_data(x_data, lengths, keys, labels, speakers)
    dataloader = DataLoader(dataset, batch_size=len(dataset),
        shuffle=False, drop_last=False, collate_fn=collate_fn)

    # Iterate
    for batch_seqs_padded, batch_lengths, batch_keys, batch_labels, batch_speakers in dataloader:
        batch_seqs_padded, batch_lengths = batch_seqs_padded.to(device), batch_lengths.to(device)
        np_z = model(batch_seqs_padded, batch_lengths).cpu().detach().numpy()
        break # Single batch

    print(batch_keys[-1], print(batch_labels[-1]))

    # Samediff
    def same_diff_analysis(np_z, labels_val, speakers_val, keys_val, normalise=True):        
        embed_dict = {}
        for i, utt_key in enumerate(keys_val):
            embed_dict[utt_key] = np_z[i]
        
        if normalise:
            np_z_normalised = (np_z - np_z.mean(axis=0))/(np_z.std(axis=0))
            distances = pdist(np_z_normalised, metric='cosine')
        
        word_matches = samediff_hacked.generate_matches_array(labels_val)
        speaker_matches = samediff_hacked.generate_matches_array(speakers_val)
        print(word_matches)

        sw_ap, sw_prb, swdp_ap, swdp_prb = samediff_hacked.average_precision_swdp(
            distances[np.logical_and(word_matches, speaker_matches)],
            distances[np.logical_and(word_matches, speaker_matches == False)],
            distances[word_matches == False]
            )
        # return [sw_prb, -sw_ap, swdp_prb, -swdp_ap]
        return None

    val_loss = same_diff_analysis(np_z, batch_labels, batch_speakers, batch_keys)



    embed_dict = {}
    for i, utt_key in enumerate(batch_keys):
        embed_dict[utt_key] = np_z[i]

    # return None
#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    print(sys.path)
    # args = check_argv()

    model_fn = "../models/RU+CZ+FR+PL+TH+PO.gt/train_cae_rnn/8415c0ad94/final_model.pt"
    npz_fn = "../data/SP/val.npz"
    output_npz_fn = None
    
    # Embed data
    embed_dict = apply_model(model_fn, npz_fn)

    # # Save embeddings
    # model_dir, model_fn = path.split(args.model_fn)
    # if args.output_npz_fn is None:
    #     npz_fn = path.join(
    #         model_dir, path.splitext(args.model_fn)[0] + "." +
    #         path.split(args.npz_fn)[-1]
    #         )
    # else:
    #     npz_fn = args.output_npz_fn
    # print("Writing:", npz_fn)
    # np.savez_compressed(npz_fn, **embed_dict)
    # print(datetime.now())


if __name__ == "__main__":
    main()
