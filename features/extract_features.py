
"""
Extract MFCC features for a particular NCHLT language.

Modified from Herman Kamper
"""

import os
from os import path
import sys
from tqdm import tqdm
import codecs
import glob
import numpy as np
import shutil
import features
import utils
import argparse

sys.path.append("..")
from paths import nchlt_data_dir, nchlt_alignments_dir

from process_alignments import nchlt_languages


def check_argv():
    """Check the command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__.strip().split("\n")[0], add_help=False
        )
    parser.add_argument(
        "language", type=str, help="NCHLT language",
        choices=nchlt_languages
        )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def link_wav_for_subset(language, speakers, wav_dir):
    # print(speakers)
    for speaker in speakers:
        speaker_wav_dir_list = glob.glob(path.join(nchlt_data_dir, "nchlt_" + language, "audio", speaker, "*.wav"))
        for speaker_wav_fn in speaker_wav_dir_list:
            s = speaker_wav_fn.split("/")[-1].split("_")
            
            link_fn = path.join(wav_dir, (s[1]+s[2]+"_"+s[3]))
            assert (
                    path.isfile(speaker_wav_fn)
                    ), "missing file: {}".format(speaker_wav_fn)
            if not path.isfile(link_fn):
                # print("Linking:", speaker_wav_fn, "to", link_fn)
                os.symlink(speaker_wav_fn, link_fn)


def read_speakers(speakers_fn, language):
    with open(speakers_fn) as f:
        for line in f:
            line = line.strip().split()
            # print(line)
            if line[0] == language:
                return line[1:]
    assert False, "invalid language"


def extract_features_for_subset(language, subset, feat_type, output_fn):
    """
    Extract features for the subset in this language.

    The `feat_type` parameter can be "mfcc" or "fbank".
    """

    # Get speakers for subset
    speakers_fn = path.join("..", "data", subset + "_spk.list")
    print("Reading:", speakers_fn)
    speakers = read_speakers(speakers_fn, language)
    # print(speakers)
    
    # link wav from speakers for each subset
    wav_dir = path.join("wav", language, subset)
    if not path.isdir(wav_dir):
        os.makedirs(wav_dir)
    print("Linking wav files:", wav_dir)
    link_wav_for_subset(language, speakers, wav_dir)
    

    # Extract raw features
    print("Extracting features:")
    if feat_type == "mfcc":
        feat_dict = features.extract_mfcc_dir(wav_dir)
    elif feat_type == "fbank":
        feat_dict = features.extract_fbank_dir(wav_dir)
    else:
        assert False, "invalid feature type"

    # Perform per speaker mean and variance normalisation
    print("Per speaker mean and variance normalisation:")
    feat_dict = features.speaker_mvn(feat_dict)

    # Write output
    print("Writing:", output_fn)
    np.savez_compressed(output_fn, **feat_dict)


def main():
    args = check_argv()
    feat_type = "mfcc"

    # Extract MFCCs for the different sets
    feat_dir = path.join(feat_type, args.language)
    if not path.isdir(feat_dir):
        os.makedirs(feat_dir)

    for subset in ["dev", "test", "train"]:
        raw_feat_fn = path.join(
            feat_dir, args.language.lower() + "." + subset + ".npz"    
            )
        if not path.isfile(raw_feat_fn):
            print("Extracting MFCCs:", subset)
            extract_features_for_subset(
                args.language, subset, feat_type, raw_feat_fn
                )
        else:
            print("Using existing file:", raw_feat_fn)
            
        
    # GROUND TRUTH WORD SEGMENTS
    list_dir = path.join("lists", args.language)
    if not path.isdir(list_dir):
        os.makedirs(list_dir)
    for subset in ["dev", "test", "train"]:

        # Create a ground truth word list (at least 50 frames and 5 characters)
        fa_fn = path.join(nchlt_alignments_dir, args.language, subset + ".ctm")
        list_fn = path.join(list_dir, subset + ".gt_words.list")
        if not path.isfile(list_fn):
            if args.language == "nso":
                min_frames = 30
                min_chars = 3
            elif args.language == "tsn":
                min_frames = 40
                min_chars = 4
            elif args.language == "sot":
                min_frames = 30
                min_chars = 3
            elif args.language == "tso":
                min_frames = 30
                min_chars = 3
            elif args.language == "ven":
                min_frames = 30
                min_chars = 3
            else:
                min_frames = 50
                min_chars = 5
            utils.filter_words(fa_fn, list_fn, min_frames=min_frames, min_chars=min_chars)
        else:
            print("Using existing file:", list_fn)

        # Extract word segments from the MFCC NumPy archives
        input_npz_fn = path.join(
            feat_dir, args.language.lower() + "." + subset + ".npz"
            )
        output_npz_fn = path.join(
            feat_dir, args.language.lower() + "." + subset + ".gt_words.npz"
            )
        if not path.isfile(output_npz_fn):
            print("Extracting MFCCs for ground truth word tokens:", subset)
            utils.segments_from_npz(input_npz_fn, list_fn, output_npz_fn)
        else:
            print("Using existing file:", output_npz_fn)


if __name__ == "__main__":
    main()
