import os
from os import path
import sys
import numpy as np
import argparse

sys.path.append("..")
from paths import nchlt_data_dir, nchlt_alignments_dir

nchlt_languages = ["nbl", "nso", "sot", "ssw", "tsn", "tso", "ven", "xho", "zul", "afr", "eng"]


def main(languages):
    """
    Removes empty transcriptions and add all subsets to language folder
    """

    for language in languages:

        for subset in ["dev", "test", "train"]:
            if subset is "train":
                alignment_fn = path.join(
                    nchlt_alignments_dir, "nchlt_9languages_wordlevel_ctm_trn_set", "nchlt_" + language, "tri3_ali", "ctm"
                    )
            else:
                alignment_fn = path.join(
                    nchlt_alignments_dir, "nchlt_9languages_wordlevel_ctm_dev_test_sets", "nchlt_" + language, "tri3_ali_" + subset, "ctm"
                    )

            print("Reading:", alignment_fn)
            alignments = set()
            with open(alignment_fn) as f:
                for line in f:
                    if "<eps>" not in line.strip() and "[s]" not in line.strip():
                        s = line.split()
                        s2 = s[0].split("_")
                        alignments.add(s2[1]+s2[2]+"_"+s2[3]+" "+s[1]+" "+s[2]+" "+s[3]+" "+s[4])

            alignments_output_dir = path.join(nchlt_alignments_dir, language)
            if not path.isdir(alignments_output_dir):
                os.makedirs(alignments_output_dir)

            
            subset_alignments_output_fn = path.join(alignments_output_dir, subset + ".ctm")
            print("Output:", subset_alignments_output_fn)
            with open(subset_alignments_output_fn, "w") as set_f:
                for alignment in sorted(alignments):
                    set_f.write(alignment + "\n")  


if __name__ == "__main__":

    main(nchlt_languages)
