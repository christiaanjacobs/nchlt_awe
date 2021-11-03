
"""
Perform same-different evaluation of fixed-dimensional representations.
Author: Herman Kamper
Contact: kamperh@gmail.com
Date: 2015, 2016, 2018, 2019
"""

from datetime import datetime
from os import path
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import argparse
import numpy as np
import sys

sys.path.append(path.join("..", "..", "..", "src", "speech_dtw", "utils"))

import samediff


#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#

def check_argv():
    """Check the command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__.strip().split("\n")[0], add_help=False
        )
    parser.add_argument("npz_fn", type=str, help="NumPy archive")
    parser.add_argument(
        "--metric", choices=["cosine", "euclidean", "hamming", "chebyshev",
        "kl"], default="cosine", help="distance metric (default: %(default)s)"
        )
    parser.add_argument(
        "--mean_ap", dest="mean_ap", action="store_true",
        help="also compute mean average precision (this is significantly "
        "more resource intensive)"
        )
    parser.add_argument(
        "--mvn", action="store_true",
        help="mean and variance normalise (default: False)"
        )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    args = check_argv()

    print(datetime.now())

    print("Reading:", args.npz_fn)
    npz = np.load(args.npz_fn)

    print(datetime.now())

    print("Ordering embeddings")
    n_embeds = 0
    X = []
    ids = []
    for label in sorted(npz):
        ids.append(label)
        X.append(npz[label])
        n_embeds += 1
    X = np.array(X)
    print("No. embeddings:", n_embeds)
    print("Embedding dimensionality:", X.shape[1])

    if args.mvn:
        normed = (X - X.mean(axis=0)) / X.std(axis=0)
        X = normed

    print(datetime.now())

    print("Calculating distances")
    metric = args.metric
    if metric == "kl":
        import scipy.stats
        metric = scipy.stats.entropy
    distances = pdist(X, metric=metric)
    print("Distances shape:", distances.shape)
    squareform_distances = squareform(distances)
    print("Squareform distances shape", squareform_distances.shape)
    print("Example of first row:", squareform_distances[0])

    print(datetime.now())

    print("Getting labels and speakers")
    labels = []
    speakers = []
    durations = []
    for utt_id in ids:
        utt_id = utt_id.split("_")
        word = utt_id[0]
        speaker = utt_id[1]
        time_frame = utt_id[3]
        time_frame = time_frame.split("-")
        duration = int(time_frame[1]) - int(time_frame[0])
        # print(duration)
        labels.append(word)
        speakers.append(speaker)
        durations.append(duration)

    # print("Labels", labels[:10])
    # print("Duration", durations[:10])

    total_duration_difference = np.zeros(len(labels))
    for i, row in enumerate(squareform_distances):
        sorted_idx = np.argsort(row)
        # print(sorted_idx)
        labels_sorted = [labels[idx] for idx in sorted_idx]
        duration_sorted = [durations[idx] for idx in sorted_idx]
        # print(labels[:20])
        # print(duration[:20])
        duration_difference = np.array(duration_sorted) - duration_sorted[0] 
        total_duration_difference += duration_difference
        # print(duration_difference[:10])
        # print(labels_sorted[:10])
        # print(row[sorted_idx][:10])
    print(total_duration_difference/len(labels))

    from numpy import savetxt
    savetxt('cae_multi_sw.csv', total_duration_difference/len(labels), delimiter=",")
        


    # if args.mean_ap:
    #     print(datetime.now())
    #     print("Calculating mean average precision")
    #     mean_ap, mean_prb, ap_dict = samediff.mean_average_precision(
    #         distances, labels
    #         )
    #     print("Mean average precision:", mean_ap)
    #     print("Mean precision-recall breakeven:", mean_prb)

    # print(datetime.now())

    # print("Calculating average precision")
    # # matches = samediff.generate_matches_array(labels)  # Temp
    # word_matches = samediff.generate_matches_array(labels)
    # speaker_matches = samediff.generate_matches_array(speakers)
    # print("No. same-word pairs:", sum(word_matches))
    # print("No. same-speaker pairs:", sum(speaker_matches))
    
    # sw_ap, sw_prb, swdp_ap, swdp_prb = samediff.average_precision_swdp(
    #     distances[np.logical_and(word_matches, speaker_matches)],
    #     distances[np.logical_and(word_matches, speaker_matches == False)],
    #     distances[word_matches == False]
    #     )
    # print("-"*79)
    # print("Average precision: {:.8f}".format(sw_ap))
    # print("Precision-recall breakeven: {:.8f}".format(sw_prb))
    # print("SWDP average precision: {:.8f}".format(swdp_ap))
    # print("SWDP precision-recall breakeven: {:.8f}".format(swdp_prb))
    # print("-"*79)

    print(datetime.now())


if __name__ == "__main__":
    main()