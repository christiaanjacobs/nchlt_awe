import os
from os import path
import sys
from process_alignments import nchlt_languages

sys.path.append("..")
from paths import nchlt_data_dir, nchlt_alignments_dir

def main(languages):
    for subset in ["dev", "test", "train"]:
        speaker_fn = path.join("..", "data", subset + "_spk.list")
        # Delete previous speaker lists
        if os.path.exists(speaker_fn):
            os.remove(speaker_fn)
        # Append all speakers for each language in new list
        for language in languages:
            subset_alignment_fn = path.join(nchlt_alignments_dir, language, subset + ".ctm")
            speakers = set()
            with open(subset_alignment_fn, "r") as f:
                for line in f:
                    print(line.strip())
                    s = line.strip().split("_")
                    speakers.add(s[0][len(language):-1])

            with open(speaker_fn, "a") as f:
                f.write(language + " ")
                for speaker in sorted(speakers):
                    f.write(speaker + " ")
                f.write("\n")


if __name__ == "__main__":
    main(nchlt_languages)





