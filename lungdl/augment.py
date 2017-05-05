import util
import argparse
import os
import time

def main(args):
    if not os.path.exists(args.OUT_FOLDER):
        os.makedirs(args.OUT_FOLDER)
    since = time.time()
    #data = util.LabeledKaggleDataset(args.INPUT_FOLDER, args.LABELS_FILE)
    data = util.get_data(args.INPUT_FOLDER, args.LABELS_FILE, 1)
    for t in data["train"]:
        pass
    for t in data["val"]:
        pass
    time_elapsed = time.time() - since
    print("Done, time_elapsed =", time_elapsed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('INPUT_FOLDER', help="Path to input data directory")
    parser.add_argument('LABELS_FILE', help="Path to input data directory")
    parser.add_argument('OUT_FOLDER', help="Path to output data directory")
    r = parser.parse_args()
    main(r)
