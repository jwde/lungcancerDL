import util
import argparse
import os
import time

def main(args):
    if not os.path.exists(args.OUT_FOLDER):
        os.makedirs(args.OUT_FOLDER)
    data = util.LabeledKaggleDataset(args.INPUT_FOLDER, args.LABELS_FILE)
    since = time.time()
    for i,(img,target) in enumerate(data):
        #print (i, target)
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
