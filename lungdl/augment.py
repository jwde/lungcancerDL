import util
import argparse
import os
import time
import explore

def test_load_time(args):
    since = time.time()
    data = util.get_data(args.INPUT_FOLDER, args.LABELS_FILE, 10)
    for i,t in enumerate(data["train"]):
        pass
    for i,t in enumerate(data["train"]):
        pass
    time_elapsed = time.time() - since
    print("Done, time_elapsed =", time_elapsed)

def main(args):
    if not os.path.exists(args.OUT_FOLDER):
        os.makedirs(args.OUT_FOLDER)
    since = time.time()
    data = util.LabeledKaggleDataset(args.INPUT_FOLDER, args.LABELS_FILE)
    img, t = data.__getitem__(0)
    explore.plot(img)
    time_elapsed = time.time() - since
    print("Done, time_elapsed =", time_elapsed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('INPUT_FOLDER', help="Path to input data directory")
    parser.add_argument('LABELS_FILE', help="Path to input data directory")
    parser.add_argument('OUT_FOLDER', help="Path to output data directory")
    r = parser.parse_args()
    main(r)
