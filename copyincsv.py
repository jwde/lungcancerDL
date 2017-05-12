import sys
import csv
from shutil import copyfile
import os

def main(args):
    lines = []
    with open(args[1]) as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            lines += [row]
    lines = lines[1:]
    
    for id, label, usage in lines:
        copyfile(os.path.join(args[2], id + '.npy'), os.path.join(args[3], id + '.npy'))

if __name__ == '__main__':
    main(sys.argv)
