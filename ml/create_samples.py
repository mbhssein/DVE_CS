import os
import sys
import csv

import cv2

sys.path.insert(0, os.path.abspath('..'))

import cs_utils

def create_samples(directory):
    out_fp = open(os.path.join(directory, 'samples.csv'), 'w')
    writer = csv.writer(out_fp)

    for filename in os.listdir(directory):
        filename_low = filename.lower()
        if filename_low.endswith('.png') or filename_low.endswith('.jpg') or filename_low.endswith('.jpeg'):
            print 'Processing: ' + filename
            image = cv2.resize(cv2.imread(os.path.join(directory, filename), 0), (20, 20))
            coefficients = cs_utils.fourier_coeffs(image, 100)
            writer.writerow(coefficients)


def main(argv):
    """
    :param argv: list of command line arguments
    """
    if len(argv) == 1:
        print "Not enough arguments\nUsage: python create_samples.py <path to samples dir>"
        return
    else:
        path = argv[1]
        assert os.path.isdir(path)

        create_samples(path)

if __name__ == '__main__':
    main(sys.argv)
