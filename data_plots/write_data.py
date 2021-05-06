#!/usr/bin/env python3

# before we do much of anything, disable writing bytecode, as that suffers
#  from race conditions with distributed jobs on shared file systems
import sys
sys.dont_write_bytecode = True

import numpy as np
#import tensorflow as tf
#from tensorflow.keras import models, layers, optimizers
import reader_fp
from reader_fp_piersall import DataReader
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data', default='/gpfs/alpine/syb105/proj-shared/Projects/NV_ORNL_XAIClimate/data/climate_layers/primary/TerraClim',
                    help='path to dataset')
parser.add_argument('-l', '--land', default='/gpfs/alpine/syb105/proj-shared/Personal/jmerlet/projects/climatypes/data/land_coords/paris.npy',
                    help='path to list of land coordinates')
parser.add_argument('-b', '--batch', type=int, default=1,
                    help='training batch size')
parser.add_argument('-w', '--window', type=int, default=3,
                    help='geographic window size')
parser.add_argument('-n', '--num-iterations', type=int, default=200,
                    help='number of batches to use for training')
parser.add_argument('--lr', type=float, default=0.0001,
                    help='(fixed) learning rate')
parser.add_argument('-v', '--verbose', action='store_true',
                    help='generate verbose output')
args = parser.parse_args()

# instantiate DataReader
reader = DataReader(verbose = args.verbose)
# create and/or load .npy xy-coordinate file
#data = reader.scan_input_data(data_root = args.data,
#                       land_xy_file = args.land,
#                       years_only = [1988, 1989],
#                       subregion=[[0, 4], [46, 50]])
reader.all_data(data_root=args.data, years=[1988])


