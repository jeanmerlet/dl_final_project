#!/usr/bin/env python3

# before we do much of anything, disable writing bytecode, as that suffers
# from race conditions with distributed jobs on shared file systems
import sys
sys.dont_write_bytecode = True

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, layers, optimizers
from reader_fp import CNNDataReader
import argparse
import json
import time
import os

def n_years_to_one_year_cnn(num_input_layers, num_output_layers, window_diam, area_size, nyears):
    input_shape = (area_size + window_diam - 1, area_size + window_diam - 1, nyears * 12 * num_input_layers)
    model = models.Sequential()
    model.add(layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(area_size**2 * 12 * num_output_layers))
    model.add(layers.Reshape((area_size**2, 12, num_output_layers)))
    print(model.summary())
    return model

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data', default='/gpfs/alpine/syb105/proj-shared/Projects/NV_ORNL_XAIClimate/data/climate_layers/primary/TerraClim',
                    help='path to dataset')
parser.add_argument('-l', '--land', default='/gpfs/alpine/syb105/proj-shared/Personal/jmerlet/projects/climatypes/data/land_coords/paris.npy',
                    help='path to list of land coordinates')
parser.add_argument('-b', '--batch', type=int, default=1,
                    help='training batch size')
parser.add_argument('-w', '--window', type=int, default=10,
                    help='geographic window size')
parser.add_argument('-a', '--area', type=int, default=10,
                    help='geographic area size')
parser.add_argument('-y', '--years', type=int, default=10,
                    help='number of input years')
parser.add_argument('-n', '--num-iterations', type=int, default=100,
                    help='number of iterations to run')
parser.add_argument('--lr', type=float, default=0.01,
                    help='(fixed) learning rate')
parser.add_argument('-v', '--verbose', action='store_true',
                    help='generate verbose output')
parser.add_argument('-p', '--distributed', action='store_true',
                    help='enable distributed learning')
args = parser.parse_args()

# instantiate DataReader
reader = CNNDataReader(verbose = args.verbose)

# create and/or load .npy xy-coordinate file
reader.scan_input_data(data_root = args.data,
                       land_xy_file = args.land,
                       year_min=1958,
                       year_max=2006,
                       point = (48.86, 2.34))

# configure batches for the generator
reader.configure_batch(batch_size = args.batch,
                       window_size = args.window,
                       area_size = args.area,
                       num_years = args.years,
                       dtype = np.float32)

log_dir = '/gpfs/alpine/syb105/proj-shared/Personal/jmerlet/projects/climatypes/scripts/models/merlet_final/dl_final_project/logs'
loss_filename = f'cnn_lr{args.lr}_area{args.area}_window{args.window}_years{args.years}_layers{reader.num_input_layers()}_paris.npy'

# distribution for multiple GPUs on one node
mirrored_strategy = tf.distribute.MirroredStrategy()
num_gpus = mirrored_strategy.num_replicas_in_sync
lr = args.lr * num_gpus
print("# GPUs available: ", num_gpus)

# create model
with mirrored_strategy.scope():
    model = n_years_to_one_year_cnn(reader.num_input_layers(),
                                    reader.num_output_layers(),
                                    reader.window_diam,
                                    args.area,
                                    args.years)
    opt = keras.optimizers.SGD(lr=args.lr)
    loss = keras.losses.MeanSquaredError()

model.compile(loss=loss, optimizer=opt)

# train
start = time.time()
loss_history = model.fit(x=reader, epochs=args.num_iterations)
print(f'total train time: {round(time.time() - start, 2)}')
np.save(os.path.join(log_dir, loss_filename), np.array(loss_history.history['loss']))

# predict a random sample
np.set_printoptions(suppress=True)
batch_data, target_data, _ = reader.__getitem__()
target_data = np.squeeze(target_data)
predictions = np.round(np.squeeze(model.predict(batch_data)), 2)
print(predictions.shape, target_data.shape)
for i in range(args.area**2):
    for j, l in enumerate(reader.layers):
        print(f'predictions for point {i+1}, layer {l}:')
        print(predictions[i, :, j])
        print(f'ground truth for point {i+1}, layer {l}:')
        print(target_data[i, :, j])
        print('')

# validate predictions with unused training year ranges
# and save prediction and ground truth as np arrays
# in the shape (2, num_points, 12, num_layers)
year_min, year_max = 2007, 2017
val_filename = f'year_range_{year_min}-{year_max}_pred_and_gt.npy'
print('** using the model to predict a different year **')
reader.scan_input_data(data_root = args.data,
                       land_xy_file = args.land,
                       year_min=year_min,
                       year_max=year_max,
                       point = (48.86, 2.34))

log_data = []
batch_data, target_data, _ = reader.__getitem__()
target_data = np.squeeze(target_data)
predictions = np.round(np.squeeze(model.predict(batch_data)), 2)
log_data.append(predictions)
log_data.append(target_data)
print(predictions.shape, target_data.shape)
np.save(os.path.join(log_dir, val_filename), np.array(log_data))
for i in range(args.area**2):
    for j, l in enumerate(reader.layers):
        print(f'predictions for point {i+1}, layer {l}:')
        print(predictions[i, :, j])
        print(f'ground truth for point {i+1}, layer {l}:')
        print(target_data[i, :, j])
        print('')
