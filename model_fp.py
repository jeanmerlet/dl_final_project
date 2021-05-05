#!/usr/bin/env python3

# before we do much of anything, disable writing bytecode, as that suffers
# from race conditions with distributed jobs on shared file systems
import sys
sys.dont_write_bytecode = True

import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers
from reader_fp import DataReader
import argparse
import time


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

def basic_cnn(num_input_layers, num_output_layers, window_diam):
    model = models.Sequential()
    model.add(layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=(window_diam, window_diam, 12*num_input_layers)))
    model.add(layers.BatchNormalization())
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(12*num_output_layers))
    model.add(layers.Reshape((num_input_layers, 12)))
    print(model.summary())
    return model

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data', default='/gpfs/alpine/syb105/proj-shared/Projects/NV_ORNL_XAIClimate/data/climate_layers/primary/TerraClim',
                    help='path to dataset')
parser.add_argument('-l', '--land', default='/gpfs/alpine/syb105/proj-shared/Personal/jmerlet/projects/climatypes/data/land_coords/paris.npy',
                    help='path to list of land coordinates')
parser.add_argument('-b', '--batch', type=int, default=1,
                    help='training batch size')
parser.add_argument('-w', '--window', type=int, default=3,
                    help='geographic window size')
parser.add_argument('-a', '--area', type=int, default=3,
                    help='area size')
parser.add_argument('-y', '--years', type=int, default=5,
                    help='number of inputs years')
<<<<<<< HEAD
parser.add_argument('-n', '--num-iterations', type=int, default=10,
                    help='number of iterations to run')
=======
parser.add_argument('-n', '--num-iterations', type=int, default=100,
                    help='number of batches to use for training')
>>>>>>> bd528ef034e2492754c12b1a9b7da49d43987809
parser.add_argument('--lr', type=float, default=0.01,
                    help='(fixed) learning rate')
parser.add_argument('-v', '--verbose', action='store_true',
                    help='generate verbose output')
<<<<<<< HEAD
parser.add_argument('-p', '--distributed', action='store_true',
                    help='enable distributed learning')
=======
#parser.add_argument('-y', '--years', type=int, default=1, help = 'number of years for training data')

>>>>>>> bd528ef034e2492754c12b1a9b7da49d43987809
args = parser.parse_args()

# instantiate DataReader
reader = DataReader(verbose = args.verbose)

# create and/or load .npy xy-coordinate file
reader.scan_input_data(data_root = args.data,
                       land_xy_file = args.land,
                       year_min=1960,
                       year_max=1980,
                       point = (48.86, 2.34))

# configure batches
reader.configure_batch(batch_size = args.batch,
                       window_size = args.window,
                       area_size = args.area,
                       num_years = args.years,
                       dtype = np.float32)

# distribution for multiple GPUs on one node
mirrored_strategy = tf.distribute.MirroredStrategy()
num_gpus = mirrored_strategy.num_replicas_in_sync
lr = args.lr * num_gpus
print("Num GPUs Available: ", num_gpus)

# create model
<<<<<<< HEAD
with mirrored_strategy.scope():
    model = n_years_to_one_year_cnn(reader.num_input_layers(),
                                    reader.num_output_layers(),
                                    reader.window_diam,
                                    args.area,
                                    args.years)
    opt = keras.optimizers.SGD(lr=args.lr)
    loss = keras.losses.MeanSquaredError()

=======
window_diam = 2 * args.window + 1
# model = basic_cnn(reader.num_input_layers(),
#                   reader.num_output_layers(),
#                   window_diam, args.years)
model = n_years_to_one_year_cnn(reader.num_input_layers(),
                                reader.num_output_layers(),
                                window_diam,
                                args.area,
                                args.years)
loss = tf.keras.losses.MeanSquaredError()
opt = tf.keras.optimizers.SGD(lr=args.lr)
>>>>>>> bd528ef034e2492754c12b1a9b7da49d43987809
model.compile(loss=loss, optimizer=opt)

# train
start = time.time()
<<<<<<< HEAD
model.fit(x=reader, epochs=args.num_iterations)
=======
for n in range(args.num_iterations):
    batch_data, target_data = reader.next_batch(args.years)
    batch_data = batch_data.reshape(1, window_diam, window_diam, args.years*12*reader.num_input_layers())
    with tf.GradientTape() as tape:
        logits = model(batch_data, training=True)
        loss_value = loss(target_data, logits)
    grads = tape.gradient(loss_value, model.trainable_weights)
    opt.apply_gradients(zip(grads, model.trainable_weights))
    print(f'iteration {n}/{args.num_iterations}, loss={loss_value}')

>>>>>>> bd528ef034e2492754c12b1a9b7da49d43987809
print(f'total train time: {round(time.time() - start, 2)}')

np.set_printoptions(suppress=True)
batch_data, target_data, _ = reader.__getitem__()
predictions = np.round(model.predict(batch_data), 2)
for i in range(args.area**2):
    for j, l in enumerate(reader.layers):
        print(f'predictions for point {i+1}, layer {l}:')
        print(predictions[:, i, j, :])
        print(f'ground truth for point {i+1}, layer {l}:')
        print(target_data[:, i, j, :])
        print('')

# try to predict the following year
print('** using the model to predict a different year **')
reader.scan_input_data(data_root = args.data,
                       land_xy_file = args.land,
                       year_min=1990,
                       year_max=2000,
                       point = (48.86, 2.34))

batch_data, target_data, _ = reader.__getitem__()
predictions = np.round(model.predict(batch_data), 2)
for i in range(args.area**2):
    for j, l in enumerate(reader.layers):
        print(f'predictions for point {i+1}, layer {l}:')
        print(predictions[:, i, j, :])
        print(f'ground truth for point {i+1}, layer {l}:')
        print(target_data[:, i, j, :])
        print('')
