#!/usr/bin/env python3

# before we do much of anything, disable writing bytecode, as that suffers
#  from race conditions with distributed jobs on shared file systems
import sys
sys.dont_write_bytecode = True

import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers
from reader_fp_refactor import DataReader
import argparse

def basic_cnn(num_input_layers, num_output_layers, window_diam):
    model = models.Sequential()
    model.add(layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=(window_diam, window_diam, 12*num_input_layers)))
    model.add(layers.BatchNormalization())
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(12*num_output_layers))
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
parser.add_argument('-n', '--num-iterations', type=int, default=200,
                    help='number of batches to use for training')
parser.add_argument('--lr', type=float, default=0.01,
                    help='(fixed) learning rate')
parser.add_argument('-v', '--verbose', action='store_true',
                    help='generate verbose output')
args = parser.parse_args()

# instantiate DataReader
reader = DataReader(verbose = args.verbose)
# create and/or load .npy xy-coordinate file
reader.scan_input_data(data_root = args.data,
                       land_xy_file = args.land,
                       years_only = [1988, 1989],
                       subregion=[[0, 4], [46, 50]])
# configure batches
reader.configure_batch(batch_size = args.batch,
                       window_size = args.window,
                       dtype = np.float32)


#obtain variables from reader
num_input_layers = reader.num_input_layers()
num_output_layers = reader.num_output_layers()

# create model
window_diam = 2 * args.window + 1
model = basic_cnn(num_input_layers,
                  num_output_layers,
                  window_diam)
loss = tf.keras.losses.MeanSquaredError()
opt = tf.keras.optimizers.SGD(lr=args.lr)
model.compile(loss=loss, optimizer=opt)

# train
for n in range(args.num_iterations):
    batch_data, target_data = reader.next_batch()
    batch_data = batch_data.reshape(1, num_input_layers*window_diam, num_input_layers*window_diam, 12)
    with tf.GradientTape() as tape:
        logits = model(batch_data, training=True)
        loss_value = loss(target_data, logits)
    grads = tape.gradient(loss_value, model.trainable_weights)
    opt.apply_gradients(zip(grads, model.trainable_weights))
    print(f'iteration {n}/{args.num_iterations}, loss={loss_value}')

np.set_printoptions(suppress=True)
print(np.round(model.predict(batch_data), 2))
print(target_data)
