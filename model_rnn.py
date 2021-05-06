#!/usr/bin/env python3

# before we do much of anything, disable writing bytecode, as that suffers
#  from race conditions with distributed jobs on shared file systems
import sys
sys.dont_write_bytecode = True

import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers
from reader_rnn import RNNDataReader
import argparse


def basic_cnn(num_input_layers, num_output_layers, window_diam, nyears):
    model = models.Sequential()
    model.add(layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=(window_diam, window_diam, nyears*12*num_input_layers)))
    model.add(layers.BatchNormalization())
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(12*num_output_layers))
    model.add(layers.Reshape((num_input_layers, 12)))
    print(model.summary())
    return model

def basic_rnn(hidden_layer, steps, input_vals, lr, num_layers):
    model = models.Sequential()
    model.add(layers.SimpleRNN(units = hidden_layer, input_shape = (steps,input_vals), activation = 'relu'))
    model.add(layers.Dense(12))
    model.add(layers.Dense(num_layers))
    optimizer = optimizers.Adam(lr = lr)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    print(model.summary())
    return model

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data', default='/gpfs/alpine/syb105/proj-shared/Projects/NV_ORNL_XAIClimate/data/climate_layers/primary/TerraClim',
                    help='path to dataset')
parser.add_argument('-l', '--land', default='/gpfs/alpine/syb105/proj-shared/Personal/jmerlet/projects/climatypes/data/land_coords/paris.npy',
                    help='path to list of land coordinates')
parser.add_argument('-b', '--batch', type=int, default=1,
                    help='training batch size')
parser.add_argument('-w', '--window', type=int, default=5,
                    help='geographic window size')
parser.add_argument('-n', '--num-iterations', type=int, default=100,
                    help='number of batches to use for training')
parser.add_argument('--lr', type=float, default=0.01,
                    help='(fixed) learning rate')
parser.add_argument('-v', '--verbose', action='store_true',
                    help='generate verbose output')
parser.add_argument('-y', '--years', type=int, default=1, help = 'number of years for training data')
parser.add_argument('-s', '--step', type = int, default = 2, help = 'step size for RNN')
args = parser.parse_args()

# instantiate RNNDataReader
reader = RNNDataReader(verbose = args.verbose)
# create and/or load .npy xy-coordinate file
reader.scan_input_data(data_root = args.data,
                       land_xy_file = args.land,
                       year_min = 1958, year_max = 2006,
                       #point = (48.86, 2.34)) #Paris
                       #point = (24.77, 46.74)) #Riyadh
                       #point = (35.96, -83.92)) #Knoxville
                       point = (-33.87, 151.21)) #Sydney
                       #point = (39.54, 116.21)) #Beijing
                       #point = (-1.29, 36.82)) #Nairobi

# configure batches
reader.configure_batch(batch_size = args.batch,
                       window_size = args.window,
                       dtype = np.float32)

np.set_printoptions(suppress=True)

# find data
def scheduler(epoch, lr):
    if epoch % 20 == 0: return lr * 0.5
    return lr

callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

batch_data, target_data = reader.rnn_data(args.step)
print(np.array(batch_data).shape)
print(np.array(target_data).shape)
input_vals = batch_data.shape[2]
model = basic_rnn(300, args.step, input_vals, args.lr, reader.num_input_layers())
history = model.fit(batch_data,target_data, epochs=100, batch_size=16, verbose=1, callbacks = [callback])
loss = np.array(history.history['loss'])
print(loss)


validation = RNNDataReader(verbose = args.verbose)

# create and/or load .npy xy-coordinate file
validation.scan_input_data(data_root = args.data,
                       land_xy_file = args.land,
                       year_min = 2007, year_max = 2017,
                       #point = (48.86, 2.34)) #Paris
                       #point = (24.77, 46.74)) #Riyadh
                       #point = (35.96, -83.92)) #Knoxville
                       point = (-33.87, 151.21)) #Sydney
                       #point = (39.54, 116.21)) #Beijing
                       #point = (-1.29, 36.82)) #Nairobi


validation.configure_batch(batch_size = args.batch,
                       window_size = args.window,
                       dtype = np.float32)
print('predicting')
test_dat, truth = validation.rnn_data(args.step)
outs = model.predict(test_dat)
print(f'output: {outs}')
print(f'ground truth: {truth}')
print(outs.shape)
np.save('outputs/Sydneytruth.npy', truth)
np.save('outputs/Sydneyloss.npy', loss)
np.save('outputs/Sydneypreds.npy', outs)
