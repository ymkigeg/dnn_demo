

import argparse
import os
import sys
import time

import tensorflow as tf
import keras
from keras import backend as K

import pandas as pd
import numpy as np
from six.moves import xrange

import keras_utils

FLAGS = None

def genrate_data():
  # train_data = pd.read_csv(FLAGS.data_file, header=None, chunksize=FLAGS.batch_size)
  while 1:
    train_data = pd.read_csv(FLAGS.data_file, header=None, chunksize=FLAGS.batch_size)
    for data in train_data:
      data = data.fillna(0)
      batch_X = data.iloc[:,1:].values
      batch_y = data.iloc[:,0].values.reshape(-1, 1)
      yield (batch_X, batch_y)


def run_training():
  # test on MNIST.
  df = pd.read_csv('norm_test_data.csv', header=None, nrows=10000)
  df = df.fillna(0)
  feature_length = df.shape[1]-1

  # Tell TensorFlow that the model will be built into the default Graph.
  with tf.Graph().as_default():
    config = {
      'lr': FLAGS.learning_rate,
      'batch_size': FLAGS.batch_size,
      'hidden1': FLAGS.hidden1,
      'hidden2': FLAGS.hidden2,
      'fake_data': FLAGS.fake_data,
      'num_class': 2,
      'feature_length': feature_length,
      'loss': FLAGS.loss_type
    }

    model = keras.models.Sequential()
    model.add(keras.layers.Dense(FLAGS.hidden1, activation='relu', input_dim=feature_length))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(FLAGS.hidden2, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    optimizer = keras.optimizers.Adagrad(lr=FLAGS.learning_rate, epsilon=1e-06)

    bestModelPath = 'keras/keras.model'

    cb = [
      # keras_utils.RocAucMetricCallback(FLAGS.batch_size), 
      keras.callbacks.EarlyStopping(monitor='val_auc', patience=5, verbose=1, mode='max'),
      keras.callbacks.ModelCheckpoint(bestModelPath, monitor='val_auc', verbose=1, save_best_only=True, save_weights_only=False, mode='max', period=1),
      keras.callbacks.TensorBoard(log_dir='./keras', histogram_freq=1),
    ]

    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=[keras_utils.auc])

    y_test = df.iloc[:,0].values.reshape(-1, 1)
    x_test = df.iloc[:,1:].values
    history = model.fit_generator(genrate_data(), steps_per_epoch=990000/FLAGS.batch_size, epochs=FLAGS.epoch, verbose=1, validation_data=(x_test, y_test), callbacks=cb)

    model.save('keras/model.h5')
    # model = keras.models.load_model('keras/model.h5')
    

def main(_):
  if tf.gfile.Exists(FLAGS.log_dir):
    tf.gfile.DeleteRecursively(FLAGS.log_dir)
  tf.gfile.MakeDirs(FLAGS.log_dir)
  run_training()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--learning_rate',
      type=float,
      default=0.0001,
      help='Initial learning rate.'
  )
  parser.add_argument(
      '--epoch',
      type=int,
      default=100,
      help='Number of epoch to run trainer.'
  )
  parser.add_argument(
      '--hidden1',
      type=int,
      default=512,
      help='Number of units in hidden layer 1.'
  )
  parser.add_argument(
      '--hidden2',
      type=int,
      default=64,
      help='Number of units in hidden layer 2.'
  )
  parser.add_argument(
      '--loss_type',
      type=str,
      default='logloss',
      help='loss type: logloss or mse'
  )
  parser.add_argument(
      '--batch_size',
      type=int,
      default=10000,
      help='Batch size.  Must divide evenly into the dataset sizes.'
  )
  parser.add_argument(
      '--log_dir',
      type=str,
      default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                           'tensorflow/dnn/logs/note_recommend'),
      help='Directory to put the log data.'
  )
  parser.add_argument(
      '--fake_data',
      default=False,
      help='If true, uses fake data for unit testing.',
      action='store_true'
  )
  parser.add_argument(
      '--data_file',
      default='xgb_feature_data.csv',
      help='data file for training',
  )

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
