
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=missing-docstring
import argparse
import os
import sys
import time

import tensorflow as tf

import dnn_demo
import pandas as pd
import numpy as np
from six.moves import xrange

FLAGS = None

def run_training():
  # test on MNIST.
  df = pd.read_csv(FLAGS.data_file, header=None, nrows=10000)
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

    model = dnn_demo.DNN_DEMO(config)
    model.build_graph()

    summary = tf.summary.merge_all()
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    sess = tf.Session()
    summary_writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

    sess.run(init)
    sess.run(tf.local_variables_initializer())

    # Start the training loop.
    for epoch in xrange(FLAGS.epoch):
      losses = []
      aucs = []
      num_samples = 0

      start_time = time.time()
      
      train_data = pd.read_csv(FLAGS.data_file, header=None, chunksize=FLAGS.batch_size)
      for data in train_data:
        actual_batch_size = len(data)
        #if actual_batch_size < FLAGS.batch_size:
        #  break
        data = data.fillna(0)
        batch_X = data.iloc[:,1:].values
        batch_y = data.iloc[:,0].values.reshape(-1, 1)
        # print("batch_X: ", batch_X.shape, "batch_y:", batch_y.shape)
        
        feed_dict = model.fill_feed_dict(batch_X, batch_y, 0.5)

        _, loss_value, _, model_auc = sess.run([model.train_op, model.loss, model.auc_op, model.auc_value],
                                 feed_dict=feed_dict)

        print("batch loss=%.6f, auc=%s" % (loss_value, model_auc))
        num_samples += actual_batch_size
        losses.append(loss_value*actual_batch_size)
        aucs.append(model_auc*actual_batch_size)

      duration = time.time() - start_time

      mean_loss = np.sum(losses)/num_samples
      mean_auc = np.sum(aucs)/num_samples
      
      label = df.iloc[:,0].values.reshape(-1, 1)
      feed_dict = model.fill_feed_dict(df.iloc[:,1:].values, label, 1)
      eval_loss, _, eval_auc = sess.run([model.loss, model.auc_op, model.auc_value], feed_dict=feed_dict)
      print('epoch %d: train-loss = %.6f, train-auc = %.6f, test-loss = %.6f, test-auc = %.6f  (%.3f sec)' 
            % (epoch, mean_loss, mean_auc, eval_loss, eval_auc, duration))
      # Write the summaries and print an overview fairly often.
      if epoch % 1 == 0:
        # Print status to stdout.
        # print('epoch %d: loss = %.6f (%.3f sec)' % (epoch, total_loss, duration))
        # Update the events file.
        summary_str = sess.run(summary, feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, epoch)
        summary_writer.flush()

      checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt')
      saver.save(sess, checkpoint_file, global_step=epoch)
      
      # Save a checkpoint and evaluate the model periodically.
      #if (epoch + 1) % 100 == 0 or (epoch + 1) == FLAGS.epoch:
      #  checkpoint_file = os.path.join(FLAGS.log_dir, 'model.ckpt')
      #  saver.save(sess, checkpoint_file, global_step=epoch)
      #  # Evaluate against the training set.
      #  print('Training Data Eval:')
      #  model.do_eval(sess, df.iloc[:,1:].values, df.iloc[:,0].values)


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
      default=10,
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
