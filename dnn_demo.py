
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf
from six.moves import xrange


class DNN_DEMO(object):
  def __init__(self, config):
    self.num_class = config['num_class']
    self.feature_length = config['feature_length']
    self.lr = config['lr']
    self.batch_size = config['batch_size']
    self.hidden1_units = config['hidden1']
    self.hidden2_units = config['hidden2']
    self.fake_data = config['fake_data']
    self.loss_type = config['loss']

  def add_placeholders(self):
    self.X = tf.placeholder(tf.float32, shape=[None, self.feature_length])
    self.y = tf.placeholder(tf.float32, shape=[None, 1])
    self.keep_prob = tf.placeholder('float32')

  def fill_feed_dict(self, X, y, keep_prob):
    feed_dict = {
        self.X: X,
        self.y: y,
        self.keep_prob: keep_prob
    }
    return feed_dict

  def inference(self):
    # Hidden 1
    with tf.name_scope('hidden1'):
      weights1 = tf.Variable(
          tf.truncated_normal([self.feature_length, self.hidden1_units],
                            stddev=1.0 / math.sqrt(float(self.feature_length))),
          name='weights1')
      biases1 = tf.Variable(tf.zeros([self.hidden1_units]),
                           name='biases1')
      tf.summary.histogram('weights1', weights1)
      tf.summary.histogram('bias1', biases1)
      hidden1 = tf.nn.relu(tf.add(tf.matmul(self.X, weights1), biases1))
      hidden1 = tf.nn.dropout(hidden1, self.keep_prob)
    # Hidden 2
    with tf.name_scope('hidden2'):
      weights2 = tf.Variable(
          tf.truncated_normal([self.hidden1_units, self.hidden2_units],
                            stddev=1.0 / math.sqrt(float(self.hidden1_units))),
          name='weights2')
      biases2 = tf.Variable(tf.zeros([self.hidden2_units]),
                           name='biases2')
      hidden2 = tf.nn.relu(tf.add(tf.matmul(hidden1, weights2), biases2))
      hidden2 = tf.nn.dropout(hidden2, self.keep_prob)
    # Linear
    with tf.name_scope('softmax_linear'):
      weights_out = tf.Variable(
          tf.truncated_normal([self.hidden2_units, 1],
                              stddev=1.0 / math.sqrt(float(self.hidden2_units))),
          name='weights_out')
      biases_out = tf.Variable(tf.zeros([1]),
                         name='biases_out')
      self.logits = tf.add(tf.matmul(hidden2, weights_out), biases_out)
      self.out = tf.nn.sigmoid(self.logits)


  def add_loss(self):
    if self.loss_type == "logloss":
      self.loss = tf.losses.log_loss(self.y, self.out)
    elif self.loss_type == "mse":
      self.loss = tf.nn.l2_loss(tf.subtract(self.y, self.logits))

    self.auc_value, self.auc_op = tf.metrics.auc(self.y, self.out, num_thresholds=2000)
    # self.loss = tf.losses.sparse_softmax_cross_entropy(labels=self.y, logits=self.logits)


  def training(self):
    tf.summary.scalar('loss', self.loss)
    tf.summary.scalar('auc', self.auc_value)
    # optimizer = tf.train.GradientDescentOptimizer(self.lr)

    optimizer = tf.train.AdagradOptimizer(learning_rate=self.lr,
                                               initial_accumulator_value=1e-8)    

    self.global_step = tf.Variable(0, name='global_step', trainable=False)
    self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)


  def build_graph(self):
    """build graph for model"""
    self.add_placeholders()
    self.inference()
    self.add_loss()
    # self.add_accuracy()
    self.training()
    # self.evaluation()
