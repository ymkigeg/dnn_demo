

import tensorflow as tf
import keras
from keras import backend as K

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

class RocAucMetricCallback(keras.callbacks.Callback):
  def __init__(self, predict_batch_size=1024, include_on_batch=False):
    super(RocAucMetricCallback, self).__init__()
    self.predict_batch_size=predict_batch_size
    self.include_on_batch=include_on_batch
 
  def on_batch_begin(self, batch, logs={}):
    pass
 
  def on_batch_end(self, batch, logs={}):
    if(self.include_on_batch):
      logs['roc_auc_val']=float('-inf')
      if(self.validation_data):
        logs['roc_auc_val']=roc_auc_score(self.validation_data[1], 
                                          self.model.predict(self.validation_data[0],
                                          batch_size=self.predict_batch_size))
 
  def on_train_begin(self, logs={}):
    if not ('roc_auc_val' in self.params['metrics']):
      self.params['metrics'].append('roc_auc_val')
 
  def on_train_end(self, logs={}):
    pass
 
  def on_epoch_begin(self, epoch, logs={}):
    pass

  def on_epoch_end(self, epoch, logs={}):
    logs['roc_auc_val']=float('-inf')
    if(self.validation_data):
      logs['roc_auc_val']=roc_auc_score(self.validation_data[1], 
                                        self.model.predict(self.validation_data[0],
                                        batch_size=self.predict_batch_size))

def auc(y_true, y_pred):
    ptas = tf.stack([binary_PTA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
    pfas = tf.stack([binary_PFA(y_true,y_pred,k) for k in np.linspace(0, 1, 1000)],axis=0)
    pfas = tf.concat([tf.ones((1,)) ,pfas],axis=0)
    binSizes = -(pfas[1:]-pfas[:-1])
    s = ptas*binSizes
    return K.sum(s, axis=0)

# PFA, prob false alert for binary classifier
def binary_PFA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    N = K.sum(1 - y_true)
    FP = K.sum(y_pred - y_pred * y_true)
    return FP/N

# P_TA prob true alerts for binary classifier
def binary_PTA(y_true, y_pred, threshold=K.variable(value=0.5)):
    y_pred = K.cast(y_pred >= threshold, 'float32')
    P = K.sum(y_true)
    TP = K.sum(y_pred * y_true)
    return TP/P


