
import tensorflow as tf
import pandas as pd
import dnn_demo

data_file = 'norm_feature_data.csv'
df = pd.read_csv(data_file, header=None, nrows=10000)
df = df.fillna(0)
feature_length = df.shape[1]-1

config = {
      'lr': 0.01, 
      'batch_size': 1000, 
      'hidden1': 512,
      'hidden2': 64,
      'fake_data': False,
      'num_class': 2,
      'feature_length': feature_length,
      'loss': 'logloss'
}

model = dnn_demo.DNN_DEMO(config)
model.build_graph()

init = tf.global_variables_initializer()
saver = tf.train.Saver()

sess = tf.Session()
sess.run(init)
sess.run(tf.local_variables_initializer())

model_path = '/tmp/tensorflow/dnn/logs/note_recommend'
model_file=tf.train.latest_checkpoint(model_path)

saver.restore(sess, model_file)

label = df.iloc[:,0].values.reshape(-1, 1)
feed_dict = model.fill_feed_dict(df.iloc[:,1:].values, label, 1)
val_loss, _ = sess.run([model.loss, model.auc_op], feed_dict=feed_dict)
val_auc = sess.run(model.auc_value, feed_dict=feed_dict)
print('val_loss:%f, val_auc:%f' % (val_loss, val_auc))


sess.close()
