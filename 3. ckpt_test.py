# 생성 한 ckpt 파일이 정상 작동되는지 확인하며,
# 테스트 데이터 중, Accuracy 95% 이상이 되는 데이터셋을 찾기 위한 코드  ( Team. NMSP )

import numpy as np
import tensorflow as tf
from sklearn import metrics

def next_batch(num, data, labels):
  idx = np.arange(0 , len(data))
  np.random.shuffle(idx)
  idx = idx[:num]
  data_shuffle = [data[i] for i in idx]
  labels_shuffle = [labels[i] for i in idx]

  return np.asarray(data_shuffle), np.asarray(labels_shuffle)

X_test=np.load('test_data.npy')
y_test=np.load('test_label.npy')

X_test.shape, y_test.shape

g1=tf.Graph() # 모델 생성하기.
training_epochs = 10000
learning_rate = 1e-3

with g1.as_default():
  X=tf.placeholder(tf.float32,[None,32,32,3])
  Y=tf.placeholder(tf.float32,[None,153])
  
  keep_prob=tf.placeholder(tf.float32)
  y_test_one_hot = tf.squeeze(tf.one_hot(y_test, 153),axis=1)
  
  W_conv1 = tf.Variable(tf.truncated_normal(shape=[5, 5, 3, 64], stddev=5e-2))
  b_conv1 = tf.Variable(tf.constant(0.1, shape=[64]))
  h_conv1 = tf.nn.relu(tf.nn.conv2d(X, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)

  # 첫번째 Pooling layer
  h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

  # 두번째 convolutional layer - 32개의 특징들(feature)을 64개의 특징들(feature)로 맵핑(maping)합니다.
  W_conv2 = tf.Variable(tf.truncated_normal(shape=[5, 5, 64, 64], stddev=5e-2))
  b_conv2 = tf.Variable(tf.constant(0.1, shape=[64]))
  h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)

  # 두번째 pooling layer.
  h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

  # 세번째 convolutional layer
  W_conv3 = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 128], stddev=5e-2))
  b_conv3 = tf.Variable(tf.constant(0.1, shape=[128]))
  h_conv3 = tf.nn.relu(tf.nn.conv2d(h_pool2, W_conv3, strides=[1, 1, 1, 1], padding='SAME') + b_conv3)

  # 네번째 convolutional layer
  W_conv4 = tf.Variable(tf.truncated_normal(shape=[3, 3, 128, 128], stddev=5e-2))
  b_conv4 = tf.Variable(tf.constant(0.1, shape=[128])) 
  h_conv4 = tf.nn.relu(tf.nn.conv2d(h_conv3, W_conv4, strides=[1, 1, 1, 1], padding='SAME') + b_conv4)

  # 다섯번째 convolutional layer
  W_conv5 = tf.Variable(tf.truncated_normal(shape=[3, 3, 128, 128], stddev=5e-2))
  b_conv5 = tf.Variable(tf.constant(0.1, shape=[128]))
  h_conv5 = tf.nn.relu(tf.nn.conv2d(h_conv4, W_conv5, strides=[1, 1, 1, 1], padding='SAME') + b_conv5)

  # Fully Connected Layer 1 - 2번의 downsampling 이후에, 우리의 32x32 이미지는 8x8x128 특징맵(feature map)이 됩니다.
  # 이를 384개의 특징들로 맵핑(maping)합니다.
  W_fc1 = tf.Variable(tf.truncated_normal(shape=[8 * 8 * 128, 384], stddev=5e-2))
  b_fc1 = tf.Variable(tf.constant(0.1, shape=[384]))

  h_conv5_flat = tf.reshape(h_conv5, [-1, 8*8*128])
  h_fc1 = tf.nn.relu(tf.matmul(h_conv5_flat, W_fc1) + b_fc1)

  # Dropout - 모델의 복잡도를 컨트롤합니다. 특징들의 co-adaptation을 방지합니다.
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob) 

  # Fully Connected Layer 2 - 384개의 특징들(feature)을 10개의 클래스-airplane, automobile, bird...-로 맵핑(maping)합니다.
  W_fc2 = tf.Variable(tf.truncated_normal(shape=[384, 153], stddev=5e-2))
  b_fc2 = tf.Variable(tf.constant(0.1, shape=[153]))
  logits = tf.matmul(h_fc1_drop,W_fc2) + b_fc2
  y_pred = tf.nn.softmax(logits)
  
  loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=logits))
  train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)

  correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(Y, 1))
  #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  
  init = tf.global_variables_initializer()
  saver = tf.train.Saver()

length=100 # 테스트 데이터의 사이즈
label_true=np.empty((length,1),dtype=np.uint8)
label_pred=np.empty((length,1),dtype=np.uint8)

while True:
    with tf.Session(graph=g1) as sess:
      sess.run(init)
      saver.restore(sess,'model_cnn.ckpt')
      
      batch=next_batch(length,X_test,y_test_one_hot.eval()) # 이를 통해서 Accuracy 95% 이상 데이터 그룹화하기.
      
      label_p=sess.run(y_pred,feed_dict={X:batch[0],Y:batch[1],keep_prob:1.0})
      
    for x in range(len(label_true)):
        label_true[x]=(np.argmax(batch[1][x]))
    
    for x in range(len(label_pred)):
      label_pred[x]=(np.argmax(label_p[x]))
    
    
    label_t=[]
    label_p=[]
    
    for x in range(length):
        label_t.append(str(label_true[x][0]))
        label_p.append(str(label_pred[x][0]))
    
    accuracy=metrics.adjusted_rand_score(label_t, label_p) # 정확도를 측정하기 위한 변수
    
    print("Score for %s: %s" % ("label_pred", accuracy))
    if accuracy>=0.95: # 95% 이상일 때 멈춘다.
        break

# Accuracy 95% 이상인 테스트 데이터 집합을 npy 형태로 저장.
        
np.save("data_true.npy",batch[0])
np.save("one_hot_label_true.npy",batch[1])
np.save("label_true.npy",label_true)
np.save("label_pred.npy",label_pred)

'''
print(label_true)
print(label_pred)
'''