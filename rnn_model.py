import tensorflow.compat.v1 as tf
import numpy as np

# data preprocessing
alpha_data = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

num_dic = {item: index for index, item in enumerate(alpha_data)}
dic_len = len(num_dic)

print(dic_len)

seq_data = ['word', 'park', 'lean', 'deep', 'dive', 'cool', 'cold', 'load', 'love', 'kiss', 'kind', 'wood']

def make_batch(seq_data):
    input_batch = []
    target_batch = []
    for seq in seq_data:
        input_d = [num_dic[n] for n in seq[: -1]]
        target_d = num_dic[seq[-1]]

        input_batch.append(np.eye(dic_len)[input_d])
        target_batch.append(target_d)
    return input_batch, target_batch


# set variables
learning_rate = 0.01
n_hidden = 124
epoch = 30

n_step = 3
n_input = n_class = dic_len

# make model
tf.compat.v1.disable_eager_execution()
X = tf.placeholder(tf.float32, [None, n_step, n_input])
Y = tf.placeholder(tf.int32, [None])

W = tf.Variable(tf.random_normal([n_hidden, n_class]))
B = tf.Variable(tf.random_normal([n_class]))

cell1 = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
cell1 = tf.nn.rnn_cell.DropoutWrapper(cell1, output_keep_prob=0.5)
cell2 = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)

multi_cell = tf.nn.rnn_cell.MultiRNNCell([cell1, cell2])

outputs, states = tf.nn.dynamic_rnn(multi_cell, X, dtype=tf.float32)

outputs = tf.transpose(outputs, [1, 0, 2])
outputs = outputs[-1]
model = tf.matmul(outputs, W) + B

cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model, labels = Y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# train model
sess = tf.Session()
sess.run(tf.global_variables_initializer())

input_batch, target_batch = make_batch(seq_data)

for e in range(epoch):
    _, loss = sess.run([optimizer, cost], feed_dict={X: input_batch, Y: target_batch})

    print(f'Epoch : {e+1:04d} Cost = {loss:.6f}')

print('optimization finished')

prediction = tf.cast(tf.argmax(model, 1), tf.int32)
prediction_check = tf.equal(prediction, Y)
accuracy = tf.reduce_mean(tf.cast(prediction_check, tf.float32))

# input_batch, target_batch = make_batch(seq_data)
predict, accuracy_val = sess.run([prediction, accuracy], feed_dict={X: input_batch, Y: target_batch})

# print predicted word
predict_words = []

for idx, val in enumerate(seq_data):
    last_char = alpha_data[predict[idx]]
    predict_words.append(val[:3] + last_char)

print('\n============result==============')
print('input : ', [w[:3] + ' ' for w in seq_data])
print('prediction : ', predict_words)
print('accuracy : ', accuracy_val)