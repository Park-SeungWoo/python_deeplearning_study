from tensorflow import keras
from keras.layers import Dense, Flatten, Conv2D, AveragePooling2D, Dropout
from keras.models import Sequential
from keras.utils import np_utils
import matplotlib.pyplot as plt

# mnist 데이터를 불러오고 train과 test로 나눔
mnist_data = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist_data.load_data()

# output node의 수, batch size, epoch
class_num = 10
batch_size = 200
epoch = 10

# train, test image가 몇개씩 있는지 확인
print('train data amount : ' + str(len(train_images)))
print('test data amount : ' + str(len(test_images)))

# 데이터 전처리
# keras dataset의 mnist는 이미지의 픽셀 값이 0~255사이의 정수로 되어있어 0~1사이의 실수로 변환하며 28,28,1형태로 reshpe
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32') / 255
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1).astype('float32') / 255

#label을 one-hot벡터로 변환
train_labels = np_utils.to_categorical(train_labels, 10)
test_labels = np_utils.to_categorical(test_labels, 10)

# 모델 구성
model = Sequential()
# convnet에서 9*9 필터 20개와 input shape을 데이터 전처리에서 reshpe했던대로 28,28,1로 하고
# 활성화함수는 relu, padding은 same으로 해서 이미지의 크기가 작아지지 않게 함, stride는 1로 해서 한칸씩 움직이며 수행하도록 하고, 가중치는 he를 사용해 초기화 함(이하 동일)
model.add(Conv2D(20, kernel_size=(9, 9), input_shape=(28, 28, 1), activation='relu', padding='same', strides=1, kernel_initializer='he_normal'))
# 평균풀링 사용해서 풀링 사이즈는 2*2로 정함
model.add(AveragePooling2D(pool_size=(2, 2)))
# Drop out을 25% 적용
model.add(Dropout(0.25))
# 특징추출이 끝난 2차원형태의 배열을 1차원으로 변환
model.add(Flatten())
# hidden layer 노드 100개 활성화 함수 relu
model.add(Dense(100, activation='relu', kernel_initializer='he_normal'))
# Drop out 25%
model.add(Dropout(0.25))
# 출력층 노드의 수는 미리 정했던 10으로 지정(0~9), 활성화 함수는 softmax사용
model.add(Dense(class_num, activation='softmax', kernel_initializer='he_normal'))

# model을 compile하며 학습 과정 설정, loss함수는 cross entropy, 갱신 기법은 adam을 사용
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# hist에 학습을 시키면서 발생한 모든 정보를 dictionary로 담음, 여기엔 history 속성(학습 acc, 학습 loss, 테스트 acc, 테스트 loss)이 들어있음
hist = model.fit(train_images, train_labels, batch_size=batch_size, epochs=epoch, validation_data=(test_images, test_labels))

model.save('mnist_CNN_test.h5')

# loss, accuracy 시각화
fig, loss_ax = plt.subplots()
acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'], 'y', label='train loss')
loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')
loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
loss_ax.legend(loc='lower left')

acc_ax.plot(hist.history['accuracy'], 'b', label='train acc')
acc_ax.plot(hist.history['val_accuracy'], 'g', label='val acc')
acc_ax.set_ylabel('accuracy')
acc_ax.legend(loc='upper left')

plt.show() 
