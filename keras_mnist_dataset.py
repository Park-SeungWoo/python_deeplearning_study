from tensorflow import keras
import random
import matplotlib.pyplot as plt

# mnist 데이터를 불러오고 train과 test로 나눔
mnist_data = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist_data.load_data()

# 존재하는 label들을 리스트에 저장해둠
class_name = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]

# train, test image가 몇개씩 있는지 확인
print('train data amount : ' + str(len(train_images)))
print('test data amount : ' + str(len(test_images)))

# label structure checking (index in range(0, 100))
# for i in range(0, 100):
#     print(train_labels[i], end=' ')
# print()

# 원하는 label중 하나를 추출하기 위해 변수를 선언해 둠
want_label = 7  # 이부분에 원하는 label값 넣기
image_index = 0
label_index_dict = dict()

# 각각의 label이 있는 index의 모음을 list에 저장하여 dictionary에 label과 함께 저장
for i in range(10):
    index_list = list()
    for index, item in enumerate(train_labels):
        if item == i:
            index_list.append(index)
    label_index_dict[i] = index_list

# 원하는 label의 image의 index를 랜덤으로 뽑아옴
want_index_list = label_index_dict[want_label]
idx = random.randrange(len(label_index_dict[want_label]) - 1)
image_index = want_index_list[idx]

print(f"train_images[{image_index}]")

# print pixel values in train_images[image_index]
for i, item in enumerate(train_images[image_index]):
    for p in train_images[image_index][i]:
        print(f"{p:^3}", end="")
    print()

print(f"train_images[{image_index}]'s label : " + str(train_labels[image_index]))

# print image as a real image
plt.title(train_labels[image_index])
plt.imshow(train_images[image_index], cmap='Greys')
plt.show()
