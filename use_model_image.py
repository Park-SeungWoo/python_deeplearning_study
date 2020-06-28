import cv2
import numpy as np
import matplotlib.pyplot as plt

from keras.models import load_model

model = load_model('mnist_CNN_test.h5')

img = cv2.imread('handwriting_num.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)

# object들 detect하기
ret, img_th = cv2.threshold(img_blur, 100, 230, cv2.THRESH_BINARY_INV)
contours, hierachy= cv2.findContours(img_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
rects = [cv2.boundingRect(each) for each in contours]
print(rects)

img_result = []
img_for_class = img_blur.copy()

margin_pixel = 50

for rect in rects:
    # [y:y+h, x:x+w]
    target_num = img_for_class[rect[1] - margin_pixel: rect[1] + rect[3] + margin_pixel, rect[0] - margin_pixel: rect[0] + rect[2] + margin_pixel]
    test_num = cv2.resize(target_num, (28, 28))
    test_num = (test_num < 70) * test_num
    test_num = test_num.astype('float32') / 255
    test_num = test_num.reshape((1, 28, 28, 1))
    predicted_num = model.predict_classes(test_num)

    # Draw the rectangles
    cv2.rectangle(img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 5)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, str(predicted_num[0]), (rect[0], rect[1]), font, 4, (0, 0, 255), 10)

plt.imshow(img)
plt.show()







# # for rect in rects:
# #     cv2.rectangle(img, (rect[0], rect[1]),
# #                   (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 5)
# #
# # plt.imshow(img)
# # plt.show()
#
# img_result = []
# img_for_class = img.copy()
#
# margin_pixel = 60
#
# for rect in rects:
#     # [y:y+h, x:x+w]
#     img_result.append(
#         img_for_class[rect[1] - margin_pixel: rect[1] + rect[3] + margin_pixel,
#         rect[0] - margin_pixel: rect[0] + rect[2] + margin_pixel])
#
#     # Draw the rectangles
#     cv2.rectangle(img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 5)
#
# # plt.imshow(img)
# # plt.show()
#
# count = 0
# nrows = 3
# ncols = 4
#
# for n in img_result:
#     count += 1
#     plt.subplot(nrows, ncols, count)
#     plt.imshow(cv2.resize(n, (28, 28)), cmap='Greys', interpolation='nearest')
#
# # plt.tight_layout()
# # plt.show()
#
# model.summary()
#
#
# # test_num = cv2.resize(img_result[2], (28,28))[:,:,1]
# # test_num = (test_num < 70) * test_num
# # test_num = test_num.astype('float32') / 255.
# #
# # plt.imshow(test_num, cmap='Greys', interpolation='nearest')
# #
# # test_num = test_num.reshape((1, 28, 28, 1))
# #
# # print('The Answer is ', model.predict_classes(test_num))
#
# # count = 0
# # nrows = 3
# # ncols = 4
# #
# # for n in img_result:
# #     count += 1
# #     plt.subplot(nrows, ncols, count)
# #
# #     test_num = cv2.resize(n, (28, 28))[:, :, 1]
# #     test_num = (test_num < 70) * test_num
# #     test_num = test_num.astype('float32') / 255
# #
# #     plt.imshow(test_num, cmap='Greys', interpolation='nearest')
# #
# #     test_num = test_num.reshape((1, 28, 28, 1))
# #     plt.title(model.predict_classes(test_num))
# #
# # plt.tight_layout()
# # plt.show()