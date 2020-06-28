import cv2
import matplotlib.pyplot as plt

from keras.models import load_model

model = load_model('mnist_CNN_test.h5')

img = cv2.imread('numimage.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 모델이 흑백 이미지만을 predict할 수 있기때문에 변환
img_blur = cv2.GaussianBlur(img_gray, (15, 15), 0)  # 사진을 찍으면서 발생한 숫자 라인의 끊김등을 블러로 처리

# object들 detect하기
ret, img_th = cv2.threshold(img_blur, 100, 280, cv2.THRESH_BINARY_INV)
contours, hierachy= cv2.findContours(img_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
rects = [cv2.boundingRect(each) for each in contours]
print(rects)  # 각 숫자의 좌표

img_result = []
img_for_class = img_blur.copy()

margin_pixel = 50

for rect in rects:
    # 이미지에 detect된 숫자중 한개씩 꺼내와 model로 predict하고 사진에 predicted label 붙이기
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
