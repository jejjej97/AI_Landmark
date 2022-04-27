from PIL import Image
import os, glob
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import MaxPooling2D
from keras.layers import Conv2D
from keras.layers import Activation, Dropout, Flatten, Dense
import os

# 분류 대상 카테고리 선택하기 
accident_dir = "./landmarkimg"
categories = ['경복궁','광화문','남산타워','청계천','코엑스']
nb_classes = len(categories)
# 이미지 크기 지정 
image_w = 64 
image_h = 64
pixels = image_w * image_h * 3
# 이미지 데이터 읽어 들이기 
X = []
Y = []
for idx, cate in enumerate(categories):
    # 레이블 지정 
    label = [0 for i in range(nb_classes)]
    label[idx] = 1
    # 이미지 
    image_dir = accident_dir + "/" + cate
    print(image_dir)
    files = glob.glob(image_dir+"/*.jpg")
    # print(files)
    for i, f in enumerate(files):
        img = Image.open(f) 
        img = img.convert("RGB")
        img = img.resize((image_w, image_h))
        data = np.asarray(img)      # numpy 배열로 변환
        X.append(data)
        Y.append(label)
        if i % 10 == 0:
            print(i, "\n", data)

X = np.array(X)
Y = np.array(Y)
# 학습 전용 데이터와 테스트 전용 데이터 구분 
X_train, X_test, y_train, y_test = train_test_split(X, Y)
xy = (X_train, X_test, y_train, y_test)

# 데이터 정규화하기(0~1사이로)
X_train = X_train.astype("float") / 256
X_test  = X_test.astype("float")  / 256
print('X_train shape:', X_train.shape)

# 모델 구조 정의
model = Sequential()
model.add(Conv2D(9, (3, 3), input_shape=X_train.shape[1:], padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(9, (3, 3), padding='same'))
model.add(Activation('relu'))

model.add(Conv2D(9, (3, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# 전결합층
model.add(Flatten())    # 벡터형태로 reshape
model.add(Dense(512))   # 출력
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(nb_classes))
model.add(Activation('softmax'))
# 모델 구축하기
model.compile(loss='categorical_crossentropy',   # 최적화 함수 지정
    optimizer='rmsprop',
    metrics=['accuracy'])
# 모델 확인
#print(model.summary())

# 학습 완료된 모델 저장

model.fit(X_train, y_train, batch_size=9, epochs = 100)
model.save("landmark_model.h5")

# # 적용해볼 이미지
# test_image = 'C:/AI_CatProject-master/img/test6.jpg'
# # 이미지 resize
# img = Image.open(test_image)
# img = img.convert("RGB")
# img = img.resize((64,64))
# data = np.asarray(img)
# X = np.array(data)
# X = X.astype("float") / 256
# X = X.reshape(-1, 64, 64,3)
# # 예측
# pred = model.predict(X)
# result = [np.argmax(value) for value in pred]   # 예측 값중 가장 높은 클래스 반환
# print('New data category : ',categories[result[0]])