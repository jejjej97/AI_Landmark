from PIL import Image
import numpy as np
import tensorflow as tf

categories = ['경복궁','광화문','남산타워','청계천','코엑스']

model = tf.keras.models.load_model('C:/AI_landmark/landmark_model.h5')
# 모델 구조를 출력합니다
model.summary()

# 적용해볼 이미지
test_image = 'C:/AI_landmark/img/test1.jpg'
# 이미지 resize
img = Image.open(test_image)
img = img.convert("RGB")
img = img.resize((64,64))
data = np.asarray(img)
X = np.array(data)
X = X.astype("float") / 256
X = X.reshape(-1, 64, 64,3)
# 예측
pred = model.predict(X)
result = [np.argmax(value) for value in pred]   # 예측 값중 가장 높은 클래스 반환
print('New data category : ',categories[result[0]])