import tensorflow as tf
import pandas as pd
import numpy as np

data = pd.read_csv('gpascore.csv')

# data.isnull().sum() #빈칸있는 칸을 찾아줌
# data.fillna(100) #빈칸을 100으로 채워줌
data = data.dropna()#빈칸을 삭제 해줌

ydata = data['admit'].values #ydata에 리스트형식으로 값이 담김
xdata = []

for i,rows in data.iterrows():
  xdata.append([rows['gre'],rows['gpa'],rows['rank']])

#텐서플로우의 keras를 쓰면 딥러닝이 훨씬 쉬워짐
model = tf.keras.models.Sequential([  # <- 저 문법이 딥러닝 모델 만드는 문법이다. , Sequential쓰면 신경망 레이어들 쉽게 만들어준다.
    tf.keras.layers.Dense(64,activation='tanh'), #<- 신경망 레이어들의 노드(히든레이어)들 만들어줌.
    tf.keras.layers.Dense(128,activation='tanh'),#<- Dense()안의 숫자는 노드의 숫자이고 활성함수(activationFun)가 들어가야한다.
    tf.keras.layers.Dense(1,activation='sigmoid'), #<- 마지막 레이어는 1로 지정해야하고, 예측결과를 뱉어야해서 0~1 사이의 확률을 내뱉고 싶으면 sigmoid를 쓴다.
]) 

#모델을 컴파일을 해야하는데 optimizer,loss,mertircs가 인자로 들어가야한다
#optmizer는 weight 값을 알맞게 조정해준다.(adam을 대부분 쓴다.)
#loss는 확률 예측을 하고 싶다면 binary_crossentropy을 쓴다.
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#모델을 학습시켜준다. x데이터(트레이닝데이터), y데이터(답)를 넣어야한다.
model.fit(np.array(xdata),np.array(ydata),epochs=1000) #epochs는 몇번 학습시킬지 정해준다., x,y데이터를 넣을때는 np배열로 넣어야한다.

#모델 예측하기 (테스트해 볼 x데이터를 넣어준다.)
prediction = model.predict([[750,3.70,3],[400,2.2,1]])
print(prediction)
#예측값
# [[0.75551134]
#  [0.02272356]] <-학점 2.2가지고 하버드 지원하면 붙을 확률 ㅋㅋ


#성능향상하는 법
#Dense()수를 늘리거나 노드숫자를 바꿔보거나 해본다.