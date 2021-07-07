import numpy as np 
import matplotlib.pyplot as plt
Height = np.array([172,144,182,169,186,160,152,165,170,175,168], dtype ='float32')
Height = Height.reshape((Height.size,1))
Height/=100
Weight  = np.array([65,40,77,66,80,50,42,65,70,75,63], dtype ='float32')
Weight = Weight.reshape((Weight.size,1))
bias = np.ones((Height.size,1) , dtype = 'float32')
X = np.concatenate((bias,Height,Weight),axis =1)
print(X)
BMI = np.array([21.97,19.29,23.25,23.11,23.12,19.53,18.17,23.875,24.221,24.49,22.321], dtype = 'float32')
BMI = BMI.reshape((BMI.size,1))
print(BMI)

Theta = np.linalg.inv(X.T.dot(X)).dot(X.T.dot(BMI))
print(Theta)
predict = np.dot(X,Theta)
print(predict)
plt.figure(1)
plt.title("Test",fontsize=20)
plt.plot(X[:,1]*100,BMI,'rx')
plt.plot(X[:,1]*100,predict,'b*')
plt.xlabel('Height (cm)',fontsize=16)
plt.ylabel('BMI',fontsize=16)
plt.legend(['Sample','Predict'])
plt.show()