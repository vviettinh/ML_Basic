import numpy as np 
import matplotlib.pyplot as plt

def Loss(X,y,Theta):
    error = np.dot(X,Theta) - y
    m = np.size(y)
    J =np.round( (1/(2*m))*np.transpose(error)@error,15)
    return J
	
def Gradient_Descent(X, y,learning_rate=0.0001, iter=6000):
    theta =np.zeros([np.size(X,1),1], dtype = 'float32')
    X_T= np.transpose(X)
    m = np.size(y)
    pre_cost = Loss(X,y,theta)
    for i in range(0, iter):
        theta= theta - (learning_rate/m)*(X_T @ (np.dot(X,theta) - y))
        cost = Loss(X,y,theta)
        if(np.round(cost,15)==np.round(pre_cost,15)):
            break
        pre_cost=cost
        #print(cost,'\n')
    return theta

Height = np.array([172,144,182,169,186,160,152,165], dtype ='float32')
Height = Height.reshape((Height.size,1))
Height/=100
Weight  = np.array([65,40,77,66,80,50,42,65], dtype ='float32')
Weight  = Weight.reshape((Weight.size,1))
bias = np.ones((Height.size,1) , dtype = 'float32')
X = np.concatenate((bias,Height,Weight),axis =1)
print(X)
BMI = np.array([21.97,19.29,23.25,23.11,23.12,19.53,18.17,23.875], dtype = 'float32')
BMI = BMI.reshape((BMI.size,1))
print(BMI)
Theta= Gradient_Descent(X,BMI)
#Theta = np.linalg.inv(X.T.dot(X)).dot(X.T.dot(BMI))
print(Theta)
predict = np.dot(X,Theta)

plt.figure(1)
plt.plot(X[:,1]*100,BMI,'rx')
plt.plot(X[:,1]*100,predict,'b*')
plt.xlabel('Height (cm)',fontsize=16)
plt.ylabel('BMI',fontsize=16)
plt.legend(['Sample','Predict'])
plt.show()