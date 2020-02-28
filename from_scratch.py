import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

dataset=pd.read_csv('diabetes.csv');

print(dataset.head());
print(dataset.shape);
print(dataset.describe());
print(dataset.isnull().any());

X = dataset[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']].values
Y = dataset['Outcome'].values

#normalization
#X = (X - X.mean())/X.std()
#X = (X - X.mean())/X.std()

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
X_train, y_train = shuffle(X_train, y_train)

y_train=y_train.reshape((614, 1))
#print(y_train.shape);

#hyperparameters 
epochs=100
learning_rate=0.01;
theta=np.zeros([1,8]);
bias=0;

#training phase

def Prediction(X,theta,bias,size=614):
    summation=X @ theta.T+bias
    activation=np.zeros(shape=(size,1))
    print(summation)
    j=0;
    
    for i in summation:
        if i>0:
          
            activation[j]=1

        else:   
            activation[j]=0 
        j=j+1
    #print(activation.shape)         
    return activation


def train(X,Y,theta,bias,learning_rate,epochs):
    for i in range(epochs):
        y_prediction=Prediction(X,theta,bias);
        
        #theta=theta+learning_rate * np.sum(X * (y_prediction - Y), axis=0)
        #bias= bias+(learning_rate) * np.sum(y_prediction-Y)
        theta = theta -(learning_rate/len(X)) * np.sum(X * (y_prediction - y_train), axis=0)
        bias= bias- (learning_rate/len(X)) * np.sum(y_prediction-y_train)
    return theta,bias
    
    
theta,bias=train(X_train,y_train,theta,bias,learning_rate,epochs)


print(y_test.shape)



y_hat=Prediction(X_test,theta,bias,154);
print(theta);
print(y_test),
print(y_hat)
y_hat=y_hat.reshape((154,));
loss=pd.DataFrame({'Acutal':y_test,'predicted':y_hat});
print(loss);

j=0;
for test, predict in zip(y_test, y_hat):
    if test==predict:
        j=j+1
#print(j)

