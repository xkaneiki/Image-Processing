import numpy as np
from sklearn import svm

x=np.random.ranf(size=(1000,10))
y=np.reshape(np.sum(np.sin(x),axis=1),(1000,1))
y=np.where(y>=4,1,-1)

clf=svm.SVC()
clf.fit(x,y)


x_=np.random.ranf(size=(1000,10))
y_=np.reshape(np.sum(np.sin(x_),axis=1),(1000,1))
y_=np.where(y_>=4,1,-1)
_y=np.reshape(clf.predict(x_),(1000,1))

cn=0
for i in range(1000):
    if y_[i][0]==_y[i][0]:
        cn+=1

print(cn)