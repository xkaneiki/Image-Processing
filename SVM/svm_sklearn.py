from sklearn import svm            # 导入sklearn包的相应模块
import numpy as np
X = np.array([[0,0], [1,1]])
Y = np.array([0,1])
clf = svm.SVC()                    
clf.fit(X,Y)
print (clf.predict([[3.,6.]]))     # 用训练好的分类器对（3,6）进行分类
# [1]
print (clf.predict([[-1.,0.]]))    # 用训练好的分类器对（-1,0）进行分类
# [0]
print (clf.support_vectors_)       # 查看支持向量
# [[ 0.  0.]
#  [ 1.  1.]]
print (clf.support_)               # 查看支持向量类别
# [0 1]
print (clf.n_support_)             # 查看每个类别支持向量个数
# [1 1]
