import numpy as np
import time

s=np.array([
    [6,1,2],
    [2,5,4],
    [7,3,1]
])
t1=time.time()
print(t1)

print(np.argmax(s[:,0],axis=0))

print(s.shape)

print(time.time()-t1)

print(np.ones((2,2),dtype=np.int))

print(type(s[0]))
print(s[0].shape)
print(type(s))