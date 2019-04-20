import numpy as np
t=np.array([
    [1,1,1],
    [2,2,2],
    [3,3,3],
])

t1=np.array([1,1,1])
t2=np.array([2,2,2])
t3=np.array([3,3,3])

print(np.stack([t[0,:],t[1,:],t[2,:]]))
print(np.hstack([t1,t2]))