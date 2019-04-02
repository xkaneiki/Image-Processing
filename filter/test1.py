import numpy as np

a=np.array(
    [
        [[1,2,3],[2,3,4]],
        [[7,8,9],[10,11,12]],
        [[12,13,14],[15,16,17]],
    ]
)

print(a.shape)

print(a[:,:,0])

print(np.min(a,axis=2))

tmp=a.copy()

tmp[0][0][0]=10000

print(a[0][0][0])