#对象是引用,基本类型是传值

a=[1000,1000]

b=a.copy()

b[0]+=1

print(a)