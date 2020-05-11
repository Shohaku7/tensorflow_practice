# encoding=utf-8

iter=(list(range(10)))

filte=[1,2,3,4]
iter=filter(lambda x:x%2 == 0,filte)
print(type(filte))
print(list(iter))



print(type(iter))