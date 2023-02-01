import os


count = 0
path = './best'+str(count)
while(os.path.isfile(path)):
    count+=1
    path='./best'+str(count)
print(count)
