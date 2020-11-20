# Created by LiaoSteve on 2020-09-27
import os

Class = 'Cat'
image_list = sorted(os.listdir('./Cat'))
num = len(image_list)
# rename image list
os.chdir('./Cat')
for i in range(num):
    temp = image_list[i]
    if temp.endswith('.jpg'):
        temp = '.jpg'
    elif temp.endswith('.png'):
        temp = '.png'
    elif temp.endswith('.jpeg'):
        temp = '.jpeg'
    elif temp.endswith('.JPG'):
        temp = '.JPG'
    elif temp.endswith('.PNG'):
        temp = '.PNG'
    elif temp.endswith('.JPEG'):
        temp = '.JPEG'
    else:
        print(f'{temp}: format is not .jpg .jpeg or png')   
        continue 
    os.rename(image_list[i], Class + '_' + str(i)+temp)

