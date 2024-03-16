import math
import os

'''
TODO:
1. Merge all 01-40, 41-80 loss value file to one file respectively.
2. Sort the file
3. According to the sorted file, classify the annotation file into 4 new folders (70p, 80p, 90p, 100p). With format: video frame 7annotations
'''

root_dir_first40 = "/mnt/ceph/tco/TCO-Students/Projects/KP_curriculum_learning/lossByFrames1-40/"
root_dir_second40 = "/mnt/ceph/tco/TCO-Students/Projects/KP_curriculum_learning/lossByFrames41-80/"

files = os.listdir(root_dir_second40)

whole = open("/mnt/ceph/tco/TCO-Students/Projects/KP_curriculum_learning/LossByFramesIntegrated/lossByFrames41-80/", 'r')

buffer = []

for file in files:
    with open(file) as f:
        for line in f:
            item = f.name + '-' + line
            buffer.append(item)


sorted(buffer, key=lambda x:x.split(',')[1])

sortedList = []
for item in buffer:
    sortedList.append(item.split(',')[0])

#TODO: sort the file