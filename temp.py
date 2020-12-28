from create_dataset import createDataset

file1 = open('/home/saksham/Desktop/research/mnt/ramdisk/max/90kDICT32px/lexicon.txt')

annotations = []

while True :
    line = file1.readline()
    
    if not line :
        break
    
    annotations.append(line.strip())

file1.close()

file2 = open('/home/saksham/Desktop/research/mnt/ramdisk/max/90kDICT32px/annotation_test.txt')

imagePathList = []
labelList = []

pth = '/home/saksham/Desktop/research/mnt/ramdisk/max/90kDICT32px'

while True : 
    line = file2.readline()
    
    if not line :
        break
    
    line = line.strip('\n')
    
    arr = line.split(' ')
    
    index = int(arr[1])
    
    path = arr[0][1:]
    
    path = pth + path
    
    imagePathList.append(path)
    labelList.append(annotations[index])

file2.close()

print("CREATING DATASET")

createDataset('/home/saksham/Desktop/research/data/test', imagePathList, labelList)