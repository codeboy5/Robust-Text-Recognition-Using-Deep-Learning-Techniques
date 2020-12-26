# import xml.dom.minidom
# import os

# charrss = set()

# directory = '/home/saksham/Desktop/urdu_ocr/urdu_benchmark'

# max_len = 0
# num = 0

# for filename in os.listdir(directory) :
#     if (filename.endswith('.xml')) :
#         f = os.path.join(directory, filename)
#         docs = xml.dom.minidom.parse(f)
#         items = docs.getElementsByTagName('name')
#         num += 1
#         for element in items:
#             if (element.firstChild) :
#                 txt = element.firstChild.data
#                 chars = txt.split(' ')
#                 max_len = max(max_len, len(chars))
#                 for c in chars:
#                     charrss.add(c)

# docs = xml.dom.minidom.parse('/home/saksham/Desktop/urdu_ocr/urdu_benchmark/UR5-100-500-page-015_1.xml')

# print(len(charrss))

# items = docs.getElementsByTagName('name')

# for element in items:
#     txt = element.firstChild.data
#     print(txt.split(' '))


# directory = r'C:\Users\admin'
# for filename in os.listdir(directory):
#     if filename.endswith(".jpg") or filename.endswith(".png"):
#         print(os.path.join(directory, filename))
#     else:
#         continue

# import cv2 as cv
# import tensorflow as tf
# import numpy as np


# #Dense to corresponding text removing Unidentified Character
# def dense_to_text(dense):
#     text=''
#     for num in dense:
#         if (num < len(chars)+1 and num > 0):
#             text+=chars[num-1]
#     return text


# Load Character set
# chars=''
# with open('/home/saksham/Desktop/urdu_ocr/chars.txt',encoding='utf-8') as f:
#     chars=f.read()

# for i in range(len(chars)) :
#     print(i, chars[i])
