"""
segment_image.py: A simple python application demonstrating semantic segmentation in images.
"""

__author__ = "S Sathish Babu"
__date__   = "03-01-2021 Sunday 05:00"
__email__  = "bumblebee211196@gmail.com"

import argparse

import cv2
import numpy as np

parser = argparse.ArgumentParser('SemanticSegmentation')
parser.add_argument('-i', '--image', help='Path to the image file', required=True)
args = vars(parser.parse_args())

CLASSES = open('./resources/classes.txt', 'r').read().strip().split('\n')
COLORS = open('./resources/colors.txt', 'r').read().strip().split('\n')
COLORS = [np.array(color.split(',')).astype('int') for color in COLORS]
COLORS = np.array(COLORS, dtype='uint8')

legend = np.zeros(((len(CLASSES) * 25) + 25, 300, 4), dtype='uint8')

for (i, (class_name, color)) in enumerate(zip(CLASSES, COLORS)):
    color = [int(c) for c in color]
    cv2.putText(legend, class_name, (5, (i * 25) + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.rectangle(legend, (100, (i * 25)), (300, (i * 25) + 25), tuple(color), -1)

net = cv2.dnn.readNet('./resources/enet-model.net')

image = cv2.imread(args['image'])
blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (1024, 512), 0, swapRB=True, crop=False)
net.setInput(blob)
output = net.forward()

num_classes, height, width = output.shape[1:4]
color_map = np.argmax(output[0], axis=0)

mask = COLORS[color_map]
mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

output = ((0.4 * image) + (0.6 * mask)).astype('uint8')

cv2.imshow('Legend', legend)
cv2.imshow('Output', output)
cv2.waitKey(0)
cv2.destroyAllWindows()
