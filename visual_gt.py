import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2
import csv
from tqdm import tqdm

def read_csv_file(filename):

    filenames = []
    rois = []
    classes = []
    with open(filename) as csvfile:
        i=['filename', 'rois', 'classes']
        csvdata = csv.DictReader(csvfile, fieldnames=i)
        for row in csvdata:
            filenames.append(row['filename'])
            rois.append(row['rois'])
            classes.append(row['classes'])

    return filenames, rois, classes

def read_labels(filepath, delim):

    classes, names, colors = [], [], []
    with open(filepath,'r') as file:
        lines = file.read().splitlines()
        for line in lines:
            cls, name, color = line.split(delim)
            classes.append(int(cls))
            names.append(name)
            colors.append(eval(color))

    return classes, names, colors


csv_path = '/media/patrick/WD-PBrand/coco/train.csv'
anchors_path = '/media/patrick/WD-PBrand/coco/coco_anchors.txt'

import matplotlib.patches as patches
from matplotlib.font_manager import FontProperties

csv_path = '/media/patrick/WD-PBrand/coco/train.csv'
csv_path = '/media/patrick/WD-PBrand/VOC/yolo/data.csv'
fs,rs,cs = read_csv_file(csv_path)

for index in range(0,6):

    f = fs[index]
    r = eval(rs[index])
    c = eval(cs[index])


    print(f)
    img = cv2.imread(f)
    #image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #image = image[::-1,:]

    H,W,_ = img.shape
    print('Width: {}\nHeight: {}'.format(W,H))

    print('Regions of Interest: {}'.format(r))
    #print('Class: {}'.format(ID))

    #fig,ax = plt.subplots(1)
    #ax.imshow(image, origin='lower')
    
    for j in range(len(r)):
        x,y,w,h = [float(y) for y in r[j]]
        ID = int(c[j])

        print('Normalized:')
        print('X: {}  Y: {} \nW: {}  H: {}\n'.format(x,y,w,h))

        m = int(x * W)
        n = int(h * H)
        o = int(w * W)
        p = int(h * H)
        #x = int((x + 0.5*w)*W)
        #y = int((y + 0.5*h)*H)
        #w = int(w*W)
        #h = int(h*H)

        bottom_left = (int((x-0.5*w)*W), int((y-0.5*h)*H))
        top_right = (int((x+0.5*w)*W),  int((y+0.5*h)*H))
        print('Formatted:')
        print('Min:')
        print('X: {}  Y: {} \nW: {}  H: {}'.format(bottom_left[0], bottom_left[1],o,p))
        print('Max:')
        print('X: {}  Y: {} \nW: {}  H: {}'.format(top_right[0], top_right[1],o,p))

        text = str(ID)#names[ID]
        
        cv2.rectangle(img, top_right, bottom_left, (0,255,0), 4)#colors[class_idx], 4)

        #rect = patches.Rectangle(bottom_left,w,h,linewidth=2,edgecolor='r',facecolor='none')

        #font = FontProperties()
        #font.set_family('sans-serif')
        #font.set_weight('bold')
        #font.set_size(10)
        #ax.text(bottom_left[0]+2, top_right[1]+8, text, color='black', fontproperties=font,
        #        bbox=dict(facecolor='red', edgecolor='red', pad=2))
        #ax.add_patch(rect)
        #ax.plot(x,y, 'g+', markersize=12)

    #plt.show()
    #cv2.rectangle(img, (48, 240), (195, 371), (255,0,0), 4)#colors[class_idx], 4)
    #cv2.rectangle(img, (8, 12), (352, 498), (255,0,0), 4)#colors[class_idx], 4)
    
    cv2.imshow('img', img)
    key = cv2.waitKey()
    if key == 27: break
