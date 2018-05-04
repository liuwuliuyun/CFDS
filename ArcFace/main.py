#!/usr/bin/env python


import tensorflow as tf
import argparse
import numpy as np
import cv2
import time
from extractor import extractor
from aligner import aligner
import os
import random


parser = argparse.ArgumentParser()
parser.add_argument('--devices', default='/gpu:0')
parser.add_argument('--extractor_batch_size', default=256, type=int)
parser.add_argument('--aligner_batch_size', default=64, type=int)
args = parser.parse_args()
args.devices = args.devices.split(',')

config = tf.ConfigProto() 
config.allow_soft_placement = False
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


aligner = aligner(session, args.devices, args.aligner_batch_size)
extractor = extractor(session, args.devices, args.extractor_batch_size)


def batch_process(f, x, s):
    results = []
    for i in range(0,len(x), s):
        x_ = x[i: i + s]
        if len(x_) != s:
            x_ += [x_[0]] * (s - len(x_))
        y_ = f(x_)
        for j in y_:
            if len(results) < len(x):
                results.append(j)
        print(len(results), 'done')
    return results


def do_batch(f, x):
    keys = [i[1] for i in x]
    imgs = [i[0] for i in x]
    imgs = np.stack(imgs, axis=0)
    imgs = f(imgs)
    return list(zip(imgs, keys))


images = [(cv2.imread('data/%d.jpg' % i), 'data/%d.jpg' % i) for i in range(1,7)]
images = (images * 100)[:512] 
random.shuffle(images)
start = time.time()
images = batch_process(lambda x:do_batch(aligner.align, x), images, args.aligner_batch_size)
print('%d samples aligned in %.6fs' % (len(images), time.time() - start))
images_ = {}
for i in images:
    if i[1] not in images_:
        images_[i[1]] = i[0]
for i in images_:
    cv2.imwrite(i[:-4] + '_aligned.jpg', images_[i])
start = time.time()
images = batch_process(lambda x:do_batch(extractor.extract, x), images, args.extractor_batch_size)
print('%d samples extracted in %.6fs' % (len(images), time.time() - start))
images_  =  {}
for i in images:
    images_[i[1]] = i[0]
keys = sorted(images_.keys())
for k1 in keys:
    for k2 in keys:
        if k1 < k2:
            print('%s with %s sim %.6f' % (k1, k2, np.dot(images_[k1], images_[k2])))
