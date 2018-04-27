# ------------------------------------------
# SSH: Single Stage Headless Face Detector
# Demo
# by Mahyar Najibi
# Add mtcnn aligner 
# Add face recognition module
# by Dongfeng Yu and Yun Liu
# ------------------------------------------

from __future__ import print_function
from SSH.wrappertest import detect
from argparse import ArgumentParser
import os
from utils.get_config import cfg_from_file, cfg, cfg_print
import caffe
import cv2
import tensorflow as tf
import numpy as np
import os
import demo_0
import time


def parser():
    parser = ArgumentParser('SSH Demo!')
    parser.add_argument('--im',dest='im_path',help='Path to the image',
                        default='data/demo/test4.png',type=str)
    parser.add_argument('--gpu',dest='gpu_id',help='The GPU ide to be used for detection',
                        default=0,type=int)
    parser.add_argument('--proto',dest='prototxt',help='SSH caffe test prototxt',
                        default='SSH/models/test_ssh.prototxt',type=str)
    parser.add_argument('--model',dest='model',help='SSH trained caffemodel',
                        default='data/SSH_models/SSH.caffemodel',type=str)
    parser.add_argument('--out_path',dest='out_path',help='Output path for saving the figure',
                        default='data/demo',type=str)
    parser.add_argument('--cfg',dest='cfg',help='Config file to overwrite the default configs',
                        default='SSH/configs/yliu_wider.yml',type=str)
    parser.add_argument('--conf',dest='conf',help='Confidence when detecting the image',
                        default=0.75,type=float)
    parser.add_argument('--devices',dest='devices',help='The GPU device to be used for recognition',
                        default='/gpu:0')
    parser.add_argument('--extractor_batch_size',dest='extractor_batch_size',help='Extractor batch size', 
                        default=5, type=int)
    parser.add_argument('--aligner_batch_size',dest='aligner_batch_size',help='Anigner batch size',
                        default=5, type=int)
    return parser.parse_args()

def resize_face(im, a, b, c, d, max_height, max_width, enlarge=2.0):
    temp_h = (d - b)/2.0
    c_y = (b + d)/2.0
    temp_w = (c - a)/2.0
    c_x = (a + c)/2.0
    temp_h*=enlarge
    temp_w*=enlarge
    n_a = max(0,int(c_x-temp_w))
    n_b = max(0,int(c_y-temp_h))
    n_c = min(max_width,int(c_x+temp_w))
    n_d = min(max_height,int(c_y+temp_h))
    face_cut = im[n_b:n_d, n_a:n_c]
    temp_length = max(n_d-n_b, n_c-n_a)
    blank_img = np.zeros((temp_length,temp_length,3),np.uint8)
    blank_img[:n_d-n_b,:n_c-n_a,:]=face_cut
    face_cut = cv2.resize(blank_img, (150, 150))
    return face_cut

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

if __name__ == "__main__":
    # Parse arguments
    args = parser()
    args.devices = args.devices.split(',')
    # Load the external config
    if args.cfg is not None:
        cfg_from_file(args.cfg)
    # Print config file
    #cfg_print(cfg)

    # Loading the network
    cfg.GPU_ID = args.gpu_id
    caffe.set_mode_gpu()
    caffe.set_device(args.gpu_id)
    assert os.path.isfile(args.prototxt),'Please provide a valid path for the prototxt!'
    assert os.path.isfile(args.model),'Please provide a valid path for the caffemodel!'

    #print('Loading the network...')
    net = caffe.Net(args.prototxt, args.model, caffe.TEST)
    net.name = 'SSH'
    #print('Loading complete! Total time usage: {0} s'.format(time.time()-start))
    #print('Detection Start...')
    # Read image
    assert os.path.isfile(args.im_path),'Please provide a path to an existing image!'
    # pyramid = True if len(cfg.TEST.SCALES)>1 else False
    # use pyramid = false currently
    pyramid = False 
    
    config = tf.ConfigProto() 
    config.allow_soft_placement = False
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)

    aligner = demo_0.aligner(session, args.devices, args.aligner_batch_size)
    extractor = demo_0.extractor(session, args.devices, args.extractor_batch_size)

    # Face cut list
    images = []

    #load image files
    #for fname in os.listdir(args.im_path):
	#if fname.lower().endswith(('.jpg','.png','.jpeg')):
	    #im_comp_dir = os.pathjoin(im_path,fname)
	    #TODO
    im = cv2.imread(args.im_path)
	
    start = time.time() 
    # Perform detection
    cls_dets,_ = detect(net,im,visualization_folder=args.out_path,visualize=False,pyramid=pyramid)
    slice_idx = 0
    for det in cls_dets:
        if det[4]>args.conf:
            slice_idx+=1
            temp_face = resize_face(im, det[0], det[1], det[2], det[3], im.shape[0], im.shape[1])
            #cv2.imwrite('data/demo'+str(slice_idx)+'.jpg',temp_face)
            images.append((temp_face,'facecut_'+str(slice_idx)))
    print('Detection Complete! Total time usage: {0} s'.format(time.time()-start))
    # Perform recognition
    print('Recognition Start...')

    images = batch_process(lambda x:do_batch(aligner.align, x), images, args.aligner_batch_size)
    #for i in images:
	#if i[0][1] == False:
	    #cv2.imwrite('/SSH/faltyimg_'+str(i)+'.jpg',i[0][0])
    images = [i[0] for i in images]
    print('%d samples aligned in %.6fs' % (len(images), time.time() - start))
    images_ = {}
    for i in images:
        if i[1] not in images_:
            images_[i[1]] = i[0]
    images = batch_process(lambda x:do_batch(extractor.extract, x), images, args.extractor_batch_size)
    print('%d samples extracted in %.6fs' % (len(images), time.time() - start))
    images_  =  {}
    for i in images:
        images_[i[1]] = i[0]
    #keys = sorted(images_.keys())
	
    print('Recognition Complete! Total Time is {0}'.format(time.time()-start))
