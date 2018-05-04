from __future__ import print_function
import cv2
from argparse import ArgumentParser
import os

def parser():
	parser = ArgumentParser('Video Cutter!')
	parser.add_argument('-v',dest='vpath',default='/SSH/data/testvideo.mp4',type=str)
	parser.add_argument('-i',dest='ipath',default='/SSH/data/selfmade/',type=str)
	parser.add_argument('-c',dest='interval',default=100,type=int)
	return parser.parse_args()

if __name__ == "__main__":
	args = parser()
	
	c = 1
	a = 0
	assert os.path.isfile(args.vpath),'Please provide a valid video path'
	assert args.interval > 0,'Interval should be large than zero'
	
	v = cv2.VideoCapture(args.vpath)
	if v.isOpened():
		rval , frame = v.read()
	else:
		rval = False
	while rval:
		rval , frame = v.read()
		if (c%args.interval == 0):
			a = a + 1
			if a>99:
				path = args.ipath+'img'+str(a)+'.jpg'
			elif a>9 and a<100:
				path = args.ipath+'img0'+str(a)+'.jpg'
			else:
				path = args.ipath+'img00'+str(a)+'.jpg'
			print('Saving file: {0}'.format(path))
			cv2.imwrite(path, frame)
		c = c + 1
	v.release()
