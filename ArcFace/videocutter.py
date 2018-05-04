import cv2
from argparse import ArgumentParser
import os

def parser():
	parser = ArgumentParser('VideoFrameCutter')
	parser.add_argument('--vpath',dest='vpath',help='Path to video',default='/SSH/data/testvideo.mp4',type=str)
	parser.add_argument('--interval',dest='interval',help='Frame interval when cutting video',default=100,type=int)
	parser.add_argument('--impath',dest='impath',help='Image Path to save',default='/SSH/data/selfmade/',type=str)
	return parser.parse_args()



if __name__=="__main__":
	args = parser()
	assert os.path.isfile(args.vpath),'Please provide a valid path to video file!'
	assert args.interval>0,'Interval should be large than zero!'
	
	v = cv2.VideoCapture(args.vpath)
	c = 1
	a = 0
	
	if v.isOpened():
		rval, frame=v.read()
	else:
		rval = False
	while rval:
		rval, frame=v.read()
		if(c%args.interval == 0):
			path = args.impath+'img'+str(a)+'.jpg'
			cv2.imwrite(path,frame)
			a = a + 1
		c = c + 1
	v.release()
