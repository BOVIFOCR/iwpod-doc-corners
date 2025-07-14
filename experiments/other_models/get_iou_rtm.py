from shapely.geometry import box, Polygon
from mmdet.apis import init_detector, inference_detector
import mmcv
import time
import numpy as np
from glob import glob
import cv2
import math
import supervision as sv

def rtt(p1,o,s,c):
	p1s = (p1[0]-o[0],p1[1]-o[1])
	p1t = (p1s[0]*c-p1s[1]*s, p1s[0]*s+p1s[1]*c)
	return (p1t[0]+o[0], p1t[1]+o[1])

def get_right(preds):
	o = (preds[0], preds[1])
	h = preds[2]
	w = preds[3]
	
	s = math.sin(np.deg2rad(-preds[4]))
	c = math.cos(np.deg2rad(-preds[4]))
	p1 = (o[0]-w/2,o[1]-h/2)
	p1n = rtt(p1,o,s,c)

	p2 = (o[0]+w/2,o[1]-h/2)
	p2n = rtt(p2,o,s,c)

	p3 = (o[0]+w/2,o[1]+h/2)
	p3n = rtt(p3,o,s,c)

	p4 = (o[0]-w/2,o[1]+h/2)
	p4n = rtt(p4,o,s,c)

	return np.array([p1n[0], p1n[1], p2n[0], p2n[1], p3n[0], p3n[1], p4n[0], p4n[1]])

def iou(l1, l2):

    p1 = Polygon([[l1[0], l1[1]], [l1[2], l1[3]],
                  [l1[4], l1[5]], [l1[6], l1[7]]])
    p2 = Polygon([[l2[0], l2[1]], [l2[2], l2[3]],
                  [l2[4], l2[5]], [l2[6], l2[7]]])

    inter = p1.intersection(p2).area
    union = p1.union(p2).area
    return inter/union


config = "./configs/rtmdet/rotated/tiny_nbid_f1.py"
gts = []
pds = []

ious = []

for i in range(1, 2):
	model_file = f"./work_dirs/tiny_nbid_f{i}/epoch_100.pth"

	model = init_detector(config, model_file, device='cpu')

	all_tms = 0
	fs = glob(f".data/nbid_detr/images/*.png")

	for f in fs:
		root = f.split("/")[-1]
		img = mmcv.imread(f)
		result = inference_detector(model, img)
		pd = result.pred_instances.bboxes[np.argmax(result.pred_instances.scores)]

		with open(f"{f[:-4].replace('images', 'labelTxt')}.txt", "r") as fd:
			gt = fd.readlines()[0].split(" ")[:8]
		pd = get_right(pd)
		ious.append(iou(gt, pd))
		im = cv2.imread(f)
		h, w = im.shape[:2]
		pd[:4] /= w
		pd[4:] /= h
		with open(f"preds/{root[:-4]}.txt", "w") as fd:
			fd.write(f"f,{pd[0]},{pd[2]},{pd[4]},{pd[6]},{pd[1]},{pd[3]},{pd[5]},{pd[7]}")

	print(np.mean(ious))
print(np.mean(ious))


