from ultralytics import YOLO
from glob import glob
from time import time
import numpy as np
import tqdm
from shapely.geometry import box, Polygon

def iou(l1, l2):
    pt1 = np.array(l1.cpu())#.pts
    pt2 = np.array(l2).reshape((2,4))#.pts

    p1 = Polygon(pt1)
    #p1 = Polygon([[pt1[0][0], pt1[1][0]], [pt1[0][1], pt1[1][1]],
    #              [pt1[0][2], pt1[1][2]], [pt1[0][3], pt1[1][3]]])
    #p2 = Polygon([[], [], [], []])
    p2 = Polygon([[pt2[0][0], pt2[1][0]], [pt2[0][1], pt2[1][1]],
                  [pt2[0][2], pt2[1][2]], [pt2[0][3], pt2[1][3]]])

    inter = p1.intersection(p2).area
    union = p1.union(p2).area
    return inter/union

model_name = "runs/obb/yolo11n_2000_ptd/weights/best.pt" 

yolo = YOLO(model_name)

exs = list(glob("datasets/nbid_yolo/images/*"))
example = "./rg_dataset/train_dir/119c16d1-7684-431f-8b7c-891dc1d6f0ad.jpg"

#st = time()
#rs = yolo(exs, batch=1)
#end = time()
speeds = []
ious = []
outs = []


for f in exs:
	r = yolo(f)
	#print(r)
	#print(r[0].obb.xyxyxyxyn)
	#print(r[0].speed)
	speeds.append(r[0].speed['preprocess'] + r[0].speed['inference'] + r[0].speed['postprocess'])
	with open(f.replace("images", "labels").split(".")[0] + ".txt", "r") as fd:
		gt = [float(x) for x in fd.readlines()[0].split(" ")[1:]]
	#print(gt)
	ious.append(iou(r[0].obb.xyxyxyxyn[0], gt))
	ob = r[0].obb.xyxyxyxyn[0]
	outs.append(f"4,{ob[2][0]},{ob[2][1]},{ob[0][0]},{ob[0][1]},{ob[3][0]},{ob[3][1]},{ob[1][0]},{ob[1][1]}")
	with open(f"yolo_results/yolon_2000/{f.split('/')[-1][:-4]}.txt", "w") as fd:
		fd.write(outs[-1])

#print(((end-st)*1000)/len(exs))
print(np.mean(speeds), np.std(speeds))
print(np.mean(ious), np.std(ious))
#print(sorted(ious))
