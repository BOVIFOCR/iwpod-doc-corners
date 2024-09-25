import os
import json
import numpy as np
from src.keras_utils import load_model, reconstruct_new
import cv2
from src.keras_utils import  detect_lp_width
from src.utils                  import  im2single, image_files_from_folder, IOU_labels
from src.drawing_utils          import draw_losangle
from src.data_generator_tf2 import ALPRDataGenerator
import argparse
import time

from shapely.geometry import box, Polygon
from train_iwpodnet_tf2 import load_network

def iou(l1, l2):
    pt1 = l1.pts
    pt2 = l2.pts

    p1 = Polygon([[pt1[0][0], pt1[1][0]], [pt1[0][1], pt1[1][1]],
                  [pt1[0][2], pt1[1][2]], [pt1[0][3], pt1[1][3]]])
    #p2 = Polygon([[], [], [], []])
    p2 = Polygon([[pt2[0][0], pt2[1][0]], [pt2[0][1], pt2[1][1]],
                  [pt2[0][2], pt2[1][2]], [pt2[0][3], pt2[1][3]]])

    inter = p1.intersection(p2).area 
    union = p1.union(p2).area
    return inter/union


def detect_lp_width(model, I,  MAXWIDTH, net_step, out_size, threshold):

    #
    #  Resizes input image and run IWPOD-NET
    #

    # Computes resize factor
    factor = min(1, MAXWIDTH/I.shape[1])
    w,h = (np.array(I.shape[1::-1],dtype=float)*factor).astype(int).tolist()

    # dimensions must be multiple of the network stride
    w += (w%net_step!=0)*(net_step - w%net_step)
    h += (h%net_step!=0)*(net_step - h%net_step)

    # resizes image
    Iresized = cv2.resize(I,(w,h), interpolation = cv2.INTER_CUBIC)
    T = Iresized.copy()

    # Prepare to feed to IWPOD-NET
    T = T.reshape((1,T.shape[0],T.shape[1],T.shape[2]))

    #
    #  Runs LP detection network
    #
    start   = time.time()
    Yr      = model.predict(T)
    Yr      = np.squeeze(Yr)
    elapsed = time.time() - start
    #print(Yr.shape)
    #
    # "Decodes" network result to find the quadrilateral corners of detected plates 
    #
    L,TLps = reconstruct_new (I, Iresized, Yr, out_size, threshold)

    return L,TLps,elapsed


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-d'        ,'--dataset'          ,type=str   , default = None        ,help='Input Image')
    parser.add_argument('-m'        ,'--model'          ,type=str   , default = None        ,help='Input Image')
    parser.add_argument('-p'        ,'--partition'          ,type=str   , default = "test"        ,help='Input Image')
    parser.add_argument('-si'        ,'--save_images'          , action = 'store_true'     ,help='Input Image')
    #parser.add_argument('-v'        ,'--vtype'          ,type=str   , default = 'fullimage'     ,help = 'Image type (car, truck, bus, bike or fullimage)')
    parser.add_argument('-t'        ,'--lp_threshold'   ,type=float   , default = 0.35      ,help = 'Detection Threshold')
    parser.add_argument('-bs'       ,'--batch-size'         ,type=int   , default = 52                      ,help='Mini-batch size (default = 64)')


    args = parser.parse_args()
    dts = args.dataset
    model_dir = args.model
    batch_size = args.batch_size
    save_ims = args.save_images
    lp_threshold = args.lp_threshold
    partition = args.partition
    ocr_input_size = [int(100*1.5364), 100] # desired LP size (width x height)

    #dim = 208 # set by authors in training

    ASPECTRATIO = 1.0 #max(1, min(2.75, 1.0*Ivehicle.shape[1]/Ivehicle.shape[0]))  # width over height
    WPODResolution = 208 # faster execution
    lp_output_resolution = tuple(ocr_input_size[::-1])

    #
    #  Parameters of the method
    #
    #lp_threshold = 0.35 # detection threshold

    #
    #  Loads network and weights
    #
    iwpod_net, _, _, _ = load_network(model_dir, 208)

    # Data
    Data = image_files_from_folder(dts, partition=partition)
    print(len(Data))
    #train_generator = ALPRDataGenerator(
    #                      Data, batch_size = batch_size, dim = dim, OutputScale = 1.0)

    if save_ims:
        im_dir = os.path.join(model_dir, "images")
        if not os.path.exists(im_dir):
            os.mkdir(im_dir)
    stats = {'times': {}, 'ious': [],
            'found': 0, 'not_found': 0, 'ratio_found': 0, 'm_iou': 0}
    mel = 0

    for idx,i in enumerate(Data):
        img, gt = i
        iwh = np.array(img.shape[1::-1],dtype=float).reshape((2,1))
        L,TLps,elapsed = detect_lp_width(
                             iwpod_net,
                             im2single(img),
                             WPODResolution*ASPECTRATIO,
                             2**4, lp_output_resolution, lp_threshold)
        ordered = sorted([x for x in L], key = lambda x:iou(x, gt[0]), reverse=True)
        mel += elapsed

        # paint image with gt (green - bGr) and prediction (red - bgR)
        if save_ims and len(ordered) > 0:
            draw_losangle(img, ordered[0].pts*iwh, color = (0,0,255.), thickness = 2)
            draw_losangle(img, gt[0].pts*iwh, color = (0,255.,0), thickness = 2)
            cv2.imwrite(f"{im_dir}/{idx}.jpg", img)

        if len(ordered) > 0:
            if len(ordered) not in stats['times'].keys():
                stats['times'][len(ordered)] = {'occurrences': 0, 'ms': 0}

            stats['times'][len(ordered)]['ms'] += elapsed*1000
            stats['times'][len(ordered)]['occurrences'] += 1

            stats['found'] += 1
            stats['m_iou'] += iou(gt[0], ordered[0])
            stats['ious'].append(iou(gt[0], ordered[0]))
        else:
            if len(ordered) not in stats['times'].keys():
                stats['times'][0] = {'occurrences': 0, 'ms': 0}
            stats['times'][0]['ms'] += elapsed*1000
            stats['times'][0]['occurrences'] += 1
            stats['not_found'] += 1

    for i in stats['times'].keys():
         stats['times'][i]['ms'] /= stats['times'][i]['occurrences']
    stats['m_iou'] /= stats['found']
    stats['ratio_found'] = stats['found'] / (stats['found'] + stats['not_found'])
    with open(os.path.join(model_dir, 'stats.json'), 'w') as fd:
        json.dump(stats, fd, indent=4)

    print("mIOU: ", stats['m_iou'])
    print("Min mIOU: ", min(stats['ious']))
    print("Max mIOU: ", max(stats['ious']))
    print("Med mIOU: ", np.median(stats['ious']))
    print("IOU stdev: ", np.std(stats['ious'][1:]))

    print("Medium elapsed: ", mel*1000/len(Data))
    print("% Found: ", stats['ratio_found'], " (", stats['found'], "/", stats['not_found'], ")")
    #res = iwpod_net.predict(train_generator)
    #print(res.shape, train_generator)





