from glob import glob
import json
import cv2
import shutil
import numpy as np


def order_points(pts):
    pts = np.array(pts)
    rect = np.zeros((4, 2), dtype = "float32")
    axis_sum = pts.sum(axis = 1)
    rect[2] = pts[np.argmin(axis_sum)]
    rect[0] = pts[np.argmax(axis_sum)]
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def warp_image(image, rect):
    (bottom_right, bottom_left, top_left, top_right) = rect

    width_a = np.sqrt(((bottom_right[0] - bottom_left[0]) ** 2) + \
                      ((bottom_right[1] - bottom_left[1]) ** 2))
    width_b = np.sqrt(((top_right[0] - top_left[0]) ** 2) + \
                      ((top_right[1] - top_left[1]) ** 2))
    max_width = max(int(width_a), int(width_b))

    height_a = np.sqrt(((top_right[0] - bottom_right[0]) ** 2) + \
                       ((top_right[1] - bottom_right[1]) ** 2))
    height_b = np.sqrt(((top_left[0] - bottom_left[0]) ** 2) + \
                       ((top_left[1] - bottom_left[1]) ** 2))
    max_height = max(int(height_a), int(height_b))
    dst = np.array([
        [max_height - 1, max_width - 1],
        [max_height - 1, 0],
        [0, 0],
        [0, max_width - 1]], dtype = "float32")
    transform_matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, transform_matrix, (max_height, max_width))
    return warped, transform_matrix

def warp_regions(lb, mat):
    rg = lb['regions']
    for k,v in rg.items():
        box = v['box']
        new_box = cv2.perspectiveTransform(np.float32([box]), mat)
        rg[k]['box'] = new_box[0].tolist()
    return rg

def project_all(im, lb, box):
    box = order_points(box)
    im, mat = warp_image(im, box)
    lb['width'] = im.shape[1]
    lb['height'] = im.shape[0]
    lb['regions'] = warp_regions(lb, mat)
    return im, lb

def annotate_image(im, lb):
    for k, v in lb['regions'].items():
        b = v['box']
        
        npts = [[b[0][0], b[0][1]],
                [b[0][0], b[1][1]],
                [b[1][0], b[1][1]],
                [b[1][0], b[0][1]]]

        im = cv2.polylines(im, [np.array(npts, dtype=np.int32)], True, (0,0,255), 10)
    return im


def read_rg_dataset(transform=False, image_files_only=True, prediction_box=False,
                pred_dir="w5_predictions", save_in_aux=False, image_dir="base_images"):

    src_ims = f"synthetic/{image_dir}"
    src_lbs = "synthetic/base_labels"

    imfs = sorted(glob(f"{src_ims}/*"))
    lbfs = sorted(glob(f"{src_lbs}/*"))
    ims = []
    lbs = []

    for imf, lbf in zip(imfs, lbfs):
        im = cv2.imread(imf) if not image_files_only else imf
        with open(lbf, "r", encoding='utf-8') as fd:
            lb = json.load(fd)
        lb['fname'] = imf
        if prediction_box:
            bbf = lbf.replace("base_labels", pred_dir)
            bbf = ".".join("_".join(bbf.split("_")[:-1]).split(".")[:-1]) + ".txt"
            with open(bbf, "r", encoding='utf-8') as fd:
                ls = fd.readlines()[0].split(",")
            ab = ls[:]
            pts = np.array([float(value) for value in ls[1:9]],dtype=np.float32).reshape((2,4))
            npts = [[pts[0][0]*im.shape[1], pts[1][0]*im.shape[0]],
                    [pts[0][1]*im.shape[1], pts[1][1]*im.shape[0]],
                    [pts[0][2]*im.shape[1], pts[1][2]*im.shape[0]],
                    [pts[0][3]*im.shape[1], pts[1][3]*im.shape[0]]]
            lb['doc_box'] = npts

        if transform:
            if image_files_only:
                im = cv2.imread(imf)
            rt = imf.split("\\")[-1]
            im, lb = project_all(im, lb, lb['doc_box'])      
            #print(lb['doc_box'])
            if save_in_aux:
                rt = imf.split("\\")[-1]
                cv2.imwrite(f"aux_im/{rt}", im)
                im = f"aux_im/{rt}"
        ims.append(im)
        lbs.append(lb)
    if image_files_only:
        ims = imfs
    return ims, lbs
