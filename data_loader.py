from glob import glob
import json
import cv2
import os
import numpy as np

import argparse
from tqdm import tqdm

def rt270anti(bbox, wd):
    nbbox = []
    for x, y in bbox:
        nbbox.append([y, wd - x])
    return np.array(nbbox)

def rt180(bbox, wd, hg):
    nbbox = []
    for x, y in bbox:
        nbbox.append([hg - x, wd - y])
    return np.array(nbbox)

def rt90anti(bbox, hg):
    nbbox = []
    for x, y in bbox:
        nbbox.append([hg - y, x])
    return np.array(nbbox)


def rotate_bbox(bbox, height, width, degrees):
    if degrees == 270:
        return rt90anti(bbox, height)
    elif degrees == 180:
        return rt180(bbox, height, width)
    elif degrees == 90:
        return rt270anti(bbox, width)
    else:
        return bbox

def rotate_img(img, degrees):
    if degrees == 90:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif degrees == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    elif degrees == 270:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    else:
        return img

def rotate_all(image, annotation_json, degrees, synthed=False):
    wd, hg = int(image.shape[1]), int(image.shape[0])

    image = rotate_img(image, degrees)
    for i in range(len(annotation_json['region_s'])):
        reg = annotation_json['regions'][i]
        if not synthed:
            reg['points'] = rotate_bbox(reg['points'], hg, wd, degrees)
        else:
            reg['points'] = rotate_bbox([reg['points'][:2], reg['points'][2:]], wd, hg, degrees)
            reg['points'] = (*reg['points'][0],*reg['points'][1])
        
    if degrees in (90, 270):
        annotation_json['width'] = hg
        annotation_json['height'] = wd

    return image, annotation_json

def load_data(cfg):
    deg_files = cfg['degrees_files']
    degs_fs = {}
    for df in deg_files:
        with open(df, "r", encoding='utf-8') as fd:
            ls = [x.strip("\n").split(" ") for x in fd.readlines()]
        nn = {l[0].split(".")[0]:int(l[1]) for l in ls}
        degs_fs.update(nn)

    im_dirs = cfg['image_dirs']
    if "label_dirs" in cfg:
        if len(cfg['label_dirs']) > 1:
            ann_dirs = {im:an for im, an in zip(im_dirs, cfg['label_dirs'])}
        else:
            ann_dirs = {im:cfg['label_dirs'][0] for im in im_dirs}
    else:
        ann_dirs = {im: im for im in im_dirs}
    
    exts = cfg['extensions']
    ims = []
    for ext in exts:
        for d in im_dirs:
            ims += glob(f"{d}/*{ext}")

    anns = ims[:]
    degs = []
    for ext in exts:
        anns = [x.replace(ext, ".json") for x in anns]
    for i in range(len(anns)):
        d, f = os.path.split(anns[i])
        if f[:-5] in degs_fs.keys():
            degs.append(degs_fs[f[:-5]])
        else:
            degs.append(0)
            print(f)
        anns[i] = os.path.join(ann_dirs[d], f)
    return ims, anns, degs

def order_points(pts):
    '''
    Return a list of coordinates that will be ordered
    such that the first entry in the list is the top-left,
    the second entry is the top-right, the third is the
    bottom-right, and the fourth is the bottom-left
    '''
    rect = np.zeros((4, 2), dtype = "float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    axis_sum = pts.sum(axis = 1)
    rect[2] = pts[np.argmin(axis_sum)]
    rect[0] = pts[np.argmax(axis_sum)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect

def gen_closeup(im, pts):
    h, w = im.shape[:2]

    top_left = (pts[2][0], pts[2][1])
    bottom_left = (pts[3][0], pts[3][1])

    top_right = (pts[1][0], pts[1][1])
    bottom_right = (pts[0][0], pts[0][1])

    width =  (max(bottom_left[0], bottom_right[0])) - (min(top_left[0], top_right[0]))
    height = (max(top_right[1], bottom_right[1])) - (min(top_left[1], bottom_left[1]))

    gap_height = height*0.2
    gap_width = width*0.2

    top = min(top_left[1], top_right[1])
    top = int(max(top - gap_height, 0))

    bottom = max(bottom_left[1], bottom_right[1])
    bottom = int(min(bottom + gap_height, h))

    left = min(top_left[0], bottom_left[0])
    left = int(max(left - gap_width, 0))

    right = max(top_right[0], bottom_right[0])
    right = int(min(right + gap_width, w))

    close_im = im[top:bottom,left:right]

    close_pts = [
        [bottom_right[0] - left, bottom_right[1] - top],
        [top_right[0] - left, top_right[1] - top],
        [top_left[0] - left, top_left[1] - top],
        [bottom_left[0] - left, bottom_left[1] - top]
    ]

    return close_im, close_pts, width, height

def format_and_save(im, pts, out_location):
    hg, wd = im.shape[:2]

    top_left = pts[2][0]/wd, pts[2][1]/hg
    bottom_left = pts[3][0]/wd, pts[3][1]/hg

    top_right = pts[1][0]/wd, pts[1][1]/hg
    bottom_right = pts[0][0]/wd, pts[0][1]/hg

    formatted = f"4,{top_left[0]},{top_right[0]},{bottom_right[0]},{bottom_left[0]}" + \
                 f",{top_left[1]},{top_right[1]},{bottom_right[1]},{bottom_left[1]}"

    #im = cv2.polylines(im, [np.array(pts, dtype=np.int32)], True, (0,0,0), 3)
    cv2.imwrite(f"{out_location}.jpg", im)
    with open(f"{out_location}.txt", 'w', encoding='utf-8') as fd:
        fd.write(formatted)


def main(cfg):
    ims, anns, degs = load_data(cfg)
    ws = []
    rel_ws = []
    hs = []
    rel_hs = []

    lim = 0
    for i, a, d in tqdm(zip(ims[lim:], anns[lim:], degs[lim:]), total=len(ims)):
        # Load annotation and image
        with open(a, "r", encoding='utf-8') as fd:
            an = json.load(fd)
        base_im = cv2.imread(i)

        hg, wd = base_im.shape[:2]

        # Extract 4-point bounding box
        for reg in an:
            if 'name' in reg['region_shape_attributes'].keys() and \
                    reg['region_shape_attributes']['name'] == 'doc':
                wild_pts = reg['region_shape_attributes']['points']
                #wild_pts = order_points(np.array(wild_pts))
                wild_pts = rotate_bbox(wild_pts, hg, wd, d)
                wild_pts = order_points(np.array(wild_pts))
                break
        else:
            print(i)
        wild_im = rotate_img(base_im, d)
        hg, wd = wild_im.shape[:2]

        # Generate close-up version, format and save
        close_im, close_pts, width, height = gen_closeup(wild_im, wild_pts)

        format_and_save(wild_im, wild_pts, "rg_dataset/wild/" + os.path.split(a)[-1][:-5])
        format_and_save(close_im, close_pts, "rg_dataset/closeup/" + os.path.split(a)[-1][:-5])

        ws.append(width)
        hs.append(height)

        rel_ws.append(width/wd)
        rel_hs.append(height/hg)

    print("Doc width mean and stdev:", np.mean(ws), np.std(ws))
    print("Doc height mean and stdev:", np.mean(hs), np.std(hs))

    print("Avg aspect ratio:", np.mean(ws)/np.mean(hs))

    print("Relative width min and max:", np.min(rel_ws), np.max(rel_ws))
    print("Relative width avg:", np.average(rel_ws))
    print("Relative height min and max:", np.min(rel_hs), np.max(rel_hs))

if __name__ == "__main__":
    ps = argparse.ArgumentParser()
    ps.add_argument('--config', '-c', required=True)
    args = ps.parse_args()
    cfg_file = args.config

    with open(cfg_file, "r", encoding='utf-8') as fd:
        cfg = json.load(fd)
    main(cfg)
