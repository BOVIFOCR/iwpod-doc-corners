import cv2
import numpy as np
import argparse

if __name__ == "__main__":
    ps = argparse.ArgumentParser()
    ps.add_argument('--image', '-i', required=True)
    ps.add_argument('--label', '-l', required=True)
    ps.add_argument('--output', '-o', default="annotated.png")
    args = ps.parse_args()

    img_path = args.image
    label_path = args.label
    output_path = args.output

    with open(label_path, "r", encoding='utf-8') as fd:
        ls = fd.readlines()[0].split(",")
        ab = ls[:]
    pts = np.array([float(value) for value in ls[1:9]],dtype=np.float32).reshape((2,4))
    im = cv2.imread(img_path)

    npts = [[pts[0][0]*im.shape[1], pts[1][0]*im.shape[0]],
            [pts[0][1]*im.shape[1], pts[1][1]*im.shape[0]],
            [pts[0][2]*im.shape[1], pts[1][2]*im.shape[0]],
            [pts[0][3]*im.shape[1], pts[1][3]*im.shape[0]]]

    nim = cv2.polylines(im, [np.array(npts, dtype=np.int32)], True, (0,0,255), 10)
    cv2.imwrite(output_path, nim)