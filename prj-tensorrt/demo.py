import cv2
import scipy.io as sio
import os
from centerface import CenterFace
from tqdm import tqdm

centerface = CenterFace('models/tensorrt/centerface.trt')

def test_image_tensorrt(image_path, visualize_output=False):
    frame = cv2.imread(image_path)
    h, w = 1088, 1920  # old is 480* 640
    landmarks = True

    if landmarks:
        dets, lms = centerface(frame, h, w, threshold=0.35)
        print("count = ", len(dets))
    else:
        dets = centerface(frame, threshold=0.35)

    if visualize_output:
        for det in dets:
            boxes, score = det[:4], det[4]
            cv2.rectangle(frame, (int(boxes[0]), int(boxes[1])), (int(boxes[2]), int(boxes[3])), (2, 255, 0), 1)
        if landmarks:
            for lm in lms:
                for i in range(0, 5):
                    cv2.circle(frame, (int(lm[i * 2]), int(lm[i * 2 + 1])), 2, (0, 0, 255), -1)
        cv2.imshow('out', frame)
        cv2.waitKey(0)


if __name__ == '__main__':
    for i in tqdm(range(1000)):
        test_image_tensorrt('../prj-python/000388.jpg')
