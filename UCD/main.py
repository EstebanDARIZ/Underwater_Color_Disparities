import numpy as np
import cv2
import os
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg')
from src.tool import load_image,newbluegreen,newred,ACES,\
    ada_color,High_pass,Overlay,maxmap
from src.seathru import run_pipeline
from src.contrast import integral_contrast

from config import IMG_PATH, DEPTH_PATH, RESULT_PATH    





def main():
    data = os.listdir(IMG_PATH)
    for i in range(len(data)):
        print("################################")
        path = IMG_PATH+ "/" + data[i]
        # print(path)
        # depth_file = depth_path + data[i][:-4] + '.npy'
        depth_file = DEPTH_PATH + "/" + data[i]
        # print(depth_file)
        # print("################################")
        # depth = np.array(np.load(depth_file)).astype(np.float32)
        depth = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED).astype(np.float32)

        i0 = load_image(path)
        (win,hei,_)=i0.shape

        if depth.ndim == 3:
            depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
        depth = cv2.resize(depth, (hei, win), interpolation=cv2.INTER_NEAREST)

        g = np.mean(i0[:, :, 1])
        b = np.mean(i0[:, :, 2])
        print(f"→ Image: {i0.shape}, Depth: {depth.shape}")

        if g>b:
            i0[:, :, 1],coefsR1,coefsR2 = run_pipeline(i0[:, :, 1], depth, 0.01, 2, 1,0.1)
            k=0
        else:
            i0[:, :, 2],coefsR1,coefsR2 = run_pipeline(i0[:, :, 2], depth, 0.01, 2, 1,0.1)
            k=1
        i0 = ACES(i0, 0.4)
        i1 = newbluegreen(i0, 1,k)
        i1 = newred(i1,1)
        i1 = np.maximum(i1, 0)
        i0 = np.maximum(i0, 0)
        i1 = maxmap(i0, i1, 1.2)
        if i1.shape[2] == 4:
            i1 = i1[:, :, :3]
        i1 = np.float32(integral_contrast(i1,20,20))
        i1 = np.float32(np.minimum(np.maximum(i1, 0), 1))
        i1 = ada_color(i1)
        i2 = High_pass(i1,5)
        i1 = Overlay(i1, i2)
        i1 = np.float32(np.minimum(np.maximum(i1, 0), 1))
        plt.imsave(RESULT_PATH+'%s.jpg'%(str(data[i][:-4])), i1)



if __name__ == "__main__":
    main()
