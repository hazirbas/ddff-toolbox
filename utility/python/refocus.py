#! /usr/bin/python3

import numpy as np
import scipy.io
import cv2
import subpixel_shift

def refocus(light_field, calib_mat, output_folder, stack_size=10):
    #Load calibration parameters
    mat = scipy.io.loadmat(calib_mat)
    if 'IntParamLF' in mat:
        mat = np.squeeze(mat['IntParamLF'])
    else:
        return
    K2 = mat[1]
    fxy = mat[2:4]
    flens = max(fxy)
    fsubaperture = 521.4052 # pixel
    baseline = K2/flens*1e-3 # meters

    depth_range = [0.5, 7]
    disparity_range = (baseline*fsubaperture / depth_range)
    disparities = np.linspace(disparity_range[0], disparity_range[1], num=stack_size)

    #Read light field image according to instructions given at http://hazirbas.com/datasets/ddff12scene/
    lf = np.load(light_field) / 255.0

    for idx, disparity in enumerate(disparities):
        lfsize = (lf.shape[2], lf.shape[3])
        uvcenter = np.asarray((np.asarray([lf.shape[0],lf.shape[1]])+1)/2)
        image = np.zeros( lfsize + (3,))

        for u in range(lf.shape[0]):
            for v in range(lf.shape[1]):
                shift = (uvcenter - np.asarray([u+1,v+1])) * disparity
                shifted = subpixel_shift.subpixel_shift(
                    np.fft.fft2(np.squeeze(lf[u,v]), axes=(0,1)),
                    shift,
                    lfsize[0],
                    lfsize[1],
                    1)
                image = image + shifted

        
        image = image / np.prod([lf.shape[0], lf.shape[1]])
        image = np.uint8(image * 255.0)

        #Convert RGB to BGR (OpenCV assumes image to be BGR) and write output image
        cv2.imwrite(output_folder + "/" + "{0:02d}".format(idx+1) + ".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    refocus('LF_0001.npy', '../../caldata/lfcalib/IntParamLF.mat', './')
