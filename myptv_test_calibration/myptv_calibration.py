#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 12:15:21 2019

@author: ron

A script for setting up the camera calibration

"""


from myptv.imaging_mod import camera, img_system
from myptv.calibrate_mod import calibrate
import matplotlib.pyplot as plt



cam_name = '2_camera'
resolution = 1280,1024
cal_points_fnams = './2_calibration_points'


#%%
cam = camera(cam_name, resolution, cal_points_fname=cal_points_fnams)
cam.load('./')

cal = calibrate(cam, cam.lab_points, cam.image_points, random_sampling=10)

print(cal.mean_squared_err())

#%%

import time

t0 = time.time()

cal.stochastic_searchCalibration(iterSteps=1000)

#cal.searchCalibration(maxiter=50)
#print(cal.mean_squared_err())

print('\ntime: %.2f'%(time.time()-t0))



