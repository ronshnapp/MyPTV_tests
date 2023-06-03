# -*- coding: utf-8 -*-
"""
By Ron Shnapp
April 29. 2022


This script takes in a list of particles in 3D and a MyPTV
camera object, makes an image for the particles coordinates on the 
camera sensor.

"""


import numpy as np
from myptv.imaging_mod import camera
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt






def generate_projection_image(camera, particles, s=1, I=255):
    '''
    input - 
    
    camera - instance of the camera class
    particles - a list of Lab-space 3D particle coordinates
    s - standard deviation of the Gaussian blobs' imaes
    I - intensity of the Gaussian blobs
    '''
    
    particles = particles[:,:3]
    
    img = np.zeros((camera.resolution[1],camera.resolution[0]))
    radius = int(3*s)
    x_ = range(-radius, radius+1)
    X, Y = np.meshgrid(x_, x_)
    
    blob = I * np.exp(-( ((X)**2 + (Y)**2) /2/s**2))
    

    for i in range(len(particles)):
        
        proj = camera.projection(particles[i])
        cx, cy = np.round(proj.astype(int))
        img[cy-radius:cy+radius+1,cx-radius:cx+radius+1] += blob
        
    img[img>255] = 255
    img = img.astype('int8')
    
    return img
    






p_list = np.loadtxt('particles')[:,:3]
cam = camera('cam1', (1280,1024))
cam.load('./')


projection = generate_projection_image(cam, p_list,s=1, I=128)
img = Image.fromarray(projection, mode='L')
img.save('im_cam1_proj.tif')




    
    
    

