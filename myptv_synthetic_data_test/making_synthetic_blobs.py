# -*- coding: utf-8 -*-
"""
By Ron Shnapp
April 29. 2022

A script for generating synthetic blob files from random particle clouds.

"""


import numpy as np
from myptv.imaging_mod import camera
from PIL import Image, ImageDraw




def Generate_synthetic_data(Np, segmentation_error,
                            generate_images=False, s=1, I=255):
    
    O = np.array([35.0, 30.0, 0.0])
    S = 35.0
    resolution = (1280, 1024)
    camera_names = ['cam1', 'cam2', 'cam3', 'cam4']    
    
    
    cameras = []
    for cn in camera_names:
        cameras.append(camera(cn, resolution))
        cameras[-1].load('./')
    
    
    x_list = (2*np.random.random(size=(Np,3))-1.0) * S + O
    np.savetxt('ground_truth', x_list, fmt='%.4f', delimiter='\t')
    
    
    for e,cam in enumerate(cameras[:]):
        
        if generate_images:
            
            res = cameras[0].resolution
            img = np.zeros((res[1], res[0]))
            radius = int(3*s)
            x_ = range(-radius, radius+1)
            X, Y = np.meshgrid(x_, x_)
            
            blob_image = I * np.exp(-( ((X)**2 + (Y)**2) /2/s**2))
            
            blobs_list = []
            for x in x_list:
                eta, zeta = cam.projection(x)
                t = np.random.random()*2*np.pi
                blobs_list.append([zeta + np.sin(t)*segmentation_error,
                                   eta + np.cos(t)*segmentation_error, 
                                   2, 2, 4, 0.0])
                
                cx, cy = np.round(np.array([eta,zeta]).astype(int))
                try:
                    i0,i1,j0,j1 = cy-radius,cy+radius+1,cx-radius,cx+radius+1
                    img[i0:i1, j0:j1] += blob_image
                except:
                    continue
                
            img[img>255] = 255
            img = img.astype('int8')
            
            pil_im = Image.fromarray(img, mode='L')
            pil_im.save('im_cam%d.tif'%(e+1))
            
            np.savetxt('blobs_cam%d'%(e+1), blobs_list,
                       fmt=['%.2f','%.2f','%d','%d','%d','%.1f'],
                       delimiter='\t')
            
        else:
            blobs_list = []
            for x in x_list:
                eta, zeta = cam.projection(x)
                t = np.random.random()*2*np.pi
                blobs_list.append([zeta + np.sin(t)*segmentation_error,
                                   eta + np.cos(t)*segmentation_error, 
                                   2, 2, 4, 0.0])
            np.savetxt('blobs_cam%d'%(e+1), blobs_list,
                       fmt=['%.2f','%.2f','%d','%d','%d','%.1f'],
                       delimiter='\t')


    
    
    
    

