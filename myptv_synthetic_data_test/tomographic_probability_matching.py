#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 14:42:50 2022

@author: ron
"""

import numpy as np



class tomographic_probability_matching(object):
    
    def __init__(self, imsys, images, ROI, sampling_window, voxel_size):
        '''
        input - 
        
        imsys - An instance of the myptv img_system with loaded cameras inside
        
        images - numpy arrays, (Ncams X Nx X Ny) that represent camera images.
                     
        ROI - A nested list of 3X2 elements. The first holds the minimum and 
              maximum values of x coordinates, the second is same for y, and 
              the third for z coordinates. 
              
        sampling_window - integer, the linear size of a square window with
                          which we are subsampling the images.  
        '''
        
        print('\ninitializing matcher\n')
        
        self.imsys = imsys
        self.images = images
        self.ROI = ROI
        self.sampling_window = sampling_window
        self.voxel_size = voxel_size
        
        self.Ncams = self.images[-1]
        
        # a list of the voxels through which rays have traversed
        self.traversed_voxel_list = []
    
    
    
    def create_voxels(self):
        
        # prepare the coordinates of the voxel centers:
        self.x = np.arange(self.ROI[0][0], self.ROI[0][1], self.voxel_size)
        self.y = np.arange(self.ROI[1][0], self.ROI[1][1], self.voxel_size)
        self.z = np.arange(self.ROI[2][0], self.ROI[2][1], self.voxel_size)
        
        self.grids = np.meshgrid(self.x, self.y, self.z)
        
        
        # setting up an array that counts the ray pass throuh voxels
        self.ray_couter = np.zeros((9, 9, 8))
        
    
        
    
    def get_a_range(self, O, r):
        '''
        Given a parametric ray, O + a*r, we seek for the range of a for which 
         the ray is within the ROI. If the ray is within the range, its 
         edge values are returned. Else None is returned.
        '''
        X0, X1 = self.ROI[0]
        ax0 = (X0 - O[0])/r[0]
        ax1 = (X1 - O[0])/r[0]
        
        Y0, Y1 = self.ROI[1]
        ay0 = (Y0 - O[1])/r[1]
        ay1 = (Y1 - O[1])/r[1]
        
        Z0, Z1 = self.ROI[2]
        az0 = (Z0 - O[2])/r[2]
        az1 = (Z1 - O[2])/r[2]
        
        ax0,ax1 = sorted([ax0,ax1])
        ay0,ay1 = sorted([ay0,ay1])
        az0,az1 = sorted([az0,az1])
        
        amin = max([ax0, ay0, az0])
        amax = min([ax1, ay1, az1])
        
        if amin > amax: 
            return None
        
        else:
            return amin, amax
    
    
    
    def ray_coord_generator(self, O, r):
        '''
        Takes in an origin and unit vector of a ray, and returns the points, 
        [x,y,z] through which it traverses inside self.ROI
        '''
        sample_step = self.voxel_size / 2.0
        amin, amax = self.get_a_range(O, r)
        a_range = np.arange(amin, amax, sample_step)
        
        relative_coords = np.outer(r, a_range)
        coords = np.repeat([O], len(a_range), axis=0).T + relative_coords
        return coords
        
        
        
    def ray_trace(self, O, r):
        '''
        This takes in a 3D line (O and r being an origin and a unit vector), 
        goes through the region of interest and writes the voxels through 
        which the line passes, and returns them.
        '''
        ray_coords = self.ray_coord_generator(O, r)
        
        x0 = self.x[0]
        dx = self.x[1] - self.x[0]
        xend = self.x[-1] + dx
        Nx = len(self.x)
        x_ind = ((ray_coords[0,:] - x0) / (xend - x0) * Nx).astype('uint')
        
        y0 = self.y[0]
        dy = self.y[1] - self.y[0]
        yend = self.y[-1] + dy
        Ny = len(self.y)
        y_ind = ((ray_coords[1,:] - y0) / (yend - y0) * Ny).astype('uint')
        
        z0 = self.z[0]
        dz = self.z[1] - self.z[0]
        zend = self.z[-1] + dz
        Nz = len(self.z)
        z_ind = ((ray_coords[2,:] - z0) / (zend - z0) * Nz).astype('uint')
        
        return set(zip(x_ind, y_ind, z_ind))
        
        
        
    def get_O_r(self, camNum, pixel_coords):
        '''
        Returns the camera origin and the r vector of a pixel coordinates
        '''
        # get the direction vector and camera center
        cam = self.imsys.cameras[camNum]
        #eta, zeta = pixel_coords
        zeta, eta = pixel_coords
        r = cam.get_r(eta, zeta)
        O = cam.O
        return O, r
    
    
    
    def cycle_image(self, camNum, threshold = 10):
        
        image = self.images[camNum,:,:]
        whr = np.where(image > threshold)
        for i,j in np.array(whr).T:
            O, r = self.get_O_r(camNum, (i,j))
            ray_voxels = [[vxl, image[i,j], camNum] for vxl 
                          in self.ray_trace(O, r)]
            
            self.traversed_voxel_list += ray_voxels
    
    
    
    
    
if __name__ == '__main__':
    
    from myptv.imaging_mod import camera, img_system
    import matplotlib.pyplot as plt


    cam1 = camera('cam1', (1280,1024))
    cam2 = camera('cam2', (1280,1024))
    cam3 = camera('cam3', (1280,1024))
    cam4 = camera('cam4', (1280,1024))
    
    cam_list = [cam1, cam2, cam3, cam4]
    
    for cam in cam_list:
        cam.load('.')


    imsys = img_system(cam_list)
    ROI = ((-10,80),(-10,80),(-40,40))


    imNames = ['im_cam%d.tif'%i for i in [1,2,3,4]]
    images = np.array([plt.imread(imname) for imname in imNames])
    
    sampling_window = 1.0
    voxel_size = 1.0

    tpm = tomographic_probability_matching(imsys, 
                                           images, 
                                           ROI, 
                                           sampling_window, 
                                           voxel_size)
    
    tpm.create_voxels()
    
    O  = np.array([ 24.0, 350.0, 374.0 ])
    r = np.array([0.00805127, 0.6303802 , 0.77624479])
    
    
    import time
    t0 = time.time()
    tpm.cycle_image(0, threshold=100)
    tpm.cycle_image(1, threshold=100)
    tpm.cycle_image(2, threshold=100)
    tpm.cycle_image(3, threshold=100)
    print(time.time()-t0)
    
    
    plot = False
    if plot:
        c1 = np.zeros(tpm.grids[0].shape)
        c2 = np.zeros(tpm.grids[0].shape)
        c3 = np.zeros(tpm.grids[0].shape)
        c4 = np.zeros(tpm.grids[0].shape)
        counts = [c1,c2,c3,c4]
        
        for (x,y,z),I,cn in tpm.traversed_voxel_list:
            try:
                counts[cn][x,y,z] += I
            except:
                continue
            
        totalVoxels = (c1>50) | (c2>50) | (c3>50) | (c4>50)
        
        colors = np.empty(totalVoxels.shape, dtype=object)
        colors[c1>50] = 'red'
        colors[c2>50] = 'blue'
        colors[c3>50] = 'green'
        colors[c4>50] = 'yellow'
        
        ax = plt.figure().add_subplot(projection='3d')
        v = ax.voxels(totalVoxels, alpha=0.25, facecolors=colors)
        
        countSum = np.sum(counts, axis=0)
        v = ax.voxels(countSum > 1000, facecolors='k')