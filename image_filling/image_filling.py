# -*- coding: utf-8 -*-
#!/usr/bin/env python3

"""
Created on Mon Oct  2 22:33:16 2023

@author: ron


An algorithm that uses 2D tracking in order to fill in errors in 
the segmentation stage.

The idea is that we take a blob file, then apply 2D tracking, then perform
trajectory stitching, then use the stitched trajectories to fill out missing
bits in the segmentation through interpolation, and finally save the old with 
the interpolated data. 

"""


# Step 1: Do the 2D tracking of blobs:
    
    
#from pandas import read_csv

from numpy import array, savetxt
from myptv.tracking_2D_mod import track_2D
from myptv.imaging_mod import camera
from myptv.traj_smoothing_mod import smooth_trajectories
from myptv.traj_stitching_mod import traj_stitching





class image_filling(object):
    '''
    A class used to perform the image filling process to improve segmentation.
    '''
    
    
    def __init__(self, blob_fname, max_d, max_dv, smooth_window=3, 
                 polyorder=2, Ts=2, dm=None, only_tracked=True, 
                 save_tracking_results=False):
        '''
        A class used to perform the image filling process to improve 
        segmentation. 
        
        Input:
            blob_fname (string) - path or file name of the blob file following
                                  segmentation.
            
            max_d (float) - the maximum translation in nearest neighbour 
                            2D tracking.
            
            max_dv (float) - the maximum change of velocity in the 2D four
                             frame tracking
            
            smooth_window (int) - the smoothing window size for the 2D data 
            
            polyorder (int) - the order of the polynomial used in the 2D 
                              data smoothing
            
            Ts (int) - the number of frames difference allowed for stitching
                       of the 2D data
            
            dm (float) - The maximum distance allowed for the stitching of the
                         2D data
            
            only_tracked (bool) - If true, this will save only the blobs that 
                                  were successfully tracked in 2D
            
            save_tracking_results (bool) - If True, when saving the results, 
                                           another column will be used in order
                                           to signal the trajectory ID found
                                            in the 2D tracking. Importantly:
                                           this changes the format of the saved 
                                           file to have 6 columns instead of 5!
                                           
        '''
        self.blob_fname = blob_fname
        self.max_d = max_d
        self.max_dv = max_dv
        self.w = smooth_window
        self.polyorder = polyorder
        self.Ts = Ts
        if dm is None: 
            self.dm = self.max_d
        else:
            self.dm = dm
        self.only_tracked = only_tracked
        self.save_tracking_results = save_tracking_results
            
            
          
        # initiate the 2D tracker:
        cam = None
        z_particles = 1.0
        self.tracker = track_2D(cam, self.blob_fname, z_particles, d_max = d_max, 
                           dv_max = dv_max)
        
        self.tracker.blobs_to_particles()
        
        
    def do_image_filling(self):
        '''
        This performs the actual steps of the image filling algorithm -
        1) tracking in 2D
        2) smoothing in 2D
        3) stitching in 2D
        
        The returns are stored as XXX
        '''
        
        # 1) tracking
        self.tracker.track_all_frames()
        tr = self.tracker.return_connected_particles()
        
        # 2) smoothing
        sm = smooth_trajectories(tr, self.w, self.polyorder)
        sm.smooth()
        tr2 = sm.smoothed_trajs
        
        # 3) stitching
        st = traj_stitching( array(tr2), Ts, dm)
        st.stitch_trajectories()
        
        self.new_traj_list = st.new_traj_list
        
        
    
    def save_results(self, save_name):
        '''
        The the newly updated blob data on the disk as a blob file.
        '''
        if self.only_tracked:
            to_save = self.new_traj_list[:, [0, 1, 2, 3, 4, 5, -1]]
            to_save[:,3] = 1
            to_save[:,4] = 1
            to_save[:,5] = 1
            
            if self.save_tracking_results:
                to_save = [p[:] for p in filter(lambda x: x[0]!=-1, to_save)]
            
            else:
                to_save = [p[1:] for p in filter(lambda x: x[0]!=-1, to_save)]
            
        else:
            if self.save_tracking_results:
                to_save = self.new_traj_list[:, [0, 1, 2, 3, 4, 5, -1]]
                to_save[:,3] = 1
                to_save[:,4] = 1
                to_save[:,5] = 1
                
            else:
                to_save = self.new_traj_list[:, [1, 2, 3, 4, 5, -1]]
                to_save[:,2] = 1
                to_save[:,3] = 1
                to_save[:,4] = 1
            
        to_save = sorted(to_save, key=lambda x: x[-1])
        
        if self.save_tracking_results:
            fmt = ['%d', '%.2f', '%.2f', '%d', '%d', '%d', '%d']
        
        else:
            fmt = ['%.2f', '%.2f', '%d', '%d', '%d', '%d']
            
        savetxt(save_name, to_save, fmt=fmt, delimiter='\t')






fname = '/home/ron/Desktop/Research/plankton_sweeming/experiments/20221229/run79/blobs_cam1'
d_max = 2.0
dv_max = 2.0
window = 3
polyorder = 2
Ts = 3
dm = 3


ImF = image_filling(fname, d_max, dv_max, smooth_window=window, 
                    polyorder=polyorder, Ts=Ts, dm=dm, 
                    only_tracked=False, 
                    save_tracking_results=True)

ImF.do_image_filling()

ImF.save_results('/home/ron/Desktop/Research/plankton_sweeming/experiments/20221229/run79/filled_blobs_cam1')


#%%
import numpy as np
import matplotlib.pyplot as plt




trajs = {}
for tr_ in tr:
    try: trajs[tr_[0]].append(tr_)
    except: trajs[tr_[0]] = [tr_]
    
ids = trajs.keys()

fig, ax = plt.subplots()

for k in ids:
    tr_ = np.array(sorted(trajs[k], key = lambda x: x[-1]))
    if k!=-1:
        ax.plot(tr_[:,1], tr_[:,2], 'X-b')
    else:
        ax.plot(tr_[:,1], tr_[:,2], 'Xr')

ax.set_aspect('equal')








trajs = {}
for tr_ in tr3:
    try: trajs[tr_[0]].append(tr_)
    except: trajs[tr_[0]] = [tr_]
    
ids = trajs.keys()

for k in ids:
    tr_ = np.array(sorted(trajs[k], key = lambda x: x[-1]))
    if k!=-1:
        ax.plot(tr_[:,1], tr_[:,2], 'o-g', ms=2.5, lw=0.5)

ax.set_aspect('equal')


