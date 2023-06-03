# -*- coding: utf-8 -*-
"""
Created on Sun 20 March 2022

A script to manually do the matching of the synthetic data
with MyPTV

"""
import numpy as np
from myptv.particle_matching_mod import match_blob_files
from myptv.imaging_mod import camera, img_system
import time




def match_synthetic_data(voxel_size, max_err, save_name = 'particles', prints=False):

    
    blob_fn = ['blobs_cam1', 'blobs_cam2', 'blobs_cam3', 'blobs_cam4']
    cam_names = ['cam1', 'cam2', 'cam3', 'cam4']
    res = (1280,1024)
    
    
    # setting up the img_system 
    cams = [camera(cn, res) for cn in cam_names]
    for cam in cams:
        try:
            cam.load('')
        except:
            raise ValueError('camera file %s not found'%cam.name)
    imsys = img_system(cams)
    
    
    ROI = [[0.0, 70.0], [0.0, 70.0], [-20.0, 20.0]]
    
    max_blob_distance = 0.0
        
    t0 = time.time()
    mbf = match_blob_files(blob_fn, 
                           imsys, 
                           ROI, 
                           voxel_size, 
                           max_blob_distance,
                           max_err=max_err, 
                           reverse_eta_zeta=True)
    
    # mathing
    mbf.get_particles(frames=None)
    runtime = time.time() - t0
    
    # print matching statistics
    if prints:
        print('particles matched:', len(mbf.particles))
        c4 = sum([1 for p in mbf.particles if len(p[3])==4])
        print('quadruplets:', c4)
        c3 = sum([1 for p in mbf.particles if len(p[3])==3])
        print('triplets:', c3)
        c2 = sum([1 for p in mbf.particles if len(p[3])==2])
        print('pairs:', c2)
    
    
    # save the results
    if save_name is not None:
        mbf.save_results(save_name)
                    
    
    
    # compare to ground_truth
    ground_truth = list(np.loadtxt('ground_truth').round(decimals=2))
    
    success = 0
    for p in mbf.particles:
        p_ = np.array(p[:3])
        for x in ground_truth:
            if np.linalg.norm(p_ - x) < max_err:
                success += 1
               
    wrong = len(mbf.particles) - success
    avg_error = np.mean([p[-2] for p in mbf.particles])
    
    if prints:
        print('success: ',success)
        print('wrong: ',wrong)
        print('error: ', avg_error)
    return (success, wrong, avg_error, runtime)
