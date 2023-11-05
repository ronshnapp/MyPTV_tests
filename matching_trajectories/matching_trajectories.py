# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Created on Wed Oct  4 21:57:01 2023

@author: ron

This is a script that performs stereo matching on blobs that were
tracked in 2D, using the 2D tracking information.

"""




class matching_trajectories(object):
    '''
    This is a class that performs stereo matching on blobs that were
    tracked in 2D, using the 2D tracking information.
    '''
    
    def __init__(self, imsys, max_err):
        '''
        imsys - an instance of imaing_system 
        
        max_err (float) - maximum allowed stereo matching error 
        '''
        
        self.imsys = imsys
        self.N_cams = len(self.imsys.cameras)
        self.max_err = max_err
    
    
    def match_trajectory(self, traj_id, cam_index):
        '''
        Given a 2D trajectory, this function tries to stereo match it
        '''
        
        # fetching the trajectory data
        
        trajectory = ...
        
        # a list that holds the ids of the trajectories with which
        # the given trajecotie's blobs have been stereo matched in the 
        # previous frame
        
        previous_match_ids = [-1 for i in range(self.N_cams)]
        previous_match_ids[cam_index] = traj_id
        
        # a list of camera_indexes not including the camera of the 
        # current trajectory
        
        other_cams = [i for i in range(self.N_cams) if i != cam_index ]
        
        
        for blob in trajectory:
            
            # set up a dictionary of candidates. The keys are camera_indexes
            # and the items are a tuple of two lists: 
            #    ( [trajectory candidates], [searched candidates] )
            
            candidates = {}
            for i in other_cams:
                candidates[i] = ([], [])
            
            
            # get the frame number
            
            frame = blob[-1]
            
            # -----------------------------------------------------------
            # 1) try to match with previously connected blobs
            
            # if the blob was matched in frame i-1, we first
            # see if the other blobs have trrajectories that continue
            # on to frame i, and if so we add them to the candidate list. 
            
            count_others = 0
            for i in other_cams:
                other_id = previous_match_ids[i]
                if other_id != -1:
                    other = self.traj_in_frame(other_id, i, frame)
                    if other is not None:
                        candidates[i][0].append(other)
                        count_others += 1
                    
            # if we have coninued blobs in all other cameras, we try to 
            # match them
            
            test = False
            if count_others == len(other_cams):
                coords = {}
                coords[cam_index] = trajectory[1:3]
                
                for k in candidates.keys():
                    coords[k] = candidates[k][1:3]
                
                res = self.imsys.stereo_match(coords, 1e20)
                if res is not None:
                    particle, tmp, dist = res
                    
                    if dist < self.max_err:
                         [add particle somwhere ....]
                         test = True
            
            if test: continue
            # -----------------------------------------------------------    
                    
            
            # 2) find candidates
    
    
    
    def traj_in_frame(self, id_, camera, frame):
        '''
        If the trajectory number id_ from camera number camera has a sample
        in the given frame number, this function returns it; otherwise
        it returns None.
        '''
        
    
    
    
    def find_candidates(self, blob, frame, camera):
        '''
        Given a blob, this function searches for candidates for it in the 
        blobs of the given camera and in the fiven frame. 
        '''
