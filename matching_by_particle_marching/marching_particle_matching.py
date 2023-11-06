# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Created on Sun Nov  5 22:22:28 2023

@author: ron


A script for matching particles by minimizing the matching error for
individual particles are we march them across the measurement volume.
"""

from time import localtime, strftime
import time

from pandas import read_csv
from numpy import array, linspace, mean, ptp, dot, savetxt
from numpy import sum as npsum
from scipy.spatial import KDTree
from scipy.optimize import minimize

from myptv.imaging_mod import camera, img_system
from myptv.utils import line_dist





class matching_particle_angular_candidates(object):
    
    
    def __init__(self, imsys, blob_files, max_d_err, ROI, N0, min_cam_match=3, 
                 reverse_eta_zeta=False):
        '''
        input - 
        
        imsys - An instance of the myptv img_system with loaded cameras inside
        
        blob_files - a list of paths pointing to blob files. The list shoud
                     be orders according to the order of the camera in imsys.
        
        max_d_err - the max uncertainty in the matching, namely the max 
                    distance between any two epipolar line pairs.
                    
        ROI - Reion of interest in which we are searching for particles in 
              lab space. This is a tuple of length 6 with the following
              format: (xmin, xmax, ymin, ymax, zmin, zmax)
        
        N0 - a tuple of length 3, each specifying the number of points that 
             placed along each axis of the RIO in the initial search. The
             format is (nx, ny, nz).
        
        min_cam_match - the minimum number of cameras allowed when 
                        matching a point, e.g. 2 for pairs, 3 for quadruplets,
                        etc...
        '''
        
        print('\ninitializin matcher:\n')
        
        self.imsys = imsys
        self.blob_files = blob_files
        self.max_d_err = max_d_err
        self.min_cam_match = min_cam_match
        self.ROI = ROI
        self.N0 = N0
        self.matches = []
        self.Ncams = len(self.imsys.cameras)
        
        # k is the k nearest neighbour blobs out of which we search
        self.max_k = 2
        
        # a set that holds identifires for the blobs that have been matched,
        # each represented as (cam number, frame number, blob index)
        self.matchedBlobs = set([])
        
        print('loading blob data...')
        # extract the blob data - each blobfile is a dictionay in a list, where 
        # keys are frame numbers and values are the blob data as arrays. 
        self.frames = set([])
        self.blobs = []
        for fn in blob_files:
            bd = read_csv(fn, sep='\t', header=None)
            
            if reverse_eta_zeta:
                ncols = bd.shape[1]
                ind = list(range(ncols))
                ind[0]=1 ; ind[1]=0
                bd = bd[ind]
            
            self.blobs.append(dict([(k,array(g)) for k,g in bd.groupby(5)]))
            self.frames.update(self.blobs[-1].keys())
            
        self.frames = list(self.frames)
        
        
        # a dicionary that is used to hold kd trees of blob coordinates
        self.B_ik_trees = {'frame': None}
        
        
        # prepare the initial point list:
        self.generate_initial_search_grid()
        
        
        
        
    def generate_initial_search_grid(self):
        '''
        Generates the list of points on which we check search for the nearst
        matches when no prior info is known.
        '''
        nx,ny,nz = self.N0
        xmin,xmax,ymin,ymax,zmin,zmax = self.ROI
        x_lst = [xmin+(i+0.5)*(xmax-xmin)/nx for i in range(nx)]
        y_lst = [ymin+(i+0.5)*(ymax-ymin)/ny for i in range(ny)]
        z_lst = [zmin+(i+0.5)*(zmax-zmin)/nz for i in range(nz)]
        
        self.initPoints = []
        for x_ in x_lst:
            for y_ in y_lst:
                for z_ in z_lst:
                    self.initPoints.append([x_, y_, z_])
        
        
        
    def match_nearest_blobs(self, x, frame):
        '''
        Given a point in lab space, x, and a frame number, this function 
        find in each camera the blobs nearest to this point's projection, 
        it stereo matches them, and returns the results. 
        
        Results are returned only if the stereo match is consedered
        successfull, which means the max_d_err and min_cam_match tests 
        were successfull.
        '''
        
        # (1) if the KDTrees are setup with the wromg frame, we fix this
        if self.B_ik_trees['frame'] != frame:
            self.generate_blob_trees(frame)
            #for camNum in range(self.Ncams):
            #    self.B_ik_trees[camNum] = KDTree(self.blobs[camNum][frame][:,:2])
        
        # (2) prepare a "coords" dictionary with the items being the NN blobs 
        coords = {}
        matchBlobs = {}
        for camNum in range(self.Ncams):
            cam = self.imsys.cameras[camNum]
            projection = cam.projection(x)
            kNN = self.B_ik_trees[camNum].query([projection], k=self.max_k)  
            ind = kNN[1][0]
            for i in range(len(ind)):
                
                identifier = (camNum, frame, ind[i])
                if identifier in self.matchedBlobs: # this blob has been used
                    continue
                
                else:
                    blob = mps.blobs[camNum][frame][ind[i]]
                    coords[camNum] = blob[:2]
                    matchBlobs[camNum] = (blob[:2], ind[i])
                    break
                
        
        
        # (3) perform the stereo matching; If it fails, return None.
        res = self.imsys.stereo_match(coords, self.max_d_err*self.Ncams)
        if res is None: return None
        else: xNew, pairedCams, err = res 
        
        # if the min_cam_match passes, return the results; else, return None
        if len(pairedCams)<self.min_cam_match: return None
        
        # if the error is too big, return None
        elif err>self.max_d_err: return None
        
        else:
            for camNum in list(matchBlobs.keys()):
                if camNum not in pairedCams:
                    del(matchBlobs[camNum])
                else:
                    self.matchedBlobs.add((camNum, frame, matchBlobs[camNum][1]))
        
        return xNew, matchBlobs, err, frame
        
        
        
    
    def generate_blob_trees(self, frame):
        '''
        Will create new KDTrees of the blob coordinates for a given frame
        number. Blobs that have been used up already do not appear in the 
        trees' dataset.
        '''
        
        # used_blob_indexes = dict([(cn, []) for cn in range(self.Ncams)])
        # for b in self.matchedBlobs:
        #    if b[1]==frame:
        #         used_blob_indexes[b[0]].append(b[2])
            
        for camNum in range(self.Ncams):
            # whr = [i for i in range(self.blobs[camNum][frame].shape[0]) 
            #                              if i not in used_blob_indexes[camNum]]
            # self.B_ik_trees[camNum] = KDTree(self.blobs[camNum][frame][whr,:2])
            self.B_ik_trees[camNum] = KDTree(self.blobs[camNum][frame][:,:2])
            
        self.B_ik_trees['frame'] = frame
    
    
    
    
    def stereo_match_frame_with_initial_points(self, frame):
        '''
        This function iterates over the points in the initial point list
        and attempt to match each of then with the nearest neighbout bolb
        projections.
        '''
        print('', 'matching on initial points: frame %d'%frame)
        
        self.generate_blob_trees(frame)
        print(len(self.B_ik_trees[0].data))
        
        dx = (self.ROI[1]-self.ROI[0])/self.N0[0]
        dy = (self.ROI[3]-self.ROI[2])/self.N0[1]
        dz = (self.ROI[5]-self.ROI[4])/self.N0[2]
        
        count = 0
        for x0 in self.initPoints:
            shake = np.random.uniform(-.5, .5, size=3) * [dx,dy,dz]
            res = mps.match_nearest_blobs(x0 + shake, frame)
            if res is not None:
                self.matches.append(res)
                count += 1
        print('', 'matches found: %d\n'%count)
        
        
        
        
    def stereo_match_frame_with_previous_matches(self, frame):
        '''
        This function iterates on the points that were found in frame i-1
        (if any exists), and attemps to stereo match them with the nearest 
        neighbout bolb projections from the given frame (i).
        '''
        print('\n', 'matching using prvious matches: frame %d'%frame)
        
        self.generate_blob_trees(frame)
        print(len(self.B_ik_trees[0].data))
        
        pointToMatchOn = [m[0] for m in self.matches if m[3]==frame-1]
        
        count = 0
        for x0 in pointToMatchOn:
            res = mps.match_nearest_blobs(x0, frame)
            if res is not None:
                self.matches.append(res)
                count += 1
        print('', 'matches found: %d'%count)
        
        
        
        
        
    # ========================================================================
    
    def err_func(self, x, dist = False):
        ''' returns the sum of distancaes from the camera projections '''
        
        if dist == False:
            err = 0
            d_lst = []
            for camNum in range(self.Ncams):
                cam = self.imsys.cameras[camNum]
                projection = cam.projection(x)
                d, i = self.B_ik_trees[camNum].query([projection])
                err += d**2
                d_lst.append(d)
                
        elif dist == True:
            err = 0
            d_lst = []
            for camNum in range(self.Ncams):
                cam = self.imsys.cameras[camNum]
                projection = cam.projection(x)
                d, i = self.B_ik_trees[camNum].query([projection])
                err += d**2
                d_lst.append(d[0])
            print(d_lst)
            
        return (err / self.Ncams)**0.5
        
        
        
    def minimize_particle_position(self, x0, frame):
        '''
        Given a list of initial positions, x0, that are supposedly found
        in a given frame number, this function will march them in space 
        in an attemp to find a 3D location that minimizies the distaces
        between the particle projection on the cameras and existing blobs.
        '''
        
        # if the KDTrees are setup with the wromg frame, we fix this
        if self.B_ik_trees['frame'] != frame:
            for camNum in range(self.Ncams):
                self.B_ik_trees[camNum] = KDTree(self.blobs[camNum][frame][:,:2])
        
        res = minimize(self.err_func, x0 = x0, method='Powell')
        
        return res
    # ========================================================================
        
        








if __name__=='__main__':

    import matplotlib.pyplot as plt
    import numpy as np
    import time
    
    folder = '/home/ron/Desktop/Research/plankton_sweeming/experiments/20221229/run90'
    
    max_d_err = 0.003      # In, e.g. mm, a value similar to calibration err
    min_cam_match = 3
    xmin, xmax, nx = 0, 1, 10
    ymin, ymax, ny = 0, 1, 10
    zmin, zmax, nz = 0, 1, 10
    
    
    max_d_err = 0.3        # In, e.g. mm, a value similar to calibration err
    min_cam_match = 3
    xmin, xmax, nx = -5, 75, 16
    ymin, ymax, ny = -5, 75, 16
    zmin, zmax, nz = -25, 25, 10
    
    
    ROI = (xmin, xmax, ymin, ymax, zmin, zmax)
    N0 = (nx, ny, nz)
    
    camNames = [folder + '/cam%d'%i for i in [1,2,4]]
    
    cam_list = [camera(cn, (800,800)) for cn in camNames]
    cam_list = [camera(cn, (1280,1024)) for cn in camNames]
    
    for cam in cam_list:
        cam.load('.')
    
    imsys = img_system(cam_list)
    
    
    blob_files = [folder + '/blobs_cam%d'%i for i in [1,2,4]]
    
    
    
    
    mps = matching_particle_angular_candidates(imsys, 
                                               blob_files, 
                                               max_d_err, 
                                               ROI,
                                               N0,
                                               min_cam_match=min_cam_match,
                                               reverse_eta_zeta=True)
    
    
    print('started at: ', strftime("%H:%M:%S", localtime()))

    frames = range(100)

    t0 = time.time()
    for f in frames:
        mps.stereo_match_frame_with_previous_matches(f)
        mps.stereo_match_frame_with_initial_points(f)
    print(time.time()-t0, len(mps.matches))


#%%

fig, ax = plt.subplots()
xlist = [m[0][0] for m in mps.matches]
ylist = [m[0][1] for m in mps.matches]
ax.scatter(xlist, ylist, s=2)
ax.set_aspect('equal')