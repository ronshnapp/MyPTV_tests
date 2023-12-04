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
from numpy import array, savetxt
from scipy.spatial import KDTree

from myptv.imaging_mod import camera, img_system






class matching_with_marching_particles_algorithm(object):
    
    
    def __init__(self, imsys, blob_files, max_d_err, ROI, N0, voxel_size,
                 min_cam_match=3, reverse_eta_zeta=False):
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
        
        N0 - The number of random initial points to try and match before
             epipolar search is done.
        
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
        
        self.voxel_size= voxel_size
        
        # k is the k nearest neighbour blobs out of which we search
        self.max_k = 2
        
        # a dictionary that holds identifires for the blobs that have been 
        # matched, with keys that are frame numbers; identifiers are
        # each a tuple with (cam number, frame number, blob index)
        self.matchedBlobs = {0.0: set([])}
        
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
                if identifier in self.matchedBlobs[frame]: # this blob has been used
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
                    self.matchedBlobs[frame].add((camNum, frame, matchBlobs[camNum][1]))
        
        return xNew, matchBlobs, err, frame
        
        
        
    
    def generate_blob_trees(self, frame):
        '''
        Will create new KDTrees of the blob coordinates for a given frame
        number. Blobs that have been used up already do not appear in the 
        trees' dataset.
        '''
        
        # used_blob_indexes = dict([(cn, []) for cn in range(self.Ncams)])
        # for b in self.matchedBlobs[frame]:
        #    if b[1]==frame:
        #         used_blob_indexes[b[0]].append(b[2])
            
        for camNum in range(self.Ncams):
            # whr = [i for i in range(self.blobs[camNum][frame].shape[0]) 
            #                              if i not in used_blob_indexes[camNum]]
            # self.B_ik_trees[camNum] = KDTree(self.blobs[camNum][frame][whr,:2])
            self.B_ik_trees[camNum] = KDTree(self.blobs[camNum][frame][:,:2])
            
        self.B_ik_trees['frame'] = frame
    
    
    
    
    def stereo_match_frame_with_random_initial_points(self, frame, message=False):
        '''
        This function iterates over the points in the initial point list
        and attempt to match each of then with the nearest neighbout bolb
        projections.
        '''
        
        if frame not in self.matchedBlobs.keys():
            self.matchedBlobs[frame] = set([])
            
        self.generate_blob_trees(frame)
        
        count = 0
        for i in range(self.N0):
            x0 = [np.random.uniform(self.ROI[0], self.ROI[1]),
                  np.random.uniform(self.ROI[2], self.ROI[3]),
                  np.random.uniform(self.ROI[4], self.ROI[5])]
            res = self.match_nearest_blobs(x0, frame)
            if res is not None:
                self.matches.append(res)
                count += 1
                
        if message:
            print('', 'matches using random guesses: %d'%count)
            
        
        
        
    def stereo_match_frame_with_given_points(self, points, frame, message=False):
        '''
        Given a list of points in lab space, this function iterates over them
        and attepms to match each of them with their nearest neighbout blob
        projections.
        '''
        
        if frame not in self.matchedBlobs.keys():
            self.matchedBlobs[frame] = set([])
            
        self.generate_blob_trees(frame)
        
        count = 0
        for x0 in points:
            res = self.match_nearest_blobs(x0, frame)
            if res is not None:
                self.matches.append(res)
                count += 1
                
        if message:
            print('', 'matches using given points: %d'%count)
        
        
        
        
    def stereo_match_frame_with_previous_matches(self, frame, message=False,
                                                 backwards=False):
        '''
        This function iterates over the points that were found in frame i-1
        (if any exist), and attemps to stereo match their nearest 
        neighbout blob projections in the given frame (i).
        
        if backwards==True, then we try to use particles from i+1 (is exist)
        to find matches at frame i.
        '''
        
        if frame not in self.matchedBlobs.keys():
            self.matchedBlobs[frame] = set([])
            
        self.generate_blob_trees(frame)
        
        if backwards==False:
            pointToMatchOn = [m[0] for m in self.matches if m[3]==frame-1]
            
        elif backwards==True:
            pointToMatchOn = [m[0] for m in self.matches if m[3]==frame+1]
        
        count = 0
        for x0 in pointToMatchOn:
            res = mps.match_nearest_blobs(x0, frame)
            if res is not None:
                self.matches.append(res)
                count += 1
                
        if message:
            print('matches using prvious results: %d'%count)
        
        
        
        
    def find_candidates_with_two_cameras(self, camNum1, camNum2, frame):
        '''
        This function uses epipolar voxel traversal search to stereo match 
        blob pairs in two given cameras. The points that are found are then
        returned.
        '''
        # fetching the cameras and the blobs
        cam1 = self.imsys.cameras[camNum1] ; cam2 = self.imsys.cameras[camNum2]
        O1 = cam1.O ; O2 = cam2.O
        blobs1 = self.blobs[camNum1][frame]
        blobs2 = self.blobs[camNum2][frame]
        
        # getting the center point of the ROI (for epipolar line seach)
        O_ROI = [(self.ROI[1]+self.ROI[0])/2, 
                 (self.ROI[3]+self.ROI[2])/2, 
                 (self.ROI[5]+self.ROI[4])/2]
        
        # getting the size of the ROI diagonal
        a_range = sum([(self.ROI[2*i+1]-self.ROI[2*i])**2 for i in range(3)])**0.5
        da = self.voxel_size/4.0
        
        # the number of voxel in each direction
        nx = int((self.ROI[1]-self.ROI[0])/self.voxel_size)+1
        ny = int((self.ROI[3]-self.ROI[2])/self.voxel_size)+1
        nz = int((self.ROI[5]-self.ROI[4])/self.voxel_size)+1
        
        
        # a dicionary that holds the blob numbers traversed in each voxel
        voxel_dic = {}
        
        # listing the traversed volxels for blobs in camera 2
        for e,b in enumerate(blobs2):
            
            identifier = (camNum2, frame, e)
            if identifier in self.matchedBlobs[frame]: # this blob has been used
                continue
            
            r = cam2.get_r(b[0], b[1])
            a_center = sum([r[i]*(O_ROI[i]-O2[i]) for i in range(3)]) 
            a1, a2 = a_center - a_range/2 , a_center + a_range/2 
            
            # traversing the blob from O2+a1*r to O2+a2*r to list the voxels
            blob_voxels = set([])
            a = a1
            while a<=a2:
                x, y, z = O2[0] + r[0]*a, O2[1] + r[1]*a, O2[2] + r[2]*a
                
                if self.ROI[0] < x < self.ROI[1]:
                    if self.ROI[2] < y < self.ROI[3]:
                        if self.ROI[4] < z < self.ROI[5]:
                            i = int((x-self.ROI[0])/(self.ROI[1]-self.ROI[0])*nx)
                            j = int((y-self.ROI[2])/(self.ROI[3]-self.ROI[2])*ny)
                            k = int((z-self.ROI[4])/(self.ROI[5]-self.ROI[4])*nz)
                            blob_voxels.add((i,j,k))
                a += da
            
            for voxel in blob_voxels:
                try:
                    voxel_dic[voxel].append(e)
                except:
                    voxel_dic[voxel] = [e]
            
        candidate_pairs = set([])
        # traversing the blobs in camera1 to obtain candidates pairs
        for e,b in enumerate(blobs1):
            
            identifier = (camNum1, frame, e)
            if identifier in self.matchedBlobs[frame]: # this blob has been used
                continue
            
            r = cam1.get_r(b[0], b[1])
            a_center = sum([r[i]*(O_ROI[i]-O1[i]) for i in range(3)]) 
            a1, a2 = a_center - a_range/2 , a_center + a_range/2 
            
            # traversing the blob from O2+a1*r to O2+a2*r to list the candidates
            a = a1
            while a<=a2:
                x, y, z = O1[0] + r[0]*a, O1[1] + r[1]*a, O1[2] + r[2]*a
                
                if self.ROI[0] < x < self.ROI[1]:
                    if self.ROI[2] < y < self.ROI[3]:
                        if self.ROI[4] < z < self.ROI[5]:
                            i = int((x-self.ROI[0])/(self.ROI[1]-self.ROI[0])*nx)
                            j = int((y-self.ROI[2])/(self.ROI[3]-self.ROI[2])*ny)
                            k = int((z-self.ROI[4])/(self.ROI[5]-self.ROI[4])*nz)
                            try:
                                candidates = voxel_dic[(i,j,k)]
                                for cnd in candidates:
                                    candidate_pairs.add((e, cnd))
                            except:
                                pass
                a += da
        
        
        candidate_points = []
        for e1, e2 in candidate_pairs:
            coords = {camNum1:blobs1[e1], camNum2:blobs2[e2]}
            stereoMatch = self.imsys.stereo_match(coords, self.max_d_err)
            if stereoMatch is not None:
                candidate_points.append(stereoMatch[0])
        
        return candidate_points
        
        
        
        
        
        
    def match_frame(self, frame, backwards=False):
        '''
        Will stereo match particles in the given frame number.
        '''
        print('\n')
        print('frame: %d'%frame)
        
        if frame not in self.matchedBlobs.keys():
            self.matchedBlobs[frame] = set([])
        
        m0 = len(self.matches)
        self.stereo_match_frame_with_previous_matches(frame, backwards=backwards)
        newPrevFrame = len(self.matches) - m0
        
        # Matching with random points
        self.stereo_match_frame_with_random_initial_points(frame)
        
        # matching with pair candidates
        if self.voxel_size is not None:
            for i in range(self.Ncams-1):
                camNum1=i ; camNum2=i+1
                cands = self.find_candidates_with_two_cameras(camNum1, camNum2, frame)
                self.stereo_match_frame_with_given_points(cands, frame)
        
        newEpipolarCands = len(self.matches) - m0 - newPrevFrame
        
        newTot = newEpipolarCands + newPrevFrame
        prnt = (newTot, newPrevFrame, newEpipolarCands)
        print('Found %d matches: %d from prev. frame + %d new'%prnt)
        
        
        
       
        
    def plot_disparity_map(self, camNum):
        '''
        A disparity map is a graph that hels in assessing the uncertainty
        of the calibration. It is done by taking the results of the stereo
        matching, projecting them back on a cameras' sensor, and plotting the
        difference between the projected point and the blob with which this 
        point was matched. The disparity map is drawn separately for each
        cameras, and hence the camNum parameter.
        '''
        import matplotlib.pyplot as plt
        disparities_x = []
        disparities_y = []
        cam = self.imsys.cameras[camNum]
        for m in self.matches:
            
            try: blob = m[1][camNum][0]
            except: continue  # < cameras did not participate in this match
                
            x = m[0]
            proj = cam.projection(x)
            disparities_x.append(blob[0] - proj[0])
            disparities_y.append(blob[1] - proj[1])
        
        fig, ax = plt.subplots()
        ax.hist2d(disparities_x, disparities_y, bins=20)
        ax.scatter(disparities_x, disparities_y, s=2)
        ax.set_aspect('equal')
        return fig, ax
        
        
        
        
    def save_particles(self, saveName):
        '''
        Will save the matched particles on the disk with the given
        file name.
        '''
        
        toSave = []
        for m in self.matches:
            p = [m[0][0], m[0][1], m[0][2]]
            for cn in range(self.Ncams):
                try: 
                    p.append(m[1][cn][1])
                except:
                    p.append(-1)
            p.append(m[2])
            p.append(m[3])
            toSave.append(p)
        
        fmt = ['%.3f', '%.3f', '%.3f']
        for i in range(self.Ncams):
            fmt.append('%d')
        fmt = fmt + ['%.3f', '%.3f']
        savetxt(saveName, toSave, fmt=fmt, delimiter='\t')
        
        
        
        








if __name__=='__main__':

    import matplotlib.pyplot as plt
    import numpy as np
    import time
    
    folder = '/home/ron/Desktop/Research/PTV cases/8000_20/myptv_analysis'
    
    max_d_err = 0.001      # In, e.g. mm, a value similar to calibration err
    min_cam_match = 3
    xmin, xmax = 0, 1
    ymin, ymax = 0, 1
    zmin, zmax = 0, 1
    N0 = 30
    voxel_size = None
    
    
    #max_d_err = 0.3        # In, e.g. mm, a value similar to calibration err
    #min_cam_match = 3
    #xmin, xmax, nx = -5, 75, 20
    #ymin, ymax, ny = -5, 75, 20
    #zmin, zmax, nz = -25, 25, 14
    #voxel_size = 5
    
    ROI = (xmin, xmax, ymin, ymax, zmin, zmax)
    
    camNames = [folder + '/cam%d'%i for i in [0,1,2,3]]
    
    cam_list = [camera(cn, (800,800)) for cn in camNames]
    #cam_list = [camera(cn, (1280,1024)) for cn in camNames]
    
    for cam in cam_list:
        cam.load('.')
    
    imsys = img_system(cam_list)
    
    
    blob_files = [folder + '/blobs_cam%d'%i for i in [0,1,2,3]]
    
    
    
    
    mps = matching_with_marching_particles_algorithm(imsys, 
                                               blob_files, 
                                               max_d_err, 
                                               ROI,
                                               N0,
                                               voxel_size,
                                               min_cam_match=min_cam_match,
                                               reverse_eta_zeta=True)
    
    
    print('started at: ', strftime("%H:%M:%S", localtime()))

    frames = range(10)

    t0 = time.time()
    for f in frames:
        mps.match_frame(f)
        
    #for f in frames[::-1]:
    #    mps.match_frame(f, backwards=True)
        
    print(time.time()-t0, len(mps.matches))





#%%

z0, z1 = 0.45, 0.55

fig, ax = plt.subplots()
xlist = [m[0][0] for m in mps.matches if z0<m[0][2]<z1]
ylist = [m[0][1] for m in mps.matches if z0<m[0][2]<z1]
ax.scatter(xlist, ylist, s=2)
ax.set_aspect('equal')