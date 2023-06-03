# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Created on Tue Nov  8 08:06:21 2022

@author: ron


A new algorithm for matching particles in PTV - we sweep particles
across the measurement volume on top of rays from blobs, and look for
intersections with rays coming from other cameras along the ray.


"""

from pandas import read_csv
from numpy import array, linspace, mean
from numpy import sum as npsum
from scipy.spatial import KDTree
from random import sample




class matching_particle_sweeping(object):
    
    def __init__(self, imsys, blob_files, ROI, pixel_position_err):
        '''
        input - 
        
        imsys - An instance of the myptv img_system with loaded cameras inside
        
        blob_files - a list of paths pointing to blob files. The list shoud
                     be orders according to the order of the camera in imsys.
                     
        ROI - A nested list of 3X2 elements. The first holds the minimum and 
              maximum values of x coordinates, the second is same for y, and 
              the third for z coordinates. 
              
        pixel_position_err - the maximum error that is allowed for a detected
                             blob and a projected 3D point to be considered as
                             a physical pair.
        '''
        
        print('\ninitializin matcher:\n')
        
        self.imsys = imsys
        self.blob_files = blob_files
        self.ROI = ROI
        self.pixel_position_err = pixel_position_err
        self.max_iter = 10
        
        self.Ncams = len(self.imsys.cameras)
        
        
        print('loading blob data...')
        # extract the blob data - each blobfile is a dictionay in a list, where 
        # keys are frame numbers and values are the blob data as arrays. 
        self.blobs = []
        for fn in blob_files:
            bd = read_csv(fn, sep='\t', header=None)
            self.blobs.append(dict([(k,array(g)) for k,g in bd.groupby(5)]))
        
        
        # set up a list of the frames in the blob files
        time_lst = []
        for bl in self.blobs:
            time_lst += list(bl.keys())
        self.time_lst = sorted(list(set(time_lst)))
        
        
        # prepare a list for KDTrees; the trees are aranged in dictionaries
        # in the same structure as self.blobs
        self.trees = []
        for blobDic in self.blobs:
            self.trees.append({})
        
        print('estimating essential statistics...')
        # estimate the distance between particles in 3D
        blob_counts = [(i,k,len(self.blobs[i][k])) 
                       for i in range(len(self.blobs)) 
                       for k in self.blobs[i].keys()]
        longets_blob_dic_info = max(blob_counts, key = lambda x: x[-1]) 
        self.Np = longets_blob_dic_info[-1]
        d = [ROI[i][1]-ROI[i][0] for i in range(3)]
        volume = d[0] * d[1] * d[2]
        self.particle_distance = (volume / self.Np)**0.333
        
        
        # estimate the distance in pixels between blobs
        i,k = longets_blob_dic_info[:2]
        longets_blob_list = self.blobs[i][k]
        longest_tree = KDTree(longets_blob_list[:,1::-1])
        self.trees[i][k] = longest_tree
        nbd_list = []
        for eta, zeta in longets_blob_list[:,1::-1]:
            nbd_list.append(longest_tree.query([eta,zeta], k=2)[0][1])
        self.mean_blob_distance = mean(nbd_list)
        
        
        
        
        # estimate spacings between local minimas in particle sweeping
        spacing_list = []
        for cn in range(self.Ncams):
            fnum = list(self.blobs[cn].keys())[0]
            nsample = min([len(self.blobs[cn][fnum]), 5])
            blobnums = sample( list(range(len(self.blobs[cn][fnum]))), nsample)
            steps = self.Np*3
            
            for bn in blobnums:
                nnd, ind, a_range = self.sweepBlob(cn, bn, fnum, steps=steps)
                sumNNd = npsum(nnd, axis=0)
                where = [i for i,y in enumerate(sumNNd) 
                         if ((i==0) or (sumNNd[i-1]>y))
                         and ((i==len(sumNNd)-1) or (y<sumNNd[i+1]))]
                for i in range(len(where)-1):
                    da = a_range[where[i+1]] - a_range[where[i]]
                    spacing_list.append(abs(da))
        self.avg_spacing = sum(spacing_list) / len(spacing_list)
        
        
        print('\ndone\n')
        return None
    
    
    
    
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
    
    
    
    
    def get_particle_NND(self, x, frame_num, camSkip=-1):
        '''
        Given a point, x, this function projects it onto camra space and 
        calculates the distance to the nearest blob. This is done for each
        camera in imsys and the distances are returned in a list.
        
        input - 
        
        x - list or array, representing a point in 3D space.
        
        frame_num - The frame number in which the distances to blobs should be
                    taken.
        '''
        
        NNd, ind = [], []
        
        for i in range(len(self.imsys.cameras)):
            
            if i==camSkip: 
                NNd.append(0)
                ind.append(-1)
                continue
            
            cam = self.imsys.cameras[i]
            eta, zeta = cam.projection(x)
            
            try:
                tree = self.trees[i][frame_num] 
            
            except:
                #tree = KDTree(self.blobs[i][frame_num][:,:2])
                tree = KDTree(self.blobs[i][frame_num][:,1::-1])
                
                self.trees[i][frame_num] = tree
            
            nearestBlob = tree.query([eta, zeta])
            NNd.append(nearestBlob[0])
            ind.append(nearestBlob[1])
            
        return NNd, ind
    
    
    
    
    def sweepBlob(self, camNum, blobNum, frameNum, alims=None, steps=None):
        '''
        We take a blob (identified using the index of camera to which it 
        belongs and its own index in self.blobs[camNumber]), sweep it across 
        a measurement region, and return the NNd lists and index lists along
        the sweeping.
        
        Each blob gives rise to a parameteric ray, O + r*a, where a is the 
        free parameter. The range of a on which we sweep the particles can be
        given as a tuple in alims. If it is not given, the particle is swept
        across the ROI. 
        
        The sampling interval for the sweeping can be given in steps, or it is 
        set automatically by considering the estimated distance between 
        particles and the length of the sweeping line. 
        '''
        
        # get the direction vector and camera center
        cam = self.imsys.cameras[camNum]
        #eta, zeta = self.blobs[camNum][frameNum][blobNum][:2]
        zeta, eta = self.blobs[camNum][frameNum][blobNum][:2]
        r = cam.get_r(eta, zeta)
        O = cam.O

        # prepare lists to hold the resutls
        NNd_lst = [[] for i in range(len(self.imsys.cameras))]
        ind_lst = [[] for i in range(len(self.imsys.cameras))]
        
        
        # lines are swept parametrically. we search for the parameter value
        # for which the ray is inside the ROI
        if alims is None:
            alims = self.get_a_range(O, r)
            
            if alims is None:  # <- ray does not pass in the ROI
                return NNd_lst, ind_lst, []
            
            else:
                dx = sum((r * alims[0] - r * alims[1])**2)**0.5
                if steps is None: steps = max([3, int(dx/self.avg_spacing*1)])
                                          #int(dx / self.particle_distance * 3.5)])
                a_range = linspace(alims[0], alims[1], num=steps)
            
        else:
            dx = sum((r * alims[0] - r * alims[1])**2)**0.5
            if steps is None: steps = max([3, int(dx/self.avg_spacing*1)])
                                          #int(dx / self.particle_distance * 3.5)])
            a_range = linspace(alims[0], alims[1], num=steps)
            
        
        # sweepin the parametric ray by iterating over possible a values    
        for a in a_range:
            x = O + r * a
            NNd, ind = self.get_particle_NND(x, frameNum, camSkip=camNum)
            for j in range(len(NNd)):
                NNd_lst[j].append(NNd[j])
                ind_lst[j].append(ind[j])
    
        return NNd_lst, ind_lst, a_range
        
        
        
        
    def stereo_match_blob(self, camNum, blobNum, frameNum):
        '''
        Given a blob, this function attempts to stereo match it according to 
        the particle sweeping algoritm. A blob is identified by the index of
        its camera, the frame number and the index of the blob in the 
        given frame. 
        '''
        
        # (1) sweep the blob across the measurement volume, and list the 
        # distances to its nearest blobs in the different cameras
        nnd, ind, a_range = mps.sweepBlob(camNum, blobNum, frameNum)
        
        # (2) if there's a point along the swept line in which the nnds are
        # lower than the segmentation error then the rays correspoinding
        # to the blobs at this point are considered to belong to the same 
        # physical particle, and we return them.
        
        for i in range(len(nnd[0])):
            if all([nnd[j][i]<self.pixel_position_err for j in range(self.Ncams)]):
                matched_rays = []
                for k in range(len(ind)):
                    if ind[k][i]==-1:
                        matched_rays.append((camNum, blobNum))
                    else:
                        matched_rays.append( (k, ind[k][i]) )
                
                return matched_rays
        
        
        # (3) sum the nearest blob distances across the cameras
        sumNNd = npsum(nnd, axis=0)
        
        # (4) we search for low local minima along the swept particle line, 
        # in which the sumNNd is lower than the mean blob nearest 
        # neighbor distance times the number of cameras
        where = [(i,y) for i,y in enumerate(sumNNd) 
                 if ((i==0) or (sumNNd[i-1]>y))
                 and ((i==len(sumNNd)-1) or (y<sumNNd[i+1]))
                 and (y < self.mean_blob_distance * self.Ncams)]
        
        where = [xx[0] for xx in sorted(where, key=lambda x: x[1])]

    
        # (5) we now try to find the a values that minimize the sum of nearest
        # blobd distances around the local minima we found for sumNNd
        
        # get the direction vector and camera center
        #cam = self.imsys.cameras[camNum]
        O, r = self.get_O_r(camNum, blobNum, frameNum)
        
        # minimize the sum of nearest blobs distance from the initial points
        for i in where:
            a0 = a_range[i]  # <- initial guess
            a_list = [a0]
            f_list = [sumNNd[i]]
            da = (a_range[-1]-a_range[0])/len(a_range)/1000 # <- for gradients
            count = 0        # <- iteration counter
            convergence_check = False

            # iterate with gradient descent
            while count <= self.max_iter and convergence_check==False:
                
                # fkm1 = sum(self.get_particle_NND(O+(a_list[-1]-da)*r, 
                #                                  frameNum, camSkip=camNum)[0])
                # fkp1 = sum(self.get_particle_NND(O+(a_list[-1]+da)*r, 
                #                                  frameNum, camSkip=camNum)[0])
                # dfda = (fkp1 - fkm1) / 2 / da
                # d2fda = (fkp1 + fkm1 - 2*f_list[-1]) / da**2
                
                fkp1 = sum(self.get_particle_NND(O+(a_list[-1]+da)*r, 
                                                  frameNum, camSkip=camNum)[0])
                dfda = (fkp1 - f_list[-1]) / da
                
                learnRate = f_list[-1] / (dfda**2) / 2
                a_step = learnRate*dfda
                a_next = a_list[-1] - a_step
                NNd_next, ind_next = self.get_particle_NND(O+(a_next)*r, 
                                                           frameNum, 
                                                           camSkip=camNum)
                f_next = sum(NNd_next)
                a_list.append(a_next)
                f_list.append(f_next)
                
                #print(i, a_next, f_next)
                
                if self.check_NNd(NNd_next):
                    ind_next[camNum] = blobNum
                    ind_next = [(i,ind_next[i]) for i in range(len(ind_next))]
                    return ind_next
                
                if f_list[-1]>=f_list[-2]: learnRate = learnRate/2
                
                count += 1
                if abs(f_list[-1]-f_list[-2]) < self.pixel_position_err/5:
                    convergence_check = True
        
        # (6) if no good locations were found, we return None
        return None
        
    
    
    
    def check_NNd(self, NNd_list):
        '''
        Given a list of nearest blob distances to a point, this function 
        returns True is they should be considered belonging to the same 
        physical particle and False otherwise
        '''
        test = [NNd_list[i]<self.pixel_position_err for i in range(self.Ncams)]
        return all(test)
        
    
    
    def get_O_r(self, camNum, blobNum, frameNum):
        '''
        Returns the camera oriin and the r vector of a given blob
        '''
        # get the direction vector and camera center
        cam = self.imsys.cameras[camNum]
        #eta, zeta = self.blobs[camNum][frameNum][blobNum][:2]
        zeta, eta = self.blobs[camNum][frameNum][blobNum][:2]
        r = cam.get_r(eta, zeta)
        O = cam.O
        return O, r
    
    
        
    def minimize_sumNNd(self, sumNNd, camNum, blobNum, frameNum):
        '''
        This function attempts to minimize the sum of nearest blob distances
        list belonging to a particular blob (given by camNum, blobNum, 
        frameNum). 
        
        sumNNd is a list of descrete samples of the sum of nearest blob 
        distances.
        '''
        
        # getting local minimas
        where = [(i,y) for i,y in enumerate(sumNNd) 
                 if ((i==0) or (sumNNd[i-1]>y))
                 and ((i==len(sumNNd)-1) or (y<sumNNd[i+1]))
                 and (y < self.mean_blob_distance * self.Ncams)]
        
        # sorting the local minimas based on their hight
        where = [xx[0] for xx in sorted(where, key=lambda x: x[1])]
        
        # get blobs ray parameters
        O, r = self.get_O_r(camNum, blobNum, frameNum)
        
        
        
        




from myptv.imaging_mod import camera, img_system

cam1 = camera('cam1', (1280,1024))
cam2 = camera('cam2', (1280,1024))
cam3 = camera('cam3', (1280,1024))
cam4 = camera('cam4', (1280,1024))

cam_list = [cam1, cam2, cam3, cam4]

for cam in cam_list:
    cam.load('.')



imsys = img_system(cam_list)
blob_files = ['blobs_cam1', 'blobs_cam2', 'blobs_cam3', 'blobs_cam4']
ROI = ((-10,80),(-10,80),(-40,40))
pixel_position_err = 0.5

mps = matching_particle_sweeping(imsys, blob_files, ROI, pixel_position_err)




import matplotlib.pyplot as plt
import numpy as np

import time



runtimes = []
successes = 0
matches_list = []
unsuccessful = []

for i in range(mps.Np):
    t0 = time.time()
    #nnd, ind, a_range = mps.sweepBlob(0,i,0)
    matches = mps.stereo_match_blob(0,i,0)
    runtimes.append(time.time() - t0)
    if matches is not None: 
        matches_list.append(matches)
        if len(set([x[1] for x in matches]))==1:
            successes+=1
    else:
        unsuccessful.append(i)
        

print(sum(runtimes))
print(successes)


#fig, ax = plt.subplots()
#ax.plot(nnd[0])
#ax.plot(nnd[1])
#ax.plot(nnd[2])
#ax.plot(nnd[3])

#sumNNd = np.sum(nnd, axis=0)
#ax.set_ylim(0,np.mean(sumNNd))
#ax.plot(sumNNd, '-o', lw=0.5, ms=3)
#ax.plot([0, len(sumNNd)], 
#        [mps.mean_blob_distance*2, mps.mean_blob_distance*2], 'k-', lw=0.5)




def plot_NND_series(canNum, blobNum):
    nnd, ind, a_range = mps.sweepBlob(canNum,blobNum,0)
    sumNNd = np.sum(nnd, axis=0)
    plt.plot(np.linspace(0,1,num=len(sumNNd)), sumNNd, 'o', ms=4)
    
    nnd, ind, a_range = mps.sweepBlob(canNum,blobNum,0, steps=1500)
    sumNNd = np.sum(nnd, axis=0)
    plt.plot(np.linspace(0,1,num=len(sumNNd)), sumNNd, '-', ms=4)

