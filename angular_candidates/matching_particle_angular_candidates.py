# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
Created on Tue Nov  8 08:06:21 2022

@author: ron


An algorithm for matching particles in PTV by finding candidates
for mathcing through the angle of the projection of epipolar
lines.


"""

from pandas import read_csv
from numpy import array, linspace, mean, ptp, dot
from numpy import sum as npsum
from scipy.spatial import KDTree
from random import sample
from myptv.imaging_mod import camera, img_system
from myptv.utils import line_dist
import warnings





class matching_particle_angular_candidates(object):
    
    
    def __init__(self, imsys, blob_files, d_theta, max_d_err, max_err_3d):
        '''
        input - 
        
        imsys - An instance of the myptv img_system with loaded cameras inside
        
        blob_files - a list of paths pointing to blob files. The list shoud
                     be orders according to the order of the camera in imsys.
        
        d_theta - the max uncertainy of the projections angles
        
        max_d_err - the max uncertainty of the distances along epipolar lines
        '''
        
        print('\ninitializin matcher:\n')
        
        self.imsys = imsys
        self.blob_files = blob_files
        self.d_theta = d_theta
        self.max_d_err = max_d_err
        self.max_err_3d = max_err_3d
        
        self.Ncams = len(self.imsys.cameras)
        
        
        print('loading blob data...')
        # extract the blob data - each blobfile is a dictionay in a list, where 
        # keys are frame numbers and values are the blob data as arrays. 
        self.frames = set([])
        self.blobs = []
        for fn in blob_files:
            bd = read_csv(fn, sep='\t', header=None)
            self.blobs.append(dict([(k,array(g)) for k,g in bd.groupby(5)]))
            self.frames.update(self.blobs[-1].keys())
        self.frames = list(self.frames)
        
        
        # a list that holds kd trees of theta_ijk values
        self.theta_ik_trees = []
        for i in range(self.Ncams):
            dic = {}
            for k in range(self.Ncams):
                if i!=k:
                    dic[k] = 0
            self.theta_ik_trees.append(dic)
        
        
        
    def get_theta_ijk(self, cam_num, p_num, cam_k, frame):
        '''
        calculates the angle in camera cam_num, of blob with index
        p_num, relative to the center, O_k, of camera cam_k.
        
        cam_num - integer
        p_num - integer
        cam_k - integer
        frame - integer, index of the frame number at which the mathching done
        '''
        
        if cam_num == cam_k:
            raise ValueError('cam_num and cam_k cannot be the same')
        
        Xij = self.blobs[cam_num][self.frames[frame]][p_num][:2][::-1]
        #Xij = self.blobs[cam_num][self.frames[frame]][p_num][:2]
        O_k = self.imsys.cameras[cam_k].O
        O_ik = self.imsys.cameras[cam_num].projection(O_k)
        
        #theta_ijk = (Xij[1] - O_ik[1]) / (Xij[0] - O_ik[0])
        #return theta_ijk
        
        O_i = self.imsys.cameras[cam_num].O
        rij = self.imsys.cameras[cam_num].get_r(Xij[0], Xij[1])
        r_OkOi = (O_k - O_i)/sum((O_k - O_i)**2)**0.5
        rij_k = r_OkOi - rij
        p = self.imsys.cameras[cam_num].projection(O_k + rij_k*600)
        
        theta_ijk = (p[1] - O_ik[1]) / (p[0] - O_ik[0])
        return theta_ijk
    
    
    
    def get_THETA_ijk(self, cam_num, p_num, cam_k, frame):
        '''
        calculates the angle in camera cam_k, of the epipolar line of the
        particle with index p_num of camera cam_num.
        
        cam_num - integer
        p_num - integer
        cam_k - integer
        frame - integer, index of the frame number at which the mathching done
        '''
        
        if cam_num == cam_k:
            raise ValueError('cam_num and cam_k cannot be the same')
        
        Xij = self.blobs[cam_num][self.frames[frame]][p_num][:2][::-1]
        #Xij = self.blobs[cam_num][self.frames[frame]][p_num][:2]
        O_i = self.imsys.cameras[cam_num].O
        rij = self.imsys.cameras[cam_num].get_r(Xij[0], Xij[1])
        
        p1 = O_i
        p2 = O_i - rij*600
        
        proj1 = self.imsys.cameras[cam_k].projection(p1)
        proj2 = self.imsys.cameras[cam_k].projection(p2)
        
        #print(proj1)
        #print(proj2)
        
        THETA_ijk = (proj1[1] - proj2[1]) / (proj1[0] - proj2[0])
        return THETA_ijk
    
    
    def calculate_theta_dictionaries(self, frame):
        '''
        Goes over all the blobs and for each one calculates the angles
        it has with respect to the other cameras k.
        
        Eventually, to retrieve theta_ijk, we look at self.theta_dic[i][j][k]
        Note that for i==k the value of theta_ijk is not defined (NaN).
        '''
        
        self.theta_dic = {}
        for i in range(len(self.imsys.cameras)):
            self.theta_dic[i] = {}
            for j in range(len(self.blobs[i][frame])):
                self.theta_dic[i][j] = []
                for k in range(len(self.imsys.cameras)):
                    if i==k:
                        self.theta_dic[i][j].append(float('nan'))
                    else:
                        self.theta_dic[i][j].append(
                                            self.get_theta_ijk(i,j,k,frame))

            
        
    
    def get_blob_angular_candidates(self, i, j, frame):
        '''
        For the blob number j belonging to camera i at the given frame index,
        this function returns a dictionary that holds the candidate matches
        for it. 
        
        The candidates are indicated by their camera number i_ and blob
        number j_. Each of them is also associated with the distance along the 
        epipolar line of blob ij from Oi that corresponds to blob Xi_j_. This 
        distance is annotated dij_ij.
        
        Thus, we return a dictionary that has keys i_ and values
        are lists of tuples that hold (j_, dij_ij) of candidates in camera i_.
        '''
        
        Oi = self.imsys.cameras[i].O
        Xij = self.blobs[i][frame][j][:2][::-1]
        #Xij = self.blobs[i][frame][j][:2]
        rij = self.imsys.cameras[i].get_r(Xij[0], Xij[1])
        
        candidate_dic = {}
        for k in range(len(self.imsys.cameras)):
            if k != i:
                Ok = self.imsys.cameras[k].O
                THETA_ijk = self.get_THETA_ijk(i, j, k, frame)
                
                #print(k, THETA_ijk)
                
                if self.theta_ik_trees[i][k]==0:
                    theta_ik_list = [[self.theta_dic[k][j_][i]] for j_ in
                                      range(len(self.blobs[k][frame]))]
                    self.theta_ik_trees[i][k] = KDTree(theta_ik_list)
                
                theta_range = ptp(self.theta_ik_trees[i][k].data)
                
                # searching for candidates with similar theta using the tree
                # which returns their indexes j_
                cand_j = self.theta_ik_trees[i][k].query_ball_point([THETA_ijk],
                                                                  self.d_theta * theta_range)
                
                # calculate the distances d along the epipolar line that
                # corresponds to each candidate
                for e, j_ in enumerate(cand_j):
                    Xkj = self.blobs[k][frame][j_][:2][::-1]
                    #Xkj = self.blobs[k][frame][j_][:2]
                    rki = self.imsys.cameras[k].get_r(Xkj[0], Xkj[1])
                    
                    ld, x = line_dist(Oi, rij, Ok, rki)
                    dij_ij = sum((x - Oi)**2)**0.5
                    d_theta = abs(self.theta_dic[k][j_][0] - THETA_ijk) / theta_range
                    cand_j[e] = (j_, dij_ij, d_theta)
                    
                candidate_dic[k] = cand_j
                
        return candidate_dic
    
    
    
    def find_blob_candidates(self, i, j, frame):
        '''
        Here we take a blob identified by camera number i and blob number j,
        retrieve its angular candidates, and deterimnes among those sets of 
        candidates that their distance along the epipolar line is also 
        compatible. We then return these sets of candidates.
        '''
        c = self.get_blob_angular_candidates(i,j,frame)
        keys = list(c.keys())
        N = len(keys)
        
        index_list = [0 for k in range(len(c.keys()))]
        lengths = [len(c[k]) for k in keys]
        empty_cams = [k for k in keys if len(c[k])==0]
        if len(empty_cams)>0:
            # warnings.warn('Note: no candidates found in cam '+str(empty_cams))
            return []
        
        for k in c.keys():
            c[k] = sorted(c[k], key=lambda x: x[1])
        
        
        cands = []
        while any([index_list[e] < lengths[e] for e in range(N)]):
            current_d_lst = [c[keys[e]][index_list[e]][1] for e in range(N)] 
            mean_d = sum(current_d_lst) / N
            test = all([abs(d - mean_d)<self.max_d_err for d in current_d_lst])
            if test:
                cands.append([])
                for e in range(N):
                    cands[-1].append((keys[e], c[keys[e]][index_list[e]][0]))
            
            can_be_progressed = [(current_d_lst[e],e) for e in range(N) 
                                 if index_list[e]+1<lengths[e]]
            
            if len(can_be_progressed)>0:
                argmin = min(can_be_progressed, key=lambda x: x[0])[1]
                index_list[argmin] += 1
            else:
                break
        
        return cands
        
    
    
    def sreteo_match_blob(self, i, j, frame):
        '''
        Here we take a blob identified by camera number i and blob number j,
        retrieve its candidates and then try to stereo match all of them. 
        We return the matches that their err is smaller than max_d_err. For 
        testing purposes we also return the candidates.
        '''
        candidates = mps.find_blob_candidates(i, j, frame)
        
        coordDic = {i: self.blobs[i][frame][j][:2][::-1]}
        matches = []
        
        for cand in candidates:
            
            for i_, j_ in cand:
                coordDic[i_] = self.blobs[i_][frame][j_][:2][::-1]
            
            match = self.imsys.stereo_match(coordDic, 100000)
            
            if match[-1] < self.max_err_3d:
                err = match[-1]
                indexes = [-1 for k in range(len(self.imsys.cameras))]
                indexes[i] = j
                for i_, j_ in cand: indexes[i_] = j_
                m = list(match[0]) + indexes + [err, frame]
                matches.append(m)
                
        return matches, candidates
    
    
    
    def clear_theta_trees(self):
        '''
        Prepares a fresh list ready to be filled with theta_ik KDtrees.
        '''
        self.theta_ik_trees = []
        for i in range(self.Ncams):
            dic = {}
            for k in range(self.Ncams):
                if i!=k:
                    dic[k] = 0
            self.theta_ik_trees.append(dic)
        


if __name__=='__main__':

    import matplotlib.pyplot as plt
    import numpy as np
    import time
    
    camNames = ['./testcase_1/cam%d'%i for i in [1,2,3]]
    cam_list = [camera(cn, (1280,1024)) for cn in camNames]
    
    
    for cam in cam_list:
        cam.load('.')
    
    imsys = img_system(cam_list)
    blob_files = ['./testcase_1/blobs_cam%d'%i for i in [1,2,3]]
    
    d_theta = 0.05
    max_d_err = 0.5
    max_err_3d = 0.5
    
    mps = matching_particle_angular_candidates(imsys, 
                                               blob_files, 
                                               d_theta, 
                                               max_d_err, max_err_3d)
    
    frames = list(mps.blobs[0].keys())
    
    # We first calculate the LUT for the various angles, theta_ijk. This
    # needs to be repeated for each frame.
    t0 = time.time()
    mps.calculate_theta_dictionaries(0)
    print(time.time() - t0)
    
    
    matches = []
    match_indexes = []
    N_candidates = 0
    
    t0 = time.time()
    err_3d_max = 0.5
    
    for i in [0,1,2]:
        Np = len(mps.blobs[i][0])
        
        for j in range(Np):
            matches_j, candidates_j = mps.sreteo_match_blob(i,j,0)
            
            N_candidates += len(candidates_j)
            
            if len(matches_j)>0:
                for m in matches_j:
                    inds = m[3:-2]
                    #print(inds, matches_j)
                    
                    if inds not in match_indexes:
                        matches.append(m)
                        match_indexes.append(inds)
                    
                    
    print('Total stereo matches: ', len(matches))
    print('Total angular candidates: ', N_candidates)
    print('time: ', time.time() - t0)

    print(matches[0])

        
        
        
        
# ==========================================================



def get_CoordDic(i0, i1, i2, mps, frame=0):
    '''Given blob indexes, this returns a blob dict for stereo matching'''
    
    indexes = [i0,i1,i2]
    blobs = [mps.blobs[i_][frame][indexes[i_]] for i_ in range(len(indexes))]
    bd = dict([(k, blobs[k][:2][::-1]) for k in range(len(blobs))])
    return bd



def get_angles(i0, i1, i2, mps, frame=0, k=0):
    '''For particles with indexes i0, i1, i2 in cameras 0, 1, 2, this returns
    the epipolar line angles relative to the cemter of camera k'''
    
    ret = {}
    I = [i0,i1,i2][k]
    
    for i in [0,1,2]:
        if i==k: continue
    
        Theta = mps.get_THETA_ijk(k, I, i, frame)
        theta = mps.get_theta_ijk(i, [i0,i1,i2][i], k, frame)
        ret[i] = (Theta, theta)
        
    return ret


# matches_2 = []
# mps.d_theta = 100

# for i0 in [3]:
    
#     cd = mps.get_blob_angular_candidates(0, i0, 0)
    
#     for i1 in cd[1]:
#         for i2 in cd[2]:
            
#             bd = get_CoordDic(i0, i1[0], i2[0], mps)
            
#             match = imsys.stereo_match(bd, 10000)
            
#             if match[-1]< 1.0:
#                 matches_2.append([match[0], [i0, i1[0], i2[0]], match[-1]])
        
        
        
        