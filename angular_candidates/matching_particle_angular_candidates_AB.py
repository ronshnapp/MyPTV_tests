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
    
    
    def __init__(self, imsys, blob_files, dB, max_d_err, max_err_3d,
                 reverse_eta_zeta=False):
        '''
        input - 
        
        imsys - An instance of the myptv img_system with loaded cameras inside
        
        blob_files - a list of paths pointing to blob files. The list shoud
                     be orders according to the order of the camera in imsys.
        
        dB - the max uncertainy of the projections angles
        
        max_d_err - the max uncertainty of the distances along epipolar lines
        '''
        
        print('\ninitializin matcher:\n')
        
        self.imsys = imsys
        self.blob_files = blob_files
        self.dB = dB
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
            
            if reverse_eta_zeta:
                ncols = bd.shape[1]
                ind = list(range(ncols))
                ind[0]=1 ; ind[1]=0
                bd = bd[ind]
            
            self.blobs.append(dict([(k,array(g)) for k,g in bd.groupby(5)]))
            self.frames.update(self.blobs[-1].keys())
        self.frames = list(self.frames)
        
        
        # a list that holds kd trees of theta_ijk values
        self.B_ik_trees = []
        for i in range(self.Ncams):
            dic = {}
            for k in range(self.Ncams):
                if i!=k:
                    dic[k] = 0
            self.B_ik_trees.append(dic)
        
        
        
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
        
    ### =======================================================================
        
    def get_epipolar_projection(self, i, j, k, frame):
        '''
        Takes the epipolar line of particle j in camera i, and projects it
        on camera k. The projection is then modeled as a straight line
        with coordinates: zeta = A*eta + B.
        This function calculates A and B and returns them.
        '''
        
        # get the epipolar line parameters in 3D
        Oi = self.imsys.cameras[i].O
        Xij = self.blobs[i][frame][j][:2] #[::-1]
        rij = self.imsys.cameras[i].get_r(Xij[0], Xij[1])
        
        # estimate an a value for which the epipolar line is roughly at
        # the center of the image of camera k
        cam_k = self.imsys.cameras[k]
        
        a0 = sum(Oi**2)**0.5
        da = a0/20000
        p1 = cam_k.projection(Oi - rij * (a0 + da))
        p2 = cam_k.projection(Oi - rij * (a0 - da))
        A = (p1[1]-p2[1]) / (p1[0]-p2[0])
        B = p1[1] - A*p1[0]
        return A, B
        
    
    
    def get_virtual_origins_dic(self):
        '''
        When we project epipolar lines from one camera onto another we obtain
        (approximately) straight lines that cross at some points. This point
        is called the virtual origin of one camera in another. 
        This function generates a dictionary that holds the virtual origins
        of each camera in each other camera.
        '''
        self.virtualOrigins = {}
        
        for i in range(mps.Ncams):
            
            for j in range(mps.Ncams):
                
                if i==j: continue # <- no VO for a camera in itself
                
                self.virtualOrigins[(i,j)] = self.calculate_virtual_origin(i,j)
                
                
                
    def calculate_virtual_origin(self, i, j):
        '''
        Calculates and returns the virtual origin of camera i in camera j.
        '''
        cami = self.imsys.cameras[i]
        camj = self.imsys.cameras[j]
        
        Oi = cami.O
        a0 = sum(Oi**2)**0.5
        da = a0/20000
        
        eta_list = [0, cami.resolution[0]/2, cami.resolution[0]]
        zeta_list = [0, cami.resolution[1]/2, cami.resolution[1]]
        
        # get the epipolar line for the center of camera i
        rij = self.imsys.cameras[i].get_r(eta_list[1], zeta_list[1])
        p1 = camj.projection(Oi - rij * (a0 + da))
        p2 = camj.projection(Oi - rij * (a0 - da))
        A0 = (p1[1]-p2[1]) / (p1[0]-p2[0])
        B0 = p1[1] - A0*p1[0]
        
        VO_sum = [0, 0]
        
        for e,ee in enumerate(eta_list):
            for z,zz in enumerate(zeta_list):
                
                if ee==1 and zz==2: continue
            
                rij = self.imsys.cameras[i].get_r(e, z)
                p1 = camj.projection(Oi - rij * (a0 + da))
                p2 = camj.projection(Oi - rij * (a0 - da))
                Ai = (p1[1]-p2[1]) / (p1[0]-p2[0])
                Bi = p1[1] - Ai*p1[0]
                
                VOx = (Bi-B0)/(A0-Ai)
                VOy = Ai*VOx + Bi
                
                VO_sum[0] = VO_sum[0] + VOx
                VO_sum[1] = VO_sum[1] + VOy
        
        return array(VO_sum)/9
        
        
        
    def get_blob_relative_epipolar_projection(self, i, j, k, frame):
        '''
        Take the blob number j in camera i. There is a line in camera i image
        space that originates from the virtual origin of camera k, and crosses
        the blob j. This function returns A and B for this blob.
        '''
        VO = self.virtualOrigins[(k, i)]
        Xij = self.blobs[i][frame][j][:2]
        A = (VO[1]-Xij[1]) / (VO[0]-Xij[0])
        B = VO[1] - A*VO[0]
        return A, B
        
    
    
    def calculate_B_dictionaries(self, frame):
        '''
        Goes over all the blobs (i, j) and for each one calculates the epipolar
        line B parameter it has with respect to all other cameras k. The
        
        Eventually, to retrieve B_ijk, we look at self.B_dic[i][j][k]
        Note that for i==k the value of B_ijk is not defined (NaN).
        '''
        
        self.B_dic = {}
        for i in range(self.Ncams):
            self.B_dic[i] = {}
            for j in range(len(self.blobs[i][frame])):
                self.B_dic[i][j] = []
                for k in range(self.Ncams):
                    if i==k:
                        self.B_dic[i][j].append(float('nan'))
                    else:
                        A, B = self.get_blob_relative_epipolar_projection(i,
                                                                          j,
                                                                          k,
                                                                          frame)
                        self.B_dic[i][j].append(B)


    
    def get_blob_epipolar_candidates(self, i, j, frame):
        '''
        For the blob number j belonging to camera i at the given frame index,
        this returns a dictionary that holds candidate matches for it. 
        
        The candidates are indicated by their camera number i_ and blob
        number j_. ### Each of them is also associated with the distance along the 
        epipolar line of blob ij from Oi that corresponds to blob Xi_j_. This 
        distance is annotated dij_ij. ###
        
        Thus, we return a dictionary that has keys i_ and values
        are lists of tuples that hold (j_, dij_ij) of candidates in camera i_.
        '''
        
        candidate_dic = {}
        for k in range(len(self.imsys.cameras)):  # <- For all other k cameras
            if k != i:
                
                Aijk, Bijk = self.get_epipolar_projection(i, j, k, frame)
                
                print(k, Bijk)
                
                if self.B_ik_trees[i][k]==0:
                    B_ik_list = [[self.B_dic[k][j_][i]] for j_ in
                                      range(len(self.blobs[k][frame]))]
                    self.B_ik_trees[i][k] = KDTree(B_ik_list)
                
                B_range = ptp(self.B_ik_trees[i][k].data)
                
                # searching for candidates with similar B using the tree
                # which returns their indexes j_
                cand_j = self.B_ik_trees[i][k].query_ball_point([Bijk],
                                                                self.dB * B_range)
                
                # calculate the distances d along the epipolar line that
                # corresponds to each candidate
                for e, j_ in enumerate(cand_j):
                    
                    cand_j[e] = (j_, 0, self.B_dic[k][j_][i])
                    
                candidate_dic[k] = cand_j
                
        return candidate_dic
    
    
    
    


if __name__=='__main__':

    import matplotlib.pyplot as plt
    import numpy as np
    import time
    
    # camNames = ['./testcase_1/cam%d'%i for i in [1,2,3]]
    camNames = ['./testcase_2/cam%d'%i for i in [1,2,3]]
    cam_list = [camera(cn, (1280,1024)) for cn in camNames]
    
    
    for cam in cam_list:
        cam.load('.')
    
    imsys = img_system(cam_list)
    blob_files = ['./testcase_2/cam%d_CalBlobs'%i for i in [1,2,3]]
    
    d_theta = 0.01
    max_d_err = 0.5
    max_err_3d = 0.5
    
    mps = matching_particle_angular_candidates(imsys, 
                                               blob_files, 
                                               d_theta, 
                                               max_d_err, 
                                               max_err_3d,
                                               reverse_eta_zeta=True)
    
    mps.get_virtual_origins_dic()
    
    frames = list(mps.blobs[0].keys())
    f = frames[0]
    
    mps.calculate_B_dictionaries(f)
        
        
         
#%%
import matplotlib.pyplot as plt

fig, ax = plt.subplots()


i = 251


A, B = mps.get_epipolar_projection(0, i, 2, 0)
ax.plot([0, 1200], [A*0+B, A*1200+B], '--')

Oi = mps.imsys.cameras[0].O
Xij = mps.blobs[0][0][i]
rij = mps.imsys.cameras[0].get_r(Xij[0], Xij[1])

a = np.linspace(300, 3000, num=50)
x_ = [mps.imsys.cameras[2].projection(Oi - a_*rij)[0] for a_ in a]
y_ = [mps.imsys.cameras[2].projection(Oi - a_*rij)[1] for a_ in a]
ax.plot(x_, y_, 'k-')


for i_ in range(len(mps.blobs[2][0])):
    Xi_j_ = mps.blobs[2][0][i_]
    ax.plot(Xi_j_[0], Xi_j_[1], 'o', ms=3)
    ax.text(Xi_j_[0], Xi_j_[1], str(i_))


#for j in [30,61,17]:
#    A, B = mps.get_blob_relative_epipolar_projection(2, j, 0, 0)
#    ax.plot([0, 1200], [A*0+B, A*1200+B], 'r-', alpha=0.2)        

#%%

success = []
for j in range(len(mps.blobs[2][0])):
    
    cd = {0: mps.blobs[0][0][251][:2],
          2: mps.blobs[2][0][j][:2]}
    
    m = imsys.stereo_match(cd, 1000000)
    
    if m[-1]<=2:
        success.append(j)
        print(j, m)



