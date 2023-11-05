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
from numpy import array, linspace, mean, ptp, dot, savetxt
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
        self.matches = []
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
            
            
        # a dictionary of interpolators for B. The keys are camera numbers
        # and the values are interpolators for these two cameras
        print('preparing epipolar interpolators')
        self.B_interpolator_dict = {}
        for i in range(self.Ncams):
            for k in range(self.Ncams):
                if i==k: continue
                self.B_interpolator_dict[(i,k)] = self.build_B_interpolator(i,k)
            
            
        
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
        da = a0/10
        
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
        
        # =========================================================
        # Virtual oriin interpolation with linear interpolator
        # =========================================================
        
        # from scipy.interpolate import LinearNDInterpolator
        
        # points, values = [], []
        # for e,ee in enumerate(eta_list):
        #     for z,zz in enumerate(zeta_list):
                
        #         #if ee==1 and zz==2: continue
        
        #         rij = self.imsys.cameras[i].get_r(e, z)
        #         p1 = camj.projection(Oi - rij * (a0 + da))
        #         p2 = camj.projection(Oi - rij * (a0 - da))
        #         Ai = (p1[1]-p2[1]) / (p1[0]-p2[0])
        #         Bi = p1[1] - Ai*p1[0]
                
        #         VOx = (Bi-B0)/(A0-Ai)
        #         VOy = Ai*VOx + Bi
                
        #         points.append([ee,zz])
        #         values.append([VOx, VOy])
        
        # return LinearNDInterpolator(points, values)
        # =========================================================
        
        
        
    
    def build_B_interpolator(self, i, k):
        '''
        For cameras number i and k, this funciton draws projections of epopilar 
        lines from camera i onto camera k, and then generates a linear  
        interpolator that takes in image space coordinates in camera k and
        returns the value of B relative to camera i.
        '''
        from scipy.interpolate import LinearNDInterpolator
        from numpy import linspace
        
        cami = self.imsys.cameras[i]
        camk = self.imsys.cameras[k]
        
        Oi = cami.O
        a0 = sum(Oi**2)**0.5
        da = a0/25
        
        rxi, ryi = cami.resolution[0], cami.resolution[1]
        eta_list = linspace(0, 1, num=8)*rxi
        zeta_list = linspace(0, 1, num=8)*ryi
        
        rxk, ryk = camk.resolution[0], camk.resolution[1]
        
        points = []
        values = []
        
        for e in eta_list:
            for z in zeta_list:
                r = cami.get_r(e, z)
                a_ = 0
                while a_ < a0*3:
                    x = Oi - r*a_
                    Xij = camk.projection(x)
                    if 0<=Xij[0]<=rxk and 0<=Xij[1]<=ryk:
                        points.append(Xij)
                        x1 = camk.projection(Oi - r*(a_-da/100))
                        x2 = camk.projection(Oi - r*(a_+da/100))
                        A = (x1[1]-x2[1])/(x1[0]-x2[0])
                        B = Xij[1]-A*Xij[0]
                        values.append(B)
                    
                    a_ += da
                    
        return LinearNDInterpolator(points, values)
            
        
        
        
    def get_blob_relative_epipolar_projection(self, i, j, k, frame):
        '''
        Take the blob number j in camera i. There is a line in camera i image
        space that originates from the virtual origin of camera k, and crosses
        the blob j. This function returns A and B for this blob.
        '''
        Xij = self.blobs[i][frame][j][:2]
        
        VO = self.virtualOrigins[(k, i)]         # <- Fixed mean VO
        # VO = self.virtualOrigins[(k, i)](Xij)[0]   # <- Interpolated VO
        
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
                        #A, B = self.get_blob_relative_epipolar_projection(i,
                        #                                                  j,
                        #                                                  k,
                        #                                                  frame)
                        interpolator = self.B_interpolator_dict[(k, i)]
                        Xij = self.blobs[i][frame][j][:2]
                        B = interpolator(Xij)[0]
                        self.B_dic[i][j].append(B)


    
    def get_epipolar_candidates(self, i, j, frame):
        '''
        For the blob number j belonging to camera i at the given frame index,
        this returns a dictionary that holds candidate matches for it. 
        
        The candidates are indicated by their camera number i_ and blob
        number j_. Each of them is also associated with the distance along the 
        epipolar line of blob ij from Oi that corresponds to blob Xi_j_. This 
        distance is annotated dij_ij.
        
        Thus, we return a dictionary that has keys i_ and values
        are lists of tuples that hold (j_, dij_ij) of candidates in camera i_.
        '''
        
        candidate_dic = {}
        Oi = self.imsys.cameras[i].O
        Xij = self.blobs[i][frame][j][:2]
        rij = self.imsys.cameras[i].get_r(Xij[0], Xij[1]) 
        
        for k in range(len(self.imsys.cameras)):  # <- For all other k cameras
            if k != i:
                
                Aijk, Bijk = self.get_epipolar_projection(i, j, k, frame)
                
                if self.B_ik_trees[i][k]==0:
                    
                    B_ik_list = [[self.B_dic[k][j_][i]] for j_ in
                                      range(len(self.blobs[k][frame]))]
                    self.B_ik_trees[i][k] = KDTree(B_ik_list)
                
                B_range = ptp(self.B_ik_trees[i][k].data)
                
                # searching for candidates with similar B using the tree
                # which returns their indexes j_
                #print((k, Bijk))
                cand_j = self.B_ik_trees[i][k].query_ball_point([Bijk],
                                                                self.dB * B_range)
                
                # calculate the distances d along the epipolar line that
                # corresponds to each candidate
                for e, j_ in enumerate(cand_j):
                    Ok = self.imsys.cameras[k].O
                    Xkj = self.blobs[k][frame][j_][:2]
                    rki = self.imsys.cameras[k].get_r(Xkj[0], Xkj[1])
                    
                    ld, x = line_dist(Oi, rij, Ok, rki)
                    dij_ij = sum((x - Oi)**2)**0.5
                    
                    cand_j[e] = (j_, dij_ij, self.B_dic[k][j_][i])
                    
                candidate_dic[k] = cand_j
                
        return candidate_dic
    
    
    
    def match_blob_candidates(self, i, j, frame):
        '''
        Matches are returns all the candidates of the blob number j
        from camera i in frame.
        '''
        from itertools import product
        
        c = self.get_epipolar_candidates(i,j,frame)
        keys = list(c.keys())
        N = len(keys)
        
        ind_lst = [list(range(len(c[k]))) for k in c.keys()]
        
        matches = []
        
        ind_combinations = list(product(*ind_lst))
        
        for comb in ind_combinations:
            
            # check if d_ij of all particles is similar
            dij_lst = [c[keys[i]][comb[i]][1] for i in range(len(comb))]
            av_dij = sum(dij_lst)/len(dij_lst)
            dij_deviations = [abs(dij-av_dij)<self.max_d_err for dij in dij_lst]
            if not all(dij_deviations):
                continue
            
            # construct the coord_dic for stereo matching and index list
            cord_lst = [(i, self.blobs[i][frame][j])]
            il = [-1 for n in range(len(self.imsys.cameras))]
            il[i] = j
            
            for e, ind in enumerate(comb):
                i_ = keys[e]
                j_ = c[i_][ind][0]
                #print(i_,j_)
                cord_lst.append((i_, self.blobs[i_][frame][j_]))
                il[i_] = j_
                
            coord_dic = dict(cord_lst)
            
            # stereo metching and adding to the list
            match = self.imsys.stereo_match(coord_dic, 1e100)
            pos = list(match[0])
            err = [match[-1]]
            
            matches.append( pos + il + err + [frame])
            
        return matches
    
    
    
    def clear_B_trees(self):
        '''
        Prepares a fresh list ready to be filled with B_ik KDtrees. This is
        used every time we proceed to match particles in a new frame.
        '''
        self.B_ik_trees = []
        for i in range(self.Ncams):
            dic = {}
            for k in range(self.Ncams):
                if i!=k:
                    dic[k] = 0
            self.B_ik_trees.append(dic)
    
    
    
    def match_particles_in_frame(self, Frame):
        '''
        Given a frame number, this function will stereo match all the blobs
        in that frame.
        '''
        
        print('\nmatching frame ', Frame)
        t0 = time.time()
        
        print('\nCalculating epipolar dictionaries')
        self.clear_B_trees()
        self.calculate_B_dictionaries(Frame)
        
        N_candidates = 0
        N_matches = 0
        matched = set([])
    
        for i in range(len(self.imsys.cameras)):
            for j in range(len(mps.blobs[i][Frame])):
                candidate_matches = mps.match_blob_candidates(i, j, Frame)
                N_candidates += len(candidate_matches)
                mprev = N_matches
                for m in candidate_matches:
                    il = tuple(m[3:-2])
                    if m[-2]< self.max_err_3d and il not in matched:
                        self.matches.append(m)
                        matched.add(il)
                        N_matches += 1
                
                #print('     ', end='\r')
                #print('%d/%d: %d, %d     '%(i, j, len(candidate_matches), 
                #                       N_matches-mprev), end="\r")
               
        
        print('\ntotal matches this frame: ', N_matches)
        print('total candidates this frame: ', N_candidates)
        print('calculation time this frame: ', time.time() - t0)
        
    
    
    def save_results(self, fname):
        '''will save the list of matched particles'''
            
        fmt = ['%.3f', '%.3f', '%.3f']
        for i in range(len(self.imsys.cameras)):
            fmt.append('%d')
        fmt = fmt + ['%.3f', '%.3f']
        savetxt(fname, self.matches, fmt=fmt, delimiter='\t')
    
    


if __name__=='__main__':

    import matplotlib.pyplot as plt
    import numpy as np
    import time
    
    # camNames = ['./testcase_1/cam%d'%i for i in [1,2,3]]
    # camNames = ['./testcase_2/cam%d'%i for i in [1,2,3]]
    # camNames = ['./testcase_3/cam%d'%i for i in [1,2,3,4]]
    # camNames = ['./testcase_4/cam%d'%i for i in [1,2,4]]
    camNames = ['./testcase_4/cam%d'%i for i in [0,1,4]]
    
    cam_list = [camera(cn, (1280,1024)) for cn in camNames]
    
    
    for cam in cam_list:
        cam.load('.')
    
    imsys = img_system(cam_list)
    # blob_files = ['./testcase_1/blobs_cam%d'%i for i in [1,2,3]]
    # blob_files = ['./testcase_2/cam%d_CalBlobs'%i for i in [1,2,3]]
    # blob_files = ['./testcase_3/blobs_cam%d'%i for i in [1,2,3,4]]
    blob_files = ['./testcase_4/blobs_cam%d'%i for i in [1,2,4]]
    
    d_theta = 0.001      # Numerical - the fraction of space to look at
    max_d_err = 1.0      # In, e.g. mm, a value similar to calibration err
    max_err_3d = 0.25     # In, e.g. mm, a value similar to calibration err
    
    
    mps = matching_particle_angular_candidates(imsys, 
                                               blob_files, 
                                               d_theta, 
                                               max_d_err, 
                                               max_err_3d,
                                               reverse_eta_zeta=True)
    
    mps.get_virtual_origins_dic()
    frames = list(mps.blobs[0].keys())
    
    #mps.calculate_B_dictionaries(0)
    
    for f in frames[:10]:
        mps.match_particles_in_frame(f)
   
    print(len(mps.matches))
    
    #mps.save_results('particlesAngularMatching')

      
#%%
# import matplotlib.pyplot as plt

# fig, ax = plt.subplots()


# A, B = mps.get_epipolar_projection(0, 0, 2, 0)
# ax.plot([0, 1200], [A*0+B, A*1200+B], 'b--')
# print(A, B)


# A, B = mps.get_epipolar_projection(1, 18, 2, 0)
# ax.plot([0, 1200], [A*0+B, A*1200+B], 'r--')
# print(A, B)



# for i_ in range(len(mps.blobs[2][0])):
#     Xi_j_ = mps.blobs[2][0][i_]
#     ax.plot(Xi_j_[0], Xi_j_[1], 'o', ms=3)
#     ax.text(Xi_j_[0], Xi_j_[1], str(i_))


# for j in [17]:
#     A, B = mps.get_blob_relative_epipolar_projection(2, j, 0, 0)
#     ax.plot([0, 1200], [A*0+B, A*1200+B], 'b-', alpha=0.2)        


# for j in [18]:
#     A, B = mps.get_blob_relative_epipolar_projection(2, j, 1, 0)
#     ax.plot([0, 1200], [A*0+B, A*1200+B], 'r-', alpha=0.2)   

      










