#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 08:14:19 2023

@author: ron
"""

from numpy import dot, array, zeros
from numpy.linalg import lstsq, norm
from scipy.optimize import minimize


# ====================================== Helper functions ====================
class line(object):
    
    def __init__(self, O, r):
        '''
        A helper class for calculating line and point distances.
        O - a point through which the line passes
        r - a vector of the line direction
        '''
        self.O = O
        self.r = r
        
    def distance_to_point(self, P):
        '''
        Returns the least distance between the line and a given point P.
        '''
        s1 = sum([self.r[i]*(P[i]-self.O[i]) for i in [0,1,2]])
        s2 = sum([ri**2 for ri in self.r])
        amin = s1 / s2
        
        dmin = sum([(self.O[i]-P[i]+amin*self.r[i])**2 for i in [0,1,2]])**0.5
        
        return dmin, amin, self.O + amin*self.r
        


def get_nearest_line_crossing(line_list):
    '''
    Given a list of line objects, this function find a point that minimizes
    the sum of the distances to it.
    '''
    func = lambda P: sum([l.distance_to_point(P)[0] for l in line_list])
    P = minimize(func, array([0,0,0])).x
    return P






class Cal_image_coord(object):
    '''
    A class used for reading the calibration image files. This is called
    by the camera class if given a filename with calibration points. 
    '''
    
    def __init__(self, fname):
        '''
        input - 
        fname - String, the path to your calibration point file. The file is
                holds tab separated values with the meaning of: 
                    [x_image, y_image, x_lab, y_lab, z_lab]
        '''
        self.image_coords = []
        self.lab_coords = []
        self.fname = fname
        self.read_file()
        
        
    def read_file(self):
        
        with open(self.fname) as f:
            lines = f.readlines()
            self.N_points = len(lines)
            
            for ln in lines:
                ln_ = ln.split()
                self.image_coords.append([float(ln_[0]), float(ln_[1])])
                self.lab_coords.append( [float(ln_[2]), 
                                         float(ln_[3]),
                                         float(ln_[4])] )
                f.close()


# ====================================== Helper functions ====================







class camera_extended_zolof(object):
    '''
    an object that holds the calibration information for
    each camera. It can be used to:
    1) obtain image coordinates from given lab coordinates. 
    2) vice versa if used together with other cameras at 
       other locations (img_system.stereo_match).
      
    input:
    name - string name for the camera
    resolution - tuple (2) two integers for the camera pixels
    cal_points_fname - path to a file with calibration coordinates for the cam
    '''
    
    def __init__(self, name, resolution, cal_points_fname = None):    
        self.O = zeros(3) + 1.     # camera location
        self.name = name
        self.A = [[0.0 for i in range(17)] for j in [0,1]]
        self.B = [[0.0 for i in range(6)] for j in [0, 1, 2]]
        
        if cal_points_fname is not None:
            cic = Cal_image_coord(cal_points_fname)
            self.image_points = cic.image_coords
            self.lab_points = cic.lab_coords
    


    def __repr__(self):
        
        ret = (self.name +
               '\n O: ' + str(self.O))
        return ret


    
    def get_XCol(self, X):
        '''
        Given a point in 3D, this method returns its P17 polynomial multiplyers
        '''
        X1,X2,X3 = X[0],X[1],X[2]
        XColumn = [1.0, X1, X2, X3,
                   X1**2, X2**2, X3**2, X1*X2, X2*X3, X3*X1, X1*X2*X3]#,
                   #X1*X2**2, X1*X3**2, X2*X1**2, X2*X3**2, X3*X1**2, X3*X2**2]
                   #X1**3, X2**3, X3**3]
        return XColumn
    


    def get_xCol(self, x):
        '''
        Given a point in 2D, this method returns its P10 polynomial multiplyers
        '''
        x1, x2 = x[0], x[1]
        xColumn = [1.0, x1, x2, x1**2, x2**2, x1*x2,
                   x1**2*x2, x2**2*x1, x1**3, x2**3]
        return xColumn
    
    
    
    def projection(self, X):
        '''
        Given a point in 3D, X, this method returns its 2D image projection
        '''
        XColumn = self.get_XCol(X)
        res = dot(XColumn, self.A)
        return [res[0], res[1]]
    
    
    
    def get_r(self, x):
        '''
        Given a point in 3D, X, this method returns its 2D image projection
        '''
        xColumn = self.get_xCol(x)
        res = dot(xColumn, self.B)
        return [res[0], res[1], res[2]]
    
    
    
    def get_r_ori():
        return None
    
    
    
    def save(self):
        return None
    

    def load(self):
        return None    











class extended_zolof_calibration(camera_extended_zolof):
    
    
    def __init__(self, camera, x_list, X_list):
        '''
        Given a list of 2D points, x=(x,y), and a list of 3D point X=(X,Y,Z), 
        we assume that given a point X, we can compute x by a polynomial 
        of degree 3, as - 
        
        x = A0 + A1*X + A2*Y + A3*Z +
            A4*X^2 + A5*Y^2 + A6*Z^2 + A7*XY + A8*YZ + A9*ZX + A108XYZ
            A11*XY^2 + A12*XZ^2 + A13*YX^2 + A14*YZ^2 + A15*ZX^2 + A16*ZY^2  
        '''
        self.cam = camera
        self.A = [[0.0 for i in range(17)] for j in [0,1]]
        self.B = [[0.0 for i in range(6)] for j in [0, 1, 2]]
        self.x_list = x_list
        self.X_list = X_list
        
        
        
    
    def calibrate(self):
        '''
        Given a list of points, x and X, this function attempts to determine
        the A coefficients. 
        '''
        # 1) finding the A coefficients - 
        XColumns = [self.get_XCol(Xi) for Xi in self.X_list]
        res = lstsq(XColumns, self.x_list, rcond=None)
        self.A = res[0]
        
        # 2) finding the best camera center -
        line_list = []
        for i in range(0, len(X)):
            O, e = cp.get_ray_from_x(x[i], X0=X[i])
            line_list.append(line(O, e)) 
        self.O = get_nearest_line_crossing(line_list)
        
        # 3) finding the unit vector for each X -
        r_list = []
        for Xi in self.X_list:
            r = (Xi - self.O)/norm(Xi - self.O)
            r_list.append(r)
        
        # 4) finding the B coefficients -
        xColumns = [self.get_xCol(xi) for xi in self.x_list]
        res = lstsq(xColumns, r_list, rcond=None)
        self.B = res[0]
        
        self.cam.O = self.O
        self.cam.A = self.A
        self.cam.B = self.B
        
        
            
    def get_ray_from_x(self, x, X0=None):
        '''
        Given a point in 2D image space, this function returns a line in 3D
        that passes through this line. The line is represented with six 
        parameters: one point in 3D, O, and one unit vector in 3D, e.
        '''
        
        func = lambda X: sum((array(self.projection(X)) - array(x))**2)
        
        if X0 is None:
            X0 = array([0,0,0])
        
        X02 = array(X0) + array([1,1,1])
            
        O = minimize(func, X0).x
        dX = minimize(func, X02).x
        e = (O-dX)/sum((O-dX)**2)**0.5
        
        return O, e
        
    
    
    
    
import numpy as np
fname = '/home/ron/Desktop/Research/myptv/example/Calibration/cam2_cal_points'
data = np.loadtxt(fname)
x = data[:,:2]
X = data[:,2:]

cam = camera_extended_zolof('cam1', (1280,1024))
cp = extended_zolof_calibration(cam, x, X)
cp.calibrate()

    
    
    
#%%
import matplotlib.pyplot as plt

errs = []
for i in range(len(X)):
    l = line(cam.O, array(cam.get_r(x[i])))
    err, a, p = l.distance_to_point(X[i])
    errs.append(err)

plt.hist(errs, bins='auto')

#%%

from numpy import linspace as lnsp
X_ = [[i,j,k] for i in lnsp(0,69,num=30) for j in lnsp(0,65,num=30) for k in lnsp(0,-20,num=10)]

errs = []
for i in range(len(X_)):
    x_ = cp.projection(X_[i])
    l = line(cp.O, array(cp.get_r(x_)))
    err, a, p = l.distance_to_point(X_[i])
    errs.append(err)

plt.hist(errs, bins='auto')






#%%

'''
import matplotlib.pyplot as plt

fig, ax = plt.subplots()

x_ret = [cp.get_x(X[i]) for i in range(len(X))]

for i in range(len(x_ret)):
    ax.plot([x_ret[i][0], x[i][0]], [x_ret[i][1], x[i][1]], 'bo-', ms=3, lw=0.3)


x_line = array([cp.get_x([i, 0, 0]) for i in np.linspace(0,67, num=200)])
ax.plot(x_line[:,0], x_line[:,1])

x_line = array([cp.get_x([i, 70, 0]) for i in np.linspace(0,67, num=200)])
ax.plot(x_line[:,0], x_line[:,1])        

x_line = array([cp.get_x([i, 0, -20]) for i in np.linspace(0,67, num=200)])
ax.plot(x_line[:,0], x_line[:,1])
        
x_line = array([cp.get_x([i, 70, -20]) for i in np.linspace(0,67, num=200)])
ax.plot(x_line[:,0], x_line[:,1])        
        
x_line = array([cp.get_x([0, 0, i]) for i in np.linspace(10,-30, num=200)])
ax.plot(x_line[:,0], x_line[:,1])

x_line = array([cp.get_x([67, 0, i]) for i in np.linspace(10,-30, num=200)])
ax.plot(x_line[:,0], x_line[:,1])
        
x_line = array([cp.get_x([0, 67, i]) for i in np.linspace(10,-30, num=200)])
ax.plot(x_line[:,0], x_line[:,1])

x_line = array([cp.get_x([67, 67, i]) for i in np.linspace(10,-30, num=200)])
ax.plot(x_line[:,0], x_line[:,1])
'''