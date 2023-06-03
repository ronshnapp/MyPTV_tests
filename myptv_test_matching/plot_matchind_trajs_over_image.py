# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 09:31:42 2022

@author: ron


A script used to generate images of the trajectories that were found 
overlaid by the raw PTV images
"""

from myptv.imaging_mod import camera
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.animation import FuncAnimation
import pandas as pd
import os



def plot_one_frame(particlesPath, cam, frameNum, imgDir, imgIndex, vmax=50, 
                   radius=7, fig=None):
    '''
    Given a path to a file of MyPTV trajectories, a camera object, 
    a frame number, and a path to a raw image, this plots the PTV result 
    over the image.
    
    imgDir - directory of backgroun image
    imgIndex - index of the background image in the sorted image names of imgDir
    
    '''
    import matplotlib
    
    trajs = pd.read_csv(fname, sep='\t', header=None)
    frameColumn = trajs.shape[1]-1
    frameTrajs = np.array(trajs[trajs[frameColumn]==frameNum])
    frameTrajIDs = set(frameTrajs[:,0])
    try: frameTrajIDs.remove(-1)
    except: pass

    if frameColumn==8:
        cmap = matplotlib.cm.get_cmap('Set1')
    else:
        cmap = matplotlib.cm.get_cmap('Spectral')
    
    imgPath = os.path.join(imgDir, sorted(os.listdir(imgDir))[imgIndex])
    image = plt.imread(imgPath)
    
    if fig is None:
        fig, ax = plt.subplots()
        
    else:
        ax = fig.axes[0] 
        ax.clear()
        
    ax.imshow(image, vmax=vmax, cmap='gray')
    
    # plot circles at the projections positions
    for i in range(len(frameTrajs)):
        x,y,z = frameTrajs[i,1:4]
        eta,zeta = cam.projection([x,y,z])
        blobIndexes = frameTrajs[i][4:-2]
        
        if frameColumn==8:
            numOfCameras = sum((blobIndexes>0))
            color = cmap((numOfCameras-2)/9) 
        
        else:
            V = sum((frameTrajs[i,4:7])**2)**0.5
            color = cmap(V/5) 
        
        
        patch = Circle((eta,zeta), radius=radius, color=color,
                       fill=False, lw=1.5)
        ax.add_patch(patch)
    
    # plot lines for trajectories that pass through this frame
    for trid in frameTrajIDs:
        tr = np.array(trajs[trajs[0]==trid])
        xy = np.array([cam.projection(tr[i,1:4]) for i in range(len(tr))])
        ax.plot(xy[:,0], xy[:,1], '-', color='y', alpha=0.5, lw=1)
    
    ax.text(0.02, -0.12, 'frame: %d'%frameNum, transform=ax.transAxes)
    ax.set_xlim(0, cam.resolution[0])
    ax.set_ylim(0, cam.resolution[1])
    ax.invert_yaxis()
    return






def animateFrames(frameList, particlesPath, cam,  imagesDir, imgIndexList,
                  FPS=2, vmax=30, radius=7, figsize=(9,7)):
    
    fig, ax = plt.subplots()
    fig.set_size_inches(figsize)
    
    
    def func(frame, particlesPath, cam, imgDir, imgIndexList):
        print(frame)
        plot_one_frame(particlesPath, cam, frame, imgDir, 
                       imgIndexList[frame], vmax=vmax, radius=radius, fig=fig)
        
    animation = FuncAnimation(fig, func, frames=frameList, interval=1000/FPS,
                              repeat = True,
                              fargs=(particlesPath, cam, 
                                     imgDir,imgIndexList))
    animation.save('animation.mp4', dpi=200)
    plt.close()
    




if __name__=='__main__':
    
    fname = '/home/ron/Desktop/Research/plankton_sweeming/experiments/20220916/MyPTV_analysis/run5/trajecotries_stitched'
    frameNum = 10
    
    imgDir = '/media/ron/Elements/copepods_work/20220916/run5/Cam2_NoBG'
    imgPath = os.path.join(imgDir, sorted(os.listdir(imgDir))[frameNum])
    
    cameraName = 'cam2'
    camResolution = 1280,1024
    cameraPath = '/home/ron/Desktop/Research/plankton_sweeming/experiments/20220916/MyPTV_analysis'
    cam = camera(cameraName, camResolution)
    cam.load(cameraPath)
    imgIndex = frameNum
    
    #plot_one_frame(fname, cam, frameNum, imgDir, imgIndex, vmax=20)
    
    frameList = list(range(0,200,1))
    
    animateFrames(frameList, fname, cam, imgDir, frameList, vmax=30, radius=7, FPS=4)




























