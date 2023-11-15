#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 15:32:51 2023

@author: stonneau
"""

import pinocchio as pin 
import numpy as np
import time
from numpy.linalg import pinv,inv,norm,svd,eig
from scipy.optimize import fmin_bfgs,fmin_slsqp
from tools import collision, getcubeplacement, setcubeplacement, projecttojointlimits
from config import LEFT_HOOK, RIGHT_HOOK, LEFT_HAND, RIGHT_HAND, EPSILON
from config import CUBE_PLACEMENT, CUBE_PLACEMENT_TARGET

from tools import setcubeplacement

def cost(q):
    oMcube  = getcubeplacement(cube) #origin of the cube
    oMcubeL = getcubeplacement(cube, LEFT_HOOK) #placement of the left hand hook
    oMcubeR = getcubeplacement(cube, RIGHT_HOOK) #placement of the right hand hook
    
    pin.framesForwardKinematics(robot.model,robot.data,q)   
    pin.computeJointJacobians(robot.model,robot.data,q)
    
    frameidL = robot.model.getFrameId(LEFT_HAND)
    oMframeL = robot.data.oMf[frameidL]
    frameidR = robot.model.getFrameId(RIGHT_HAND)
    oMframeR = robot.data.oMf[frameidR]
    
    L_hand_tran = oMframeL.translation - oMcubeL.translation
    R_hand_tran = oMframeR.translation - oMcubeR.translation
    L_hand_rot = oMframeL.rotation - oMcubeL.rotation
    R_hand_rot = oMframeR.rotation - oMcubeR.rotation
    
    #hand_tran = oMframeL.translation
    #hook_tran = oMcubeL.translation
    #hand_rot = oMframeL.rotation
    #hook_rot = oMcubeL.rotation
    L_hand_difference = np.linalg.norm(L_hand_tran)**2 + np.linalg.norm(L_hand_rot)**2
    R_hand_difference = np.linalg.norm(R_hand_tran)**2 + np.linalg.norm(R_hand_rot)**2
    
    return R_hand_difference + L_hand_difference

def callback(q):
    updatevisuals(viz, robot, cube, q)
    time.sleep(.1)


def computeqgrasppose(robot, qcurrent, cube, cubetarget, viz=None):
    '''Return a collision free configuration grasping a cube at a specific location and a success flag'''
    setcubeplacement(robot, cube, cubetarget)

    q = robot.q0.copy() 
    qopt_bfgs = fmin_bfgs(cost,q, callback=callback)
    print('\n *** Optimal configuration from BFGS = %s \n\n\n\n' % qopt_bfgs)

    return qopt_bfgs, False
            
if __name__ == "__main__":
    from tools import setupwithmeshcat
    from setup_meshcat import updatevisuals
    robot, cube, viz = setupwithmeshcat()
    
    q = robot.q0.copy()
    
    #q0,successinit = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT, viz)
    qe,successend = computeqgrasppose(robot, q, cube, CUBE_PLACEMENT_TARGET,  viz)
    
    updatevisuals(viz, robot, cube, qe)
    
    
    
