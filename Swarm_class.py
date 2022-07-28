#!/usr/bin/env python
# coding: utf-8

# # Generate_swarm_data

# In[1]:


import sys
import numpy as np
import itertools as it
import multiprocessing as mp
import h5py 
sentinel = None


# In[2]:


import numpy as np
import itertools as it

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colorbar as colorbar
from matplotlib.patches import Ellipse,Polygon,FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap

import networkx as nx
import os
import shutil

from shapely import geometry
from shapely.geometry.polygon import LinearRing


# In[3]:


def _intersections(a, b):
        ea = LinearRing(a)      
        eb = LinearRing(b)
        mp = ea.intersects(eb)
        return mp
    
    
def _ellipse_polyline(ellipses, n=100):
    '''returns a polygon approximation of an ellipse with n points'''
    t = np.linspace(0, 2*np.pi, n, endpoint=False)
    st = np.sin(t)
    ct = np.cos(t)
    result = []
    for x0, y0, a, b, angle in ellipses:
        sa = np.sin(angle)
        ca = np.cos(angle)
        p = np.empty((n, 2))
        p[:, 0] = x0 + a/2.0 * ca * ct - b * sa * st
        p[:, 1] = y0 + a/2.0 * sa * ct + b * ca * st
        result.append(p)
    return result

def _get_tangent_point_parameter(w,r,theta,phi,main_axis=0.5):

    '''calculates where the tangent points lie on the ellipse, return the corresponding angles,
    these can be translated in to coordinates via using the function
    ellipse_point_from_parameter()
    '''
    w=w/2.0
    aa=np.sqrt(-2.0*main_axis*main_axis*w*w + (main_axis*main_axis + w*w)*r*r + (w*w - main_axis*main_axis)*r*r*np.cos(2.0*(theta - phi)))/np.sqrt(2.0)
    bb= w*r*np.cos(theta - phi) - main_axis*w
    psi1=2.0*np.arctan2(aa-main_axis*r*np.sin(theta - phi),bb)
    psi2= -2.0*np.arctan2(aa+main_axis*r*np.sin(theta - phi),bb)
    return [psi1,psi2]

def _ellipse_point_from_parameter(r,theta,phi,psi,w,l=0.5):
    #calculates cartesian coordinates for a point on an ellipse 
    # with long axis 1, short axis w, ellipse center at r,theta
    # that is given by the ellipse parameter psi
    
    x=r*np.cos(theta) + l*np.cos(phi)*np.cos(psi) + w*l*np.sin(phi)*np.sin(psi)
    y=r*np.sin(theta) + l*np.sin(phi)*np.cos(psi) - w*l*np.cos(phi)*np.sin(psi)
    return [x,y]

def _cast_to_pm_pi(a):
        '''Casts any (radian) angle to the 
            equivalent in the interval (-pi, pi)'''
        #invalid value in remainder error due to nans in the diagonal
        
        b = (a+np.pi)%(2.*np.pi)
        b -= np.pi
        return b
    
    
def remove_self_intersections(inters,n):
    ''' used to remove intersections of ray emitted from ellipse i's eye and intersecting with 
        ellipse i's boundary when detecting all intersections of those rays with all other ellipses,
        inters is array of interception points with indices ijklm
        i: x/y [2], 
        j: which intersection [2], 
        k: on which ellipse [n], 
        l: for which viewer [n], 
        m: for which ray [2(n-1)]'''

    for i in range(n):
        inters[:,:,i,i,:]=np.nan
    return inters


def get_closest_id(r,out,n):
    ''' used to find the closest intersection point on a ray emitted from and ellipses eye,
        r is numpy array with indices jklm as follows:
        j: which intersection [2], 
        k: on which ellipse [n], 
        l: for which viewer [n], 
        m: for which ray [2(n-1)]'''

    for j,k in it.product(range(n),range((n-1)*2)):
            if np.isnan(r[:,:,j,k]).all():
                out[j,k]=np.nan
            else:
                out[j,k]=np.nanargmin(r[:,:,j,k],axis=1)[1]
    return out


def psi_go(w,r,theta,phi,mainaxis=0.5):

    '''calculates where the tangent points lie on the ellipse, return the corresponding angles,
    these can be translated in to coordinates via using the function ellipsepoints_forgo()
    (as found in simon leblanc's go code, IMPORTANT: l is the length of the main axis,
    not the position of the eye'''
    l=mainaxis
    w=w/2.0
    aa=np.sqrt(-2.0*l*l*w*w + (l*l + w*w)*r*r + (w*w - l*l)*r*r*np.cos(2.0*(theta - phi)))/np.sqrt(2.0)
    bb= w*r*np.cos(theta - phi) - l*w
    psi1=2.0*np.arctan2(aa-l*r*np.sin(theta - phi),bb)
    psi2= -2.0*np.arctan2(aa+l*r*np.sin(theta - phi),bb)
    return [psi1,psi2]

def ellipsepoints_forgo(r,theta,phi,psi,w,l=0.5):
    #calculates cartesian coordinates for a point on an ellipse 
    # with long axis 1, short axis w, ellipse center at r,theta
    # that is given by the ellipse parameter psi
    
    x=r*np.cos(theta) + l*np.cos(phi)*np.cos(psi) + w*l*np.sin(phi)*np.sin(psi)
    y=r*np.sin(theta) + l*np.sin(phi)*np.cos(psi) - w*l*np.cos(phi)*np.sin(psi)
    return [x,y]


def get_ellipse_line_intersection_points(eyes,tps,w):
    ''' given two points of the line (eyes and tp) calculates
        the points at which this line intersects with an ellipse
        of length 1 and width w with center at the origin and
        orientation along the positive x-axis, 
        returns points as 2x2 array, 
        index1: x/y, 
        index2: which intersection point,
        if only 1 intersections found both entries are equal,
        if no intersections are found, entries are np.nan'''
    x1=eyes[0]
    y1=eyes[1]
    x2=tps[0]
    y2=tps[1]
    a=0.5
    b=w/2.
    dd=((x2-x1)**2/(a**2)+(y2-y1)**2/(b**2))
    ee=(2.*x1*(x2-x1)/(a**2)+2.*y1*(y2-y1)/(b**2))
    ff=(x1**2/(a**2)+y1**2/(b**2)-1.)
    determinant=ee**2-4.*dd*ff
    float_epsilon=0.00001
    zeromask=abs(determinant)>=1000.*float_epsilon
    determinant*=zeromask
    t=(np.array([(-ee-np.sqrt(determinant))/(2.*dd),
        (-ee+np.sqrt(determinant))/(2.*dd)]))
    mask=np.array(t>0.,dtype=float)
    mask[mask==0.]=np.nan
    x=mask*(x1+(x2-x1)*t)
    y=mask*(y1+(y2-y1)*t)
    return np.array([x,y])

def point_inside(p,s):
    pnt=geometry.Point(p)
    shape=geometry.Polygon(s)
    return pnt.within(shape)

def _numberofrings(nn):
    lower_estimate=nn/np.pi
    upper_estimate=(np.sqrt(4.*np.pi*nn+1)+1)/2.*np.pi
    return int(np.floor(lower_estimate)), int(np.floor(upper_estimate))


# In[4]:


class Swarm:
        def __init__(self,n=40,setup='grid',pos=None,pos_center=None,phi=None,w=0.4,l=0.,dist=2.,
                     noise_pos=0.1,noise_phi=0.9,eliminate_overlaps=False, alpha=None):
                """
                Generate spatial configurations with pos/pos_center/phi (not given for now) and 
                remove the overlaps between the nodes if possible
                (made the values N and n indicating number of nodes consistent)
                """
                self.w=w
                self.l=l
                self.n=n
                if pos is None and pos_center is None and phi is None:
                    #generate initial positions and orientations (phi) according to setup, N, 
                    #noise_phi, dist, noise_pos and w
                    pos,phi=self._generate_initial_spatial_configuration(setup,n,noise_pos,dist,noise_phi,w)
                    self.pos_center=pos
                    self.pos=pos-np.array([-l/2.0*np.cos(phi),-l/2.0*np.sin(phi)])
                    self.phi=phi
#                     print('orientations in _init: ' + str(phi))
                    #why reset calculated variables - erases old data and initializes variables
                    self._reset_calculated_variables()
                else:
                    # set positions and orientations according to input
                    if pos_center is not None and pos is None:
                        self.set_positions_and_orientations(pos_center,phi,center=True)
                    elif pos_center is None and pos is not None:
                        self.set_positions_and_orientations(pos,phi,center=False)
                    else:
                        print('Set either pos or pos_center. If you intend to generate positions,')
                        print('do NOT set either pos or pos_center or phi.')
                if eliminate_overlaps and self.n>1:
                    ok_to_continue=self._eliminate_overlaps()
                else:
                    ok_to_continue=True
                if ok_to_continue==False:
                    print('Overlaps could not be removed successfully. Please try again.')
         
        
        def _generate_initial_spatial_configuration(self,state,nn,noise_int,d,kappa,w):
                if state=='grid':
                        n=int(np.floor(np.sqrt(nn)))
                        xlen=n
                        ylen=n
                        number=n*n
                        grid_x=np.linspace(d,d*xlen,xlen,endpoint=True)
                        grid_y=np.linspace(d,d*ylen,ylen,endpoint=True)
                        x,y=np.meshgrid(grid_x,grid_y)
                        pos=np.array([x.flatten(),y.flatten()])
                        if n<np.sqrt(nn):
                                for i in range(nn-number):
                                        extra=np.array([d*(xlen+1+np.floor(i/n)),d*(i%n+1)]).reshape(2,1)
                                        pos=np.hstack([pos,extra])
                        orientations=np.random.vonmises(0,kappa,nn)
#                         print('orientations in _generate: ' + str(orientations))
                        noise=(np.random.random((2,nn))-np.ones((2,nn))*0.5)*2.0*noise_int*d
                        pos=pos+noise
                        return pos,orientations 

                elif state=='hexagonal':
                        d_y=d/np.sqrt(2.) 
                        n=int(np.floor(np.sqrt(nn)))
                        xlen=n
                        ylen=n
                        number=n*n
                        grid_x=np.linspace(d,d*xlen,xlen,endpoint=True)
                        grid_y=np.linspace(d_y,d_y*ylen,ylen,endpoint=True)
                        x,y=np.meshgrid(grid_x,grid_y)
                        x[0:-1:2]+=d/2.
                        pos=np.array([x.flatten(),y.flatten()])
                        if n<np.sqrt(nn):
                                for i in range(nn-number):
                                        extra=np.array([d*(xlen+1+np.floor(i/n)),d_y*(i%n+1)]).reshape(2,1)
                                        pos=np.hstack([pos,extra])
                        orientations=np.random.vonmises(0.0,kappa,nn)
                        noise_x=(np.random.random((nn))-np.ones((nn))*0.5)*2.0*noise_int*d
                        noise_y=(np.random.random((nn))-np.ones((nn))*0.5)*2.0*noise_int*d_y
                        pos[0]+=noise_x
                        pos[1]+=noise_y
                        return pos,orientations

                elif state=='milling':
                        lower, upper = _numberofrings(nn)
                        radius=(1.0/2.0+np.arange(upper))*d
                        population=np.floor((radius*2.0*np.pi)/d).astype(int)
                        totalnumber=np.cumsum(population)
                        nr_rings=np.amin(np.where(totalnumber>=nn))+1
                        radius=(1./2.+np.arange(nr_rings))*d
                        population=np.floor((radius*2.*np.pi)/d).astype(int)
                        population[-1]=nn-np.sum(population[:-1])
                        distance=2*np.pi*radius/population
                        offset=(nr_rings+1)*d
                        xpos=[]
                        ypos=[]
                        orientations=[]
                        for i in np.arange(nr_rings):
                                theta=2*np.pi*np.linspace(0,1,population[i],endpoint=False)
                                +((np.random.random(population[i])-np.ones(population[i])*0.5)*2.0*noise_int
                                  *d)/radius[i]
                                orientations.append(theta-np.pi/2.0*np.ones(population[i])
                                                    +np.random.vonmises(0.0,kappa,population[i]))
                                xpos.append(radius[i]*np.cos(theta)+offset)
                                ypos.append(radius[i]*np.sin(theta)+offset)
                        xpos=np.concatenate(xpos)
                        ypos=np.concatenate(ypos)
                        orientations=np.concatenate(orientations)
                        orientations=_cast_to_pm_pi(orientations)
                        return np.array([xpos,ypos]),orientations

                else:
                        print("state needs to be either milling or grid or hexagonal")



        def _reset_calculated_variables(self):
                ''' Resets all the variables of swarm that are calculated from the original
                        input of positions, orientations, ellipse width w and eye position l
                '''
                self.n=len(self.phi)
                self.metric_distance_center=np.zeros([self.n,self.n])
                self.tangent_pt_subj_pol=np.zeros([2,self.n,self.n])
                self.tangent_pt_obj_cart=np.zeros([2,self.n,self.n])
                self.angular_area=np.zeros([self.n,self.n])
                self.network=nx.DiGraph()
                self.visual_angles=np.zeros([self.n,self.n])
                self.eyeinside=()       
                self.visual_field=np.zeros([self.n,self.n,(self.n-1)*2])
                self.tp_subj_pol=np.zeros([2,self.n,self.n])
                self.tp_subj_pol2=np.zeros([2,self.n,self.n])
        
        def set_positions_and_orientations(self,pos,phi,center=False):
                '''
                sets the positions of ellipse centers (center=True) or eyes (center=False) 
                as well as orientations,
                resets any measures previously derived from these quantities
                
                INPUT:
                pos: numpy array of dimension 2xN or Nx2
                phi: numpy array or list of length N
                center: boolean
                '''
                self.l=l
                #first check if the shape of the positions or position centers is acceptable 
                if (np.shape(pos)[0]!=2 or np.shape(pos)[1]!=2):
                    print('positions need to be of shape [2,N] or [N,2]')
                    return
                else:
                    if center:
                        #if it's position centers redefine the pos variable
                        pos_center=pos
                        self.pos_center=pos_center
                        if np.shape(pos_center)[0]!=2:
                            if np.shape(pos_center)[1]==2:
                                pos_center=pos_center.T
                                self.pos_center=pos_center
                            else:
                                print('positions need to be of shape [2,N] or [N,2]')
                                return
                        else:
                            self.pos_center=pos_center
                        if phi is not None:
                            if len(phi)==np.shape(pos_center)[1]:
                                self.pos_center-=np.array([pos_offset/2.0*np.cos(phi),
                                                           pos_offset/2.0*np.sin(phi)])
                                self.pos=self.pos_center-np.array([-l/2.0*np.cos(phi),
                                                                   -l/2.0*np.sin(phi)])
                                self.phi=phi
                            else:
                                print('Length of orientations array must correspond to') 
                                print('number of given positions')
                                return
                        else:
                            print('Please set orientations')
                            return
                    else:
                        #if center=True and it's positions, keep the pos variable the same
                        if np.shape(pos)[0]!=2:
                            if np.shape(pos)[1]==2:
                                pos=pos.T
                                self.pos=pos
                            else:
                                self.pos=pos
                        if phi is not None:
                            if len(phi)==np.shape(pos)[1]:
                                self.pos-=np.array([pos_offset/2.0*np.cos(phi),pos_offset/2.0*np.sin(phi)])
                                self.pos_center=self.pos+np.array([-l/2.0*np.cos(phi),-l/2.0*np.sin(phi)])
                                self.phi=phi
                            else:
                                print('Length of orientations array must correspond to number of given') 
                                print('positions')
                                return
                        else:
                            print('Please set orientations')
                            return
    #                 self.bl=bl
    #                 self.reset_calculated_variables()
        
        
        def _eliminate_overlaps(self):
                overlaps_exist=self._check_for_overlaps()
                if overlaps_exist:
                    print('moving ellipses to get rid of intersections')
                    self._reposition_to_eliminate_overlaps()
                overlaps_removed_successfully = not self._check_for_overlaps()    
                return overlaps_removed_successfully
            
            
        def _check_for_overlaps(self):
                overlaps_exist=False
                metric_distance_center = self._calc_metric_distances()
                self.metric_distance_center = metric_distance_center
                # if any two ellipses are closer than 1 bodylength from each other
                if np.sum(self.metric_distance_center<1.):
                    potential_overlaps=np.array([np.array([a,b]) for a in range(self.n) 
                                                 for b in range(a) if 
                                                 self.metric_distance_center[a,b]<1.]).T
                    i=0
                    #used to return empty list if no overlaps found, 
                    #now if the list is empty, there are no overlaps
                    if len(potential_overlaps)==0:
                        overlaps_exist=False
                    else:
                        while i in range(len(potential_overlaps[0])):
                            id_1=potential_overlaps[0,i]
                            id_2=potential_overlaps[1,i]
                            if self._check_ellipse_pair_for_overlap(id_1,id_2):
                                overlaps_exist=True
                                i=np.inf
                            i+=1
                return overlaps_exist
            
        def _calc_metric_distances(self):
                '''
                calculates the euclidean distance between all 
                the geometric centers of the ellipses, accessible 
                via self.metric_distance_center
                '''
                z_center=np.array([[complex(p[0],p[1]) for p in self.pos_center.T]])
                metric_distance_center=abs(z_center.T-z_center)
                return metric_distance_center
            
        def _check_ellipse_pair_for_overlap(self,id1,id2):
                '''determines if ellipse with id1 and ellipse with id2 are intersecting '''
                phi1=self.phi[id1]
                phi2=self.phi[id2]
                pos1=self.pos_center[:,id1]
                pos2=self.pos_center[:,id2]
                w=self.w 
                pos1_eye=self.pos[:,id1]
                pos2_eye=self.pos[:,id2]

                ellipses = [(pos1[0], pos1[1], 1, w/2.0, phi1), (pos2[0], pos2[1], 1, w/2.0, phi2)]
                ellipse_a, ellipse_b =_ellipse_polyline(ellipses)
                are_intersecting = _intersections(ellipse_a,ellipse_b)
                return are_intersecting
            
            
        def _reposition_to_eliminate_overlaps(self,fileName='random',lamda1=0.05, overdamp=0.5):
            '''This function uses C++-code to shift and turn the ellipses
               such that they don't intersect anymore, positions and
               orientations are exchanged via temporary txt files
                    - lamda1 - coefficient of the repulsion area of the cells 
                                (their main body) (0.01 - 0.05) \n");
                    - overdamp - coeffiecient that controls cell inertia (0 -1).'''

            print('Function currently not working')


            
        def calc_visfield(self,check_intersects=True,reposition=True):
            # initialize class properties (used as global variables)
            self._reset_calculated_variables()
            # calculate metric distances between ellipses
            self._calc_metric_distances()
            # calculate tangent points (=outmost points on ellipses in visual field of other ellipses)
            inters=self.calc_tps(check_intersects=check_intersects)
            if inters!=-1 and reposition:
                print('moving ellipses to get rid of intersections')
                self.remove_intersections()
                self._calc_metric_distances()
                inters=self.calc_tps(check_intersects=check_intersects)
            if inters==-1:
                #self.calc_vis_angle()
                #self.rank_vis_angles()
                self._calc_vis_field_and_ang_area()
                #print('calc_visfields' + str(self.angular_area))
                #self.calc_vis_segments()
                #self.calc_ang_area()
                return 1
            else:
                print('FAILED TO REMOVE INTERSECTIONS!')
                return 0
            
        def calc_tps(self,check_intersects=True):
            ''' calculates the tangent points (tps) of ellipses after checking that they don't overlap'''
            if not check_intersects:
                print("You are not checking for intersections of the ellipses. In case an eye of one ellipses\
                    lies inside the body of another ellipses, the analytical calculation will not work\
                    and you will get an error. ")
            check1=-1
            check2=-1
            if check_intersects:
                check1,check2=self.check_intersects()
            if check1!=-1:
                print('remove_intersections() needed')
            else:
                if check2!=-1:
                    eye_inside=tuple([list(a) for a in np.array(check2).T])
                '''###########   calculate tangent points  #########################'''
                z=np.array([[complex(p[0],p[1]) for p in self.pos.T]])
                z_center=np.array([[complex(p[0],p[1]) for p in self.pos_center.T]])
                r=abs(z_center.T-z)
                self.md_eye=abs(z.T-z)
                self.md_center=abs(z_center.T-z_center)
                #indices ij: abs(z_center(i)-z(j)), j is observer, i target
                '''to avoid errors in further calc. result for these will be set manually'''
                np.fill_diagonal(self.md_center,float('NaN'))
                np.fill_diagonal(self.md_eye,float('NaN'))
                np.fill_diagonal(r,float("NaN"))
                if isinstance(check2,list):
                    if len(check2)!=0:
                        r[eye_inside]=float("NaN")
                    check1=-1
                '''initialize variables'''
                w=self.w
                phi_m=np.array([self.phi,]*self.n).transpose()
                tp_subj=[]
                tp_obj=[]
                pt_subj=np.zeros(2)
                pt_obj=np.zeros(2)
                theta_tp=0.0
                r_tp=0.0
                x=self.pos[0]
                y=self.pos[1]
                x_center=self.pos_center[0]
                y_center=self.pos_center[1]
                rel_x=x_center.reshape(len(x_center),1)-x #entry(ij)=pos(i)-pos(j)
                rel_y=y_center.reshape(len(y_center),1)-y

                '''relative position of i to j'''
                theta=np.arctan2(rel_y,rel_x)
                '''calculate tangent points' parameter psi in parametric ellipse eq.'''
                psi=psi_go(w,r,theta,phi_m)
                for p in psi:
                    '''calculate tangent point from psi in local polar coordinates'''
                    pt_subj=ellipsepoints_forgo(r,theta,phi_m,p,w)
                    z_pt_subj=pt_subj[0]+1j*pt_subj[1]
                    theta_tp=_cast_to_pm_pi(np.arctan2(pt_subj[1],pt_subj[0])-self.phi)
                    r_tp=abs(z_pt_subj)
                    np.fill_diagonal(r_tp,0.0)
                    tp_subj.append(np.array([r_tp,theta_tp]))
                    '''transform tp to cartesian global coordinates'''
                    pt_obj=pt_subj+np.array([np.array([self.pos[0],]*self.n),np.array(\
                    [self.pos[1],]*self.n)])
                    np.fill_diagonal(pt_obj[0],0.0)
                    np.fill_diagonal(pt_obj[1],0.0)
                    tp_obj.append(pt_obj)
                self.tp_subj_pol=np.array(tp_subj)
                self.tp_obj_cart=np.array(tp_obj)

                if check2!=-1:
                    self.eyeinside=tuple(eye_inside)
            return check1    
        
        def check_intersects(self):
                ''' returns -1 if there are no intersections between any of the ellipses in the swarm
                    if intersections are presents a list of ids of intersecting ellilpses is returned
                    together with a list of ids where the eye of the second entry is in the body of the
                     first entry'''
                md_center = self._calc_metric_distances()
                intersecting=False
                possible_intersect=np.array([np.array([a,b]) for a in range(self.n) 
                                             for b in range(a) if md_center[a,b]<1.]).T
                intersect_list=[]
                eye_inside_list=[]
                if np.sum(possible_intersect)!=0:
                    '''check if canditates actually intersect and raise error if intersection is found '''
                    for i in range(len(possible_intersect[0])):
                        test=self.ellipse_intersect(possible_intersect[0,i],possible_intersect[1,i])
                        if test[0]:
                            intersect_list.append(list(possible_intersect[:,i]))
                            intersecting=True
                            if test[1]==True:
                                eye_inside_list.append(list(possible_intersect[:,i]))
                            if test[2]==True:
                                eye_inside_list.append(list(np.flip(possible_intersect[:,i])))
                if intersecting:
                    return [intersect_list,eye_inside_list]
                else:
                    return [-1,-1]     
                
        def ellipse_intersect(self,id1,id2):
                '''determines if ellipse with id1 and ellipse with id2 are intersecting
                returns boolean array with 3 entries:
                    0: Do the ellipses intersect? (True/False)
                    1: is the eye of ellipses id2 inside the body of ellipse id1?
                    2: is the eye inside the other way around?
                '''
                phi1=self.phi[id1]
                phi2=self.phi[id2]
                pos1=self.pos_center[:,id1]
                pos2=self.pos_center[:,id2]
                w=self.w 
                pos1_eye=self.pos[:,id1]
                pos2_eye=self.pos[:,id2]

                ellipses = [(pos1[0], pos1[1], 1, w/2.0, phi1), (pos2[0], pos2[1], 1, w/2.0, phi2)]
                a, b =_ellipse_polyline(ellipses)
                inter=_intersections(a,b)
                if inter:
                    two_in_one=point_inside(pos2_eye,a)
                    one_in_two=point_inside(pos1_eye,b)	
                    return np.array([True,two_in_one,one_in_two])
                else:
                    return np.array([False,False,False])                
                
        
        def _calc_vis_field_and_ang_area(self):

            '''1. Calculates the visual field for each ellipse and saves it to 
               self.visual_field, an nxnx2(n-1) array, indices ijk as follows:
               i: id of ellipse visible/lower boundary/upper boundary
               j: viewer id
               k: which section of visual field
               (a np.nan entry means no occlusion of visual field in this area)
               2. then calculates the angular area of each ellipse in the visual field of all 
               other ellipses and saves it to self.ang_area, a numpy nxn array 
               indices ij:
               i: seen individual (the one who is seen by individual j)
               j: focal individual (the one whose visual field is given)'''

            # get ray angles for each ellipse
            angles=self.tp_subj_pol[:,1].flatten(order='f')
            angles=np.sort(angles[~np.isnan(angles)].reshape(2*(self.n-1),self.n,order='f').T)
            
            assert np.logical_and(angles.all()<=np.pi, 
                                  angles.all()>=-np.pi), 'angles are not in pm pi interval'
            between_angles=_cast_to_pm_pi(np.diff(angles,
                                                  append=(2.*np.pi+angles[:,0]).reshape(self.n,1),
                                                  axis=1)/2.+angles)
            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            #  transformation of angles for the calculation of intersection points
            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

            # transform the local angles into points in global cartesian coordinates
            phi=self.phi
            phi_hlp=np.repeat(phi.reshape(self.n,1),2*(self.n-1),axis=1)
            transf_betw_ang=between_angles+phi_hlp
            raypoints=np.array([np.cos(transf_betw_ang),
                                np.sin(transf_betw_ang)])+np.tile(self.pos,
                                                                  ((self.n-1)*2,1,1)).transpose(1,2,0)

            # here we need to transform the raypoints from global coordinates to local 
            # ones of the ellipse that we want to check of intersections 
            # (in a manner that will set up a nested for loop)
            raypoints=np.tile(raypoints,(self.n,1,1,1)).transpose(1,0,2,3) 
            #indices: x/y ,N repetitions (in which coordinate system),focalid 
            #(seen from which eye),raypoints (which tangent point)
            pos_hlp=np.tile(self.pos_center,(2*(self.n-1),1,1)).transpose(1,2,0)
            pos_hlp=np.tile(pos_hlp,(self.n,1,1,1)).transpose(1,2,0,3)
            #indices: ijkl x/y,id (coordinate syst.=the individual that intersections will 
            #be found for), repetition (which eye), repetitions (which tangent point)
            # shifting the raypoints to a coordinate system with origin in the center of 
            #the ellipse j (the one that intersections will be found for)
            raypoints-=pos_hlp

            #now go to polar coordinates and rotate the points by -phi, 
            # to orient the ellipse j along positive x-axis in the respective
            # coordinate system (this is needed because the function calculating 
            # intersections assumes an ellipse at the center with this orientation)
            r=np.sqrt(raypoints[0]**2+raypoints[1]**2)
            theta=np.arctan2(raypoints[1],raypoints[0])
            phi_hlp=np.tile(phi,(self.n,(self.n-1)*2,1)).transpose(2,0,1)
            theta-=phi_hlp
            # now the transofmration is over
            raypoints=np.array([r*np.cos(theta),r*np.sin(theta)])

            # Now we need to similarly transform the eye positions from 
            # global to local (in a manner that will set up a nested for loop)
            # (the id of the viewer ellipse is the second last index, thus 
            # the array needs to have repetitions for all other axes)
            eyes=np.tile(self.pos,(2*(self.n-1),1,1)).transpose(1,2,0)
            eyes=np.tile(eyes,(self.n,1,1,1)).transpose(1,0,2,3)
            #shift coordinate system origins
            eyes-=pos_hlp
            #rotate coordinate systems
            r=np.sqrt(eyes[0]**2+eyes[1]**2)
            theta=np.arctan2(eyes[1],eyes[0])
            theta-=phi_hlp
            eyes=np.array([r*np.cos(theta),r*np.sin(theta)])
            #transformation done
            #+++++++++++++++++++++++++++++++++++++++++++++++++++++++
            #         Calculation of intersection points            
            #+++++++++++++++++++++++++++++++++++++++++++++++++++++++
            inters=get_ellipse_line_intersection_points(eyes,raypoints,self.w)
            inters=remove_self_intersections(inters,self.n)
            # indices: [x/y, which intersection, on which ellipse, 
            # for which viewer, for which ray]
            #+++++++++++++++++++++++++++++++++++++++++++++++++++++++

            # all intersection points are still in coordinates of 
            # the 'on which ellipse' ellipse, transform to global coordinates next:
            #1. rotate by +phi 
            theta=np.arctan2(inters[1],inters[0])+phi_hlp     
            r=np.sqrt(inters[0]**2+inters[1]**2)
            inters=np.array([r*np.cos(theta),r*np.sin(theta)])
            # 2. and shift position of origin
            pos_hlp=np.tile(pos_hlp,(2,1,1,1,1)).transpose(1,0,2,3,4)
            inters=inters+pos_hlp

            # in order to decide which intersection point is closest to an 
            # ellipse we need to move to the coordinate system of the ellipse 
            # which is emitting the rays from its eye (second last index)
            # (we skip the rotation because we are only interested in the
            # distances r anyways)
            pos_hlp=np.tile(self.pos,(2*(self.n-1),1,1)).transpose(1,2,0)
            pos_hlp=np.tile(pos_hlp,(self.n,1,1,1)).transpose(1,2,0,3)
            pos_hlp=np.tile(pos_hlp,(2,1,1,1,1)).transpose(1,0,3,2,4)
            #shift to the local coordinates
            inters-=pos_hlp
            #calculate the distances:
            r=np.sqrt(inters[0]**2+inters[1]**2)

            #Here want to find for each ray emitted from the eye of a viewer ellipse, 
            # the id of the closest ellipse it intersects with
            out=np.empty([self.n,(self.n-1)*2],dtype=float)
            closest_id=get_closest_id(r,out,self.n)

            self.visual_field=np.stack([closest_id,angles,np.roll(angles,-1,axis=-1)]) 
            # 1st index: id of ellipse visible/lower boundary/upper boundary
            # 2nd index: viewer id
            # 3rd index: which section of visual field

            area=np.stack([closest_id,(np.diff(self.visual_field[1::,:,:],axis=0)%np.pi)[0]])
            # id and area for each section of visual field of each ellipse
            # indices ijk:
            # i: id/angle
            # j: viewer id
            # k: section id
            # calculate angular area:
            angular_area=np.zeros([self.n,self.n],dtype=float)
            for i in range(self.n):
                mask=area[0]==i
                angular_area[i,:]=np.sum(mask*area[1],axis=-1)
            self.ang_area=angular_area
            #0 < wij = ang_area/pi < 1


            
        #Plotting-related functions    
        def plot_ellipses(self,fig=None,ax=None,color='seagreen',zorder=100,alpha=0.7,
                          show_index=False,edgecolor='none', cmap=cmx.Greys,
                          show_eyes=True, eyecolor='k',eyesize=5,edgewidth=1,
                          z_label='',norm_z=False,show_colorbar=True):
            ellipses=[]
            if fig is None:
                fig=plt.gcf()
            if ax is None:
                ax=plt.gca()
            if type(color)==str or np.shape(color)==(4,) or np.shape(color)==(3,):
                color=[color for i in range(self.n)]
            else:
                cmax=np.amax(color)
                cmin=np.amin(color)
                cmap_z=cmap
                if not norm_z:
                    color=cmap((color-cmin)/(cmax-cmin))
                    norm_z=cm.colors.Normalize(vmin=cmin,vmax=cmax)
                    print('creating norm')
                else:
                    color=cmap(norm_z(color))

                if show_colorbar:
                    ax1 = fig.add_axes([0.2, 0.2, 0.6, 0.03])
                    cb_z =colorbar.ColorbarBase(ax1, cmap=cmap_z,norm=norm_z, 
                                                orientation='horizontal',
                                                label=z_label)


            for i in range(self.n):
#                 print('plot in draw: ' + str(self.phi))
                ellipses.append(Ellipse(self.pos_center[:,i],self.w,1.0,
                                        _cast_to_pm_pi(self.phi[i])*180.0/np.pi-90.0))
            for i in range(self.n):
                ax.add_artist(ellipses[i])
                ellipses[i].set_clip_box(ax.bbox)
                ellipses[i].set_facecolor(color[i])
                ellipses[i].set_alpha(alpha)
                ellipses[i].set_edgecolor(edgecolor)
                ellipses[i].set_linewidth(edgewidth)
                ellipses[i].set_zorder(zorder)
                if show_index:
                    ax.text(self.pos_center[0,i],self.pos_center[1,i],str(i))
            if show_eyes:
                if eyecolor=='map':
                    self.draw_eyes(ax,color=color,size=eyesize)
                else:
                    self.draw_eyes(ax,color=eyecolor,size=eyesize)
            ax.set_xlim(np.amin(self.pos_center[0])-1,np.amax(self.pos_center[0])+1)
            ax.set_ylim(np.amin(self.pos_center[1])-1,np.amax(self.pos_center[1])+1)
            ax.set_aspect('equal')

        def draw_eyes(self,ax,color='k',size=20):
                ax.scatter(self.pos[0,:],self.pos[1,:],color=color,s=size,zorder=10000)     
        
        
#         def calculate_distance_matrix(self, pos):
#             print(self.pos)
#             X=np.reshape(pos[:],(-1,1))
#             Y=np.reshape(pos[:],(-1,1))
#             dX=np.subtract(X,X.T)
#             dY=np.subtract(Y,Y.T)
#             distmatrix=np.sqrt(dX**2+dY**2)
#             return distmatrix,dX,dY
        
        def calc_link_weight(self, dist,alpha):
            return 1./(1.+dist**alpha)


        def calculate_links_with_weights(self, adjM, distmatrix, alpha):
            if alpha is not None:
                w_adjM = self.calc_link_weight(distmatrix, alpha)*adjM
                avg_lw = np.mean(w_adjM[w_adjM>0])
                instrength=np.sum(w_adjM,axis=0)
            else:
                w_adjM = adjM
                avg_lw = np.mean(w_adjM[w_adjM>0])
                instrength=np.sum(w_adjM,axis=0)
            return w_adjM, avg_lw, instrength
        
        def binary_visual_network(self,threshold=0.,return_networkX=False):
                if np.sum(self.angular_area)==0:
                    self._calc_visual_fields()
                adjacency_matrix=np.array(self.ang_area>threshold,dtype=int)
                if return_networkX:
                    return [adjacency_matrix,self._create_network_graph(adjacency_matrix)]
                else:
                    return adjacency_matrix 
        
        
        def _calc_visual_fields(self):
                '''Calculates the visual field of all ellipses and returns 1 if successfull, 0 if not
                    The calculated quantities are saved in the corresponding properties of the 
                    class instance,
                    e.g. self.angular_area
                    '''
                if np.sum(self.metric_distance_center)==0:
                    self._calc_metric_distances()
                self.calc_tps()
                self._calc_vis_field_and_ang_area()
                
                
        def _create_network_graph(self,adjacency_matrix,allinfo=True,plotting_threshold=0.):
                network=nx.DiGraph(adjacency_matrix)
                if allinfo:
                        for i in range(len(adjacency_matrix[0])):
                                network.nodes()[i]['pos']=self.pos[:,i]
                                network.nodes()[i]['phi']=self.phi[i]
                return network   
            
            
        def draw_binary_network(self,network,fig=None,ax=None,rad=0.0,draw_ellipses=True,
                                ellipse_edgecolor='k',ellipse_facecolor='none',link_zorder=10,
                                show_index=False,scale_arrow=10,linkalpha=0.5,lw=0.8,arrowstyle='-|>',
                                linkcolor='0.4'):
                '''
                INPUT:
                network                 nx.DiGraph(p)
                
                '''
                if fig is None:
                        fig=plt.gcf()
                if ax is None:
                        ax=plt.gca()
                l=self.l
                w=self.w        
                for n in network:
                        if show_index:
                                ax.text(network.nodes[n]['pos'][0],network.nodes[n]['pos'][1],str(int(n)))      
                        c=Ellipse(network.nodes[n]['pos']+np.array([-l/2.0*np.cos(network.nodes[n]['phi']),
                                                                    -l/2.0*np.sin(network.nodes[n]['phi'])]),
                                  w,1.0,network.nodes[n]['phi']*180.0/np.pi-90.0)
                        ax.add_patch(c) 
                        c.set_facecolor(ellipse_facecolor) 
                        if draw_ellipses:
                                c.set_edgecolor(ellipse_edgecolor)
                        else:
                                c.set_edgecolor('none')
                        network.nodes[n]['patch']=c
                seen={}
                
                for (u,v,d) in network.edges(data=True):
                
                        #if d['weight']>=threshold:
                        n1=network.nodes[u]['patch']
                        n2=network.nodes[v]['patch']
                       
                        if (u,v) in seen:
                                rad=seen.get((u,v))
                                rad=(rad+np.sign(rad)*0.1)*-1
                        e = FancyArrowPatch(n1.center,n2.center,patchA=n1,patchB=n2,
                                                                arrowstyle=arrowstyle,
                                                                mutation_scale=scale_arrow,
                                                                connectionstyle='arc3,rad=%s'%rad,
                                                                lw=lw,
                                                                alpha=linkalpha,
                                                                color=linkcolor,zorder=link_zorder)
                        seen[(u,v)]=rad
                        ax.add_patch(e)
                ax.set_xlim(np.amin(self.pos_center[0])-1,np.amax(self.pos_center[0])+1)
                ax.set_ylim(np.amin(self.pos_center[1])-1,np.amax(self.pos_center[1])+1)
                ax.set_aspect('equal')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.tick_params(axis='both', colors='0.5')
                ax.spines['bottom'].set_color('0.5')
                ax.spines['left'].set_color('0.5')            
                

        def draw_weighted_network(self,network,alpha,fig=None,ax=None,rad=0.0,draw_ellipses=True,
                                  ellipse_edgecolor='k',ellipse_facecolor='none',link_zorder=10,
                                  show_index=False,scale_arrow=10,linkalpha=1,lw=1.2,
                                  arrowstyle='-',linkcolor='0.4', weight_given=True):
                """
                Draw a visual network with weighted links of one random node displayed
                """
                if fig is None:
                        fig=plt.gcf()
                if ax is None:
                        ax=plt.gca()
                l=self.l
                w=self.w        
                for n in network:
                        if show_index:
                                ax.text(network.nodes[n]['pos'][0],network.nodes[n]['pos'][1],str(int(n)))      
                        c=Ellipse(network.nodes[n]['pos']+np.array([-l/2.0*np.cos(network.nodes[n]['phi']),
                                                                    -l/2.0*np.sin(network.nodes[n]['phi'])]),
                                  w,1.0,network.nodes[n]['phi']*180.0/np.pi-90.0)
                        
                        ax.add_patch(c) 
                        c.set_facecolor(ellipse_facecolor) 
                        if draw_ellipses:
                                c.set_edgecolor(ellipse_edgecolor)
                        else:
                                c.set_edgecolor('none')
                        network.nodes[n]['patch']=c
                        
                seen={}
                data = []
                random_node = np.random.choice(n)
                viridis = cm = plt.get_cmap('rainbow') 
                
                for (u,v,d) in network.edges(data=True):
                    
                        n1=network.nodes[u]['patch']
                        n2=network.nodes[v]['patch']
                        
                        max_weight = math.dist(network.nodes[0]['patch'].center, 
                                                 network.nodes[1]['patch'].center)
                        if weight_given:
                            max_weight = self.calc_link_weight(max_weight, alpha)

                        cNorm  = colors.PowerNorm(gamma=0.1, vmin=0, vmax=max_weight)
                        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=viridis)
                        
                        c = np.linspace(0, max_weight)
                        
                        if (u,v) in seen:
                                rad=seen.get((u,v))
                                rad=(rad+np.sign(rad)*0.1)*-1
                                
                        if u==random_node or v==random_node:
                            length = math.dist(n1.center, n2.center)
            
                            if weight_given:
                                length = self.calc_link_weight(length, alpha)

                            data.append(length)
                            colorVal = scalarMap.to_rgba(length)
                            linkcolor = colorVal
                        
                            e = FancyArrowPatch(n1.center,n2.center,patchA=n1,patchB=n2,
                                                                    arrowstyle=arrowstyle,
                                                                    mutation_scale=scale_arrow,
                                                                    connectionstyle='arc3,rad=%s'%rad,
                                                                    lw=lw,
                                                                    alpha=linkalpha,
                                                                    color=linkcolor,zorder=link_zorder)
                            seen[(u,v)]=rad
                            ax.add_patch(e)
                
                ax.set_xlim(np.amin(self.pos_center[0])-1,np.amax(self.pos_center[0])+1)
                ax.set_ylim(np.amin(self.pos_center[1])-1,np.amax(self.pos_center[1])+1)
                ax.set_aspect('equal')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.tick_params(axis='both', colors='0.5')
                ax.spines['bottom'].set_color('0.5')
                ax.spines['left'].set_color('0.5') 
                 
        
        
        def lin_equ(self, l1, l2):
                """Line encoded as l=(x,y)."""
                m = np.float((l2[1] - l1[1])) / np.float(l2[0] - l1[0])
                c = (l2[1] - (m * l2[0]))
                return m, c
        
        
        def draw_regions(self,network,fig=None,ax=None,rad=0.0,draw_ellipses=True,
                                ellipse_edgecolor='k',ellipse_facecolor='none',link_zorder=10,
                                show_index=False,scale_arrow=10,linkalpha=1,lw=0.8,arrowstyle='-',
                                linkcolor='0.4'):
                """
                Draw a network with thick lines separating four regions (up, right, down, left) 
                of one random node (!!Does not work for nodes that are at the corner)
                """
                if fig is None:
                        fig=plt.gcf()
                if ax is None:
                        ax=plt.gca()
                l=self.l
                w=self.w     
                
                positions = []
                for n in network:
                        positions.append(np.max(network.nodes[n]['pos']))
                    
                        if show_index:
                                ax.text(network.nodes[n]['pos'][0],network.nodes[n]['pos'][1],str(int(n)))      
                        c=Ellipse(network.nodes[n]['pos']+np.array([-l/2.0*np.cos(network.nodes[n]['phi']),
                                                                    -l/2.0*np.sin(network.nodes[n]['phi'])]),
                                  w,1.0,network.nodes[n]['phi']*180.0/np.pi-90.0)
                        ax.add_patch(c) 
                        c.set_facecolor(ellipse_facecolor) 
                        if draw_ellipses:
                                c.set_edgecolor(ellipse_edgecolor)
                        else:
                                c.set_edgecolor('none')
                        network.nodes[n]['patch']=c
                seen={}

                limits = np.min(positions), np.max(positions)
                x = np.linspace(limits[0], limits[1])
            
                random_node = np.random.choice(n)
                node_center = network.nodes[random_node]['pos']
                distance = math.dist(network.nodes[0]['pos'], network.nodes[1]['pos'])
                diag_distance = np.round(distance*2/np.sqrt(2), 5)        
        
                arrows = []
                ms = []
                cs = []
                diagonal = []
                for (u,v,d) in network.edges(data=True):
               
                        n1=network.nodes[u]['patch']
                        n2=network.nodes[v]['patch']
                       
                        if (u,v) in seen:
                                rad=seen.get((u,v))
                                rad=(rad+np.sign(rad)*0.1)*-1
                                
                        if u==random_node or v==random_node:
                            length = np.round(math.dist(n1.center, n2.center), 5)
                            if np.round(math.dist(n1.center, n2.center), 5)==diag_distance:
                                diagonal_center = n1.center 
                                diagonal.append(diagonal_center)
                                if diagonal_center[0]!=node_center[0] and diagonal_center[1]!=node_center[1]:
                                    m, c = self.lin_equ(diagonal_center, node_center)
                                    ms.append(m)
                                    cs.append(c)
                                    ax.axline(diagonal_center, node_center, linewidth=3, 
                                              alpha=0.5, color='r')
                                    
            
                        #####################################
                            e = FancyArrowPatch(n1.center,n2.center,patchA=n1,patchB=n2,
                                                                    arrowstyle=arrowstyle,
                                                                    mutation_scale=scale_arrow,
                                                                    connectionstyle='arc3,rad=%s'%rad,
                                                                    lw=lw,
                                                                    alpha=linkalpha,
                                                                    color=linkcolor,zorder=link_zorder)
                            seen[(u,v)]=rad
                            ax.add_patch(e)
                            arrows.append(e.get_path())
                        ##################################
            
        
                print('ms: ' +str(ms))
                print('cs: ' +str(cs))
        
                x_left = np.linspace(limits[0], node_center[0])
                line1L = ms[0]*x_left+cs[0]
                line2L = ms[1]*x_left+cs[1]
                plt.fill_between(x_left, line1L, line2L, color='red', alpha=0.3)
                
                x_right = np.linspace(node_center[0], limits[1])
                line1R = ms[0]*x_right+cs[0]
                line2R = ms[1]*x_right+cs[1]
                plt.fill_between(x_right, line1R, line2R, color='green', alpha=0.3)
                
                
                ax.set_xlim(np.amin(self.pos_center[0])-1,np.amax(self.pos_center[0])+1)
                ax.set_ylim(np.amin(self.pos_center[1])-1,np.amax(self.pos_center[1])+1)
                ax.set_aspect('equal')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.tick_params(axis='both', colors='0.5')
                ax.spines['bottom'].set_color('0.5')
                ax.spines['left'].set_color('0.5') 
                
            
 
        def get_angles(self,network,alpha, fig=None,ax=None,rad=0.0,draw_ellipses=True,
                                ellipse_edgecolor='k',ellipse_facecolor='none',link_zorder=10,
                                show_index=False,scale_arrow=10,linkalpha=0.5,lw=0.8,arrowstyle='-',
                                linkcolor='blue'):
                """
                Returns two dictionaries:
                1. 
                """
                l=self.l
                w=self.w     
                
                node_loc = []
                node_angles = []
                for n in network:
                        node_loc.append(network.nodes[n]['pos'])
                        node_angles.append(network.nodes[n]['phi']) #angle in radians
                    
                        if show_index:
                            
                                ax.text(network.nodes[n]['pos'][0],network.nodes[n]['pos'][1],str(int(n)))      
                        c=Ellipse(network.nodes[n]['pos']+np.array([-l/2.0*np.cos(network.nodes[n]['phi']),
                                                                    -l/2.0*np.sin(network.nodes[n]['phi'])]),
                                  w,1.0,network.nodes[n]['phi']*180.0/np.pi-90.0)
                        network.nodes[n]['patch']=c
                seen={}

                random_node = np.random.choice(n)
                node_center = network.nodes[random_node]['pos']
                distance = math.dist(network.nodes[0]['pos'], network.nodes[1]['pos'])
                diag_distance = np.round(distance*2/np.sqrt(2), 5)        
        
                seen_node = []
                arrows = []
                diagonal = []
                arrows = []
                
                for (u,v,d) in network.edges(data=True):
                    
                        n1=network.nodes[u]['patch']
                        n2=network.nodes[v]['patch']
                       
                        if (u,v) in seen:
                                rad=seen.get((u,v))
                                rad=(rad+np.sign(rad)*0.1)*-1
                                
                        arrows.append((n1.center,n2.center))
                
                all_areas = []
                all_weight_areas = []
                for n in network:
                    node_center = network.nodes[n]['pos']
                    seen_nodes = []
                    joined_string = [element for element in node_center] 

                    for i in range(len(arrows)):
                        if all(i==j for i, j in zip(arrows[i][0], joined_string)):
                            check = arrows[i][1]
                            seen_nodes.append(check)
                        elif all(i==j for i, j in zip(arrows[i][1], joined_string)):
                            check = arrows[i][0]
                            seen_nodes.append(check)

                    seen_unique = set(map(tuple, seen_nodes))
                    #check the centers of the seen nodes only
                    seen_loc = []
                    for s in range(len(seen_unique)):
                        for n in range(len(node_loc)):
                            if np.all(list(seen_unique)[s]==node_loc[n]):
                                seen_loc.append(node_loc[n])

                    #get the coordinates of the centers of the nodes to create vectors
                    list_r_ij = [[node_center, r_ij] for r_ij in seen_loc]
                    x_points = [(list_r_ij[i][1][0] - list_r_ij[i][0][0]) for i,x in enumerate(list_r_ij)]
                    y_points = [(list_r_ij[i][1][1] - list_r_ij[i][0][1]) for i,x in enumerate(list_r_ij)]
                    vectors = [[x, y] for x, y in zip(x_points, y_points)]
                    norm_vectors = [n/np.linalg.norm(n) for n in vectors]
                    #vector from the node of interest is always [1, 0]
                    cv = [1, 0]
                    angle_list = [math.atan2(cv[0]*nv[1]-cv[1]*nv[0],cv[0]*nv[0]+cv[1]*nv[1]) 
                                  for nv in norm_vectors]
                    vector_length = [np.linalg.norm(n) for n in vectors]
                    length = [self.calc_link_weight(l, alpha) for l in vector_length]
                    areas = {'Right': 0,  'Left': 0,  'Up': 0, 'Down':0}
                    weight_areas = {'Right': 0,  'Left': 0,  'Up': 0, 'Down':0}

                    for i in range(len(angle_list)):
                        if (-np.pi/4)<angle_list[i]<(np.pi/4):
                            areas['Right'] += 1
                            weight_areas['Right'] += length[i]/areas['Right']
                        elif (np.pi/4)<angle_list[i]<((3*np.pi)/4):
                            areas['Up'] += 1
                            weight_areas['Up'] += length[i]/areas['Up']
                        elif (((3*np.pi)/4)<angle_list[i]<=(np.pi) or(-(3*np.pi)/4)>angle_list[i]>=(-np.pi)):
                            areas['Left'] += 1
                            weight_areas['Left'] += length[i]/areas['Left']
                        elif (-np.pi/4)>angle_list[i]>(-(3*np.pi)/4):
                            areas['Down'] += 1
                            weight_areas['Down'] += length[i]/areas['Down']
    
                    weights = {k:v/areas[k] if areas[k] else 0 for k, v in weight_areas.items() if k in areas}
                    all_areas.append(areas)
                    all_weight_areas.append(weights)
        
                df = pd.DataFrame(all_areas)
                number_nodes = dict(df.mean())
                
                df_weight = pd.DataFrame(all_weight_areas)
                nodes_weight = dict(df_weight.mean())
                
                return nodes_number, nodes_weight
