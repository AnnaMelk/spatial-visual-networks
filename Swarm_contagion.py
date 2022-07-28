#!/usr/bin/env python
# coding: utf-8

# Contagion model

# In[1]:
import numpy as np
import networkx as nx
from collections import OrderedDict
from numba import jit
from tqdm import tqdm


# In[2]:
def InitParameters(N=100,L=10,adj_matrix=None,simtime=100,output=0.2,dt=0.05,contagion_type='simple',
                   infection_rate=0.1,recovery_rate=0.1,threshold_complexContagion=3.0,steepness_complexContagion=1,
                   pos_noise=0.0,alpha=None,no_init_infected=2,init_infection_type='random',init_infected_nodes=[]):

    params=OrderedDict()
    params['N']=N
    params['L']=L
    params['simtime']=simtime
    params['output'] =output
    params['outstep']=int(output/dt)
    params['dt']=dt
    params['steps']=int(simtime/dt)
    params['alpha']=alpha
    params['contagion_type']=contagion_type
    params['infection_rate']=infection_rate
    params['recovery_rate'] =recovery_rate
    params['threshold_complexContagion']=threshold_complexContagion
    params['steepness_complexContagion']=steepness_complexContagion
    params['pos_noise']=pos_noise
    params['no_init_infected']=no_init_infected
    params['init_infection_type']=init_infection_type
    params['init_infected_nodes']=init_infected_nodes
    return params


def InitSimVars(params):
    simVars=OrderedDict()
    simVars['state']=np.zeros(params['N'])
    simVars['adjM'] =np.zeros((params['N'],params['N']))
    simVars['w_adjM'] =np.zeros((params['N'],params['N']))
    simVars['edgeList']=[]
    simVars['avg_linkweight']=[]
    simVars['avg_instr']=[]
    return simVars

def GenerateLatticeLikePositions(N=100,pos_noise=0.1,L=10):
    pos=np.zeros((N,2))
    n=int(np.sqrt(N))
    for i in range(N):
        pos[i,0] = (i % n)
        pos[i,1] = (i-(i%n))/n
        pos+=pos_noise*(np.random.random((N,2))-0.5)
    return pos


def CalcDistVecMatrix(pos,L,BC=1):
    ''' Calculate N^2 distance matrix (d_ij)
        
        Returns: 
        --------
        distmatrix - matrix of all pairwise distances (NxN)
        dX - matrix of all differences in x coordinate (NxN)
        dY - matrix of all differences in y coordinate (NxN)
    '''
    X=np.reshape(pos[:,0],(-1,1))
    Y=np.reshape(pos[:,1],(-1,1))
    dX=np.subtract(X,X.T)
    dY=np.subtract(Y,Y.T)
    if(BC==0):
        dX[dX>+0.5*L]-=L;
        dY[dY>+0.5*L]-=L;
        dX[dX<-0.5*L]+=L;
        dY[dY<-0.5*L]+=L;
    distmatrix=np.sqrt(dX**2+dY**2)
    return distmatrix,dX,dY


def GenerateSpatialNetworkAdjM(distmatrix,maxLinkRange=2.0):
    adjM=np.int32(distmatrix<maxLinkRange)
    np.fill_diagonal(adjM,0.0)
    return adjM


def CalcEdgeListFromAdjM(adjM):
    tmp_edges=np.argwhere(adjM)
    edgeList=tmp_edges[(tmp_edges[:,0]-tmp_edges[:,1])>0]
    return edgeList

@jit(nopython=True,parallel = True)
def SigThresh(x,x0=0.5,steepness=10):
    ''' Sigmoid function f(x)=1/2*(tanh(a*(x-x0)+1)
        
        Input parameters:
        -----------------
        x:  function argument
        x0: position of the transition point (f(x0)=1/2)
        steepness:  parameter setting the steepness of the transition.
                    (positive values: transition from 0 to 1, negative values: 
                    transition from 1 to 0)
    '''
    return 0.5*(np.tanh(steepness*(x-x0))+1)


def CalcLinkWeigth(dist,alpha):
    return 1./(1.+dist**alpha)


def CalculateLinksWithWeigths(adjM, distmatrix, alpha):
    if alpha is not None:
        w_adjM = CalcLinkWeigth(distmatrix, alpha)*adjM
        #calculate means for all non-zero values and sum of all links
        avg_lw = np.mean(w_adjM[w_adjM>0])
        instrength=np.sum(w_adjM,axis=0)
    else:
        w_adjM = adjM
        avg_lw = np.mean(w_adjM[w_adjM>0])
        instrength=np.sum(w_adjM,axis=0)
    return w_adjM, avg_lw, instrength


@jit(nopython=True,parallel = True)
def CalculateInfectedNeighborsEdges(state, N, edgeList, w_adjM):
    no_infected_neighbors=np.zeros(N)
    #for edge in edgeList:
    if alpha is not None:
        for i in range(len(edgeList)):
            edge=edgeList[i]
            if(state[edge[0]]==1):
                no_infected_neighbors[edge[1]]+=w_adjM[edge[0],edge[1]]
            if(state[edge[1]]==1):
                no_infected_neighbors[edge[0]]+=w_adjM[edge[0],edge[1]]  
    else:
        for i in range(len(edgeList)):
            edge=edgeList[i]
            if(state[edge[0]]==1):
                no_infected_neighbors[edge[1]]+=w_adjM[edge[0],edge[1]]
            if(state[edge[1]]==1):
                no_infected_neighbors[edge[0]]+=w_adjM[edge[0],edge[1]]   
    return no_infected_neighbors

@jit(nopython=True,parallel = True)
def SimpleContagionInfProb(infection_rate,dt,no_infected_neighbors):
    return 1-(1-infection_rate*dt)**no_infected_neighbors


def ProbabilityInfectionPerTimeStep(no_infected_neighbors,params, degrees):
    if(params['contagion_type']=='complex_numeric'):
        #instead of number of inf neighbors - use aggregated weight of inf neighbors
        #rename no_inf_neighbors to sum_inf_neighbors
        return params['infection_rate']*params['dt']*SigThresh(no_infected_neighbors,
                                                               params['threshold_complexContagion'],
                                                               params['steepness_complexContagion'])
    elif(params['contagion_type']=='complex_fractional'):
        inf_ratio=no_infected_neighbors/degrees
        return params['infection_rate']*params['dt']*SigThresh(inf_ratio,params['threshold_complexContagion'],params['steepness_complexContagion'])
    else:
        return SimpleContagionInfProb(params['infection_rate'],params['dt'],no_infected_neighbors)


@jit(nopython=True,parallel = True)
def LoopOverNodesSimple(state,no_infected_neighbors,infection_rate,recovery_rate,dt):
    for i in range(len(state)):
        if(state[i]==0):
            if(no_infected_neighbors[i]>0):
                inf_prob=1.-(1.-infection_rate*dt)**no_infected_neighbors[i]
                rn=np.random.random()
                if(rn<inf_prob):
                    state[i]=0
        if(state[i]==1):
                rn=np.random.random()
                if(rn<recovery_rate):
                    state[i]=-1
    return state


@jit(nopython=True,parallel = True)
def LoopOverNodesComplex(state,no_infected_neighbors,infection_rate,recovery_rate,dt,threshold_complexContagion,steepness_complexContagion):
    for i in range(len(state)):
        if(state[i]==0):
            if(no_infected_neighbors[i]>0):
                inf_prob=infection_rate*dt*SigThresh(no_infected_neighbors,threshold_complexContagion,steepness_complexContagion)
                rn=np.random.random()
                if(rn<inf_prob):
                    state[i]=0
        if(state[i]==1):
                rn=np.random.random()
                if(rn<recovery_rate):
                    state[i]=-1
    return state


def UpdateStates(simVars,params,verbose=False):
    newState=np.copy(simVars['state'])
    pos=GenerateLatticeLikePositions(params['N'],params['pos_noise'],params['L'])
    params['N']=len(pos)
    distM,dX,dY=CalcDistVecMatrix(pos,params['L'])
    adjM=GenerateSpatialNetworkAdjM(distM)
    w_adjM, avg_wl, instrength = CalculateLinksWithWeigths(adjM, distM, alpha=params['alpha']) 
    simVars['avg_linkweight']=avg_wl
    simVars['avg_instr']=instrength
    no_infected_neighbors=CalculateInfectedNeighborsEdges(simVars['state'],params['N'],simVars['edgeList'], w_adjM)
    prob_of_inf=ProbabilityInfectionPerTimeStep(no_infected_neighbors,params,simVars['degrees'])
    rn=np.random.uniform(size=params['N'])

    susceptibles=(simVars['state']==0)
    to_infect   =(rn<prob_of_inf)
    newState[susceptibles*to_infect]=1

    rn=np.random.uniform(size=params['N'])
    infected    =(simVars['state']==1)
    to_recover  =(rn<params['recovery_rate']*params['dt']) 
    newState[infected*to_recover]=-1

    if(verbose):
        print("nodes infected:",np.argwhere(susceptibles*to_infect))
        print("nodes recovered:",np.argwhere(infected*to_recover))
        print("old state:",simVars['state'])
        print("new state:",newState)
    simVars['state']=newState
    return


def InitOutput(edgeList):
    outdata=OrderedDict()
    outdata['state'] = []
    outdata['ninf']  = []
    outdata['nrec']  = []
    outdata['time']  = []
    outdata['edgeList']  = edgeList
    outdata['avg_linkweight'] = []
    outdata['avg_instr'] = []
    return outdata

def UpdateOutput(outdata,simVars,curr_time):
    if(type(outdata)==type(None)):
        outdata=OrderedDict()
        # if no outdata -> generate list 
        outdata['state']= [simVars['state']]
        outdata['ninf'] = [np.sum(simVars['state']==1)]
        outdata['nrec'] = [np.sum(simVars['state']==-1)]
        outdata['time'] = [curr_time]
        outdata['avg_linkweight'] = [simVars['avg_linkweight']]
        outdata['avg_instr'] = [simVars['avg_instr']]
    else:
        outdata['state'].append(simVars['state'])
        outdata['ninf' ].append(np.sum(simVars['state']==1))
        outdata['nrec' ].append(np.sum(simVars['state']==-1))
        outdata['time' ].append(curr_time)
        outdata['avg_linkweight'].append(simVars['avg_linkweight'])
        outdata['avg_instr'].append(simVars['avg_instr'])
    return outdata


def InitInfection(simVars,params):
    if(len(params['init_infected_nodes'])>0):
        #print("Initially infecting pre-set nodes:",params['init_infected_nodes'])
        simVars['state'][params['init_infected_nodes']]=1
    else:
        if(params['init_infection_type']=='random'):
            to_infect=np.random.choice(np.arange(params['N']),size=params['no_init_infected'],replace=False)
            simVars['state'][to_infect]=1
            print("Initially infecting randomly:",to_infect)  
            ######################################## implement?
        elif(params['init_infection_type']=='cluster'):
            print("Clustered initial infection not implemented yet!")
    return


def TimeIntegrationMatrix(simVars,outdata,params):
    outdata=UpdateOutput(outdata,simVars,0.)
    for s in range(params['steps']):
        UpdateStates(simVars,params)
        if params['output']>0:
            if(s % params['outstep']):
                outdata=UpdateOutput(outdata,simVars,s*params['dt'])
            
        ninf=np.sum(simVars['state']==1)
        nsus=np.sum(simVars['state']==0)
        nrec=np.sum(simVars['state']==-1)
        #stop the simulation of there are not infected nodes left
        if((nrec==params['N'])):
            outdata=UpdateOutput(outdata,simVars,s*params['dt'])
            break
        if((nrec+nsus==params['N'])):
            outdata=UpdateOutput(outdata,simVars,s*params['dt'])
            break    
        if ((ninf!=0) and (nsus!=0)) and params['output']<0:
            outdata=UpdateOutput(outdata,simVars,s*params['dt'])
    return outdata


def SingleRun(params,alpha,init_state=None,adjM=None,pos=None,integration_scheme='matrix'):
    
    if(type(pos)==type(None)):
        pos=GenerateLatticeLikePositions(params['N'],params['pos_noise'],params['L'])
    
    if(type(adjM)==type(None)):
        params['N']=len(pos)
        distM,dX,dY=CalcDistVecMatrix(pos,params['L'])
        adjM=GenerateSpatialNetworkAdjM(distM)
        w_adj, avg_lw =CalculateLinksWithWeigths(adjM, distM, alpha)

    if(np.shape(pos)[0]!=np.shape(adjM)[0]):
        print("Error! number of agents / shapes of adjM and pos do not match.")
        return 1    

    edgeList=CalcEdgeListFromAdjM(adjM)
    simVars=InitSimVars(params)
    outdata=InitOutput(edgeList)
    simVars['adjM']=adjM
    simVars['edgeList']=edgeList
    simVars['degrees']=np.sum(adjM,axis=1)
    
    if(type(init_state)==type(None)):
        InitInfection(simVars,params)
    else:
        simVars['state']=init_state

    if(integration_scheme=='matrix'):
        outdata=TimeIntegrationMatrix(simVars,outdata,params)

    return outdata



# In[3]:
mySwarm=Swarm(n=81, setup='grid', noise_pos=0, noise_phi=0)
visual_amat,visual_network=mySwarm.binary_visual_network(threshold=0.01,return_networkX=True)


# In[4]:
N=swarm.n
init_infected_nodes = np.random.choice(N, 5)
alpha = 1
simple_params=InitParameters(simtime=50,N=N,contagion_type='simple',recovery_rate=0.2,
                         init_infected_nodes=init_infected_nodes, infection_rate=0.5, alpha=alpha,
                         threshold_complexContagion=None,steepness_complexContagion=None)

fig,ax=plt.subplots(1)
for i in range(10):
    outdata = SingleRun(simple_params, adjM=adjacency_matrix, alpha=alpha, pos=swarm.pos.T)
    time=outdata['time']
    infected_fraction=(np.array(outdata['ninf']))/swarm.n
    recovered_fraction=(np.array(outdata['nrec']))/swarm.n
    ax.plot(time, infected_fraction,color='r')
    ax.plot(time, recovered_fraction,color='b')
