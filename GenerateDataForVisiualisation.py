import numpy as np
import ellipse_swarm as esw
import itertools as it
import multiprocessing as mp
import h5py 
sentinel = None


def handle_output(output):
    # Handles the multiprocessing
    hdf = h5py.File('NAME_OF_THE_FILE.h5', 'a') #<- This was used before, instead of 
    # the 'with h5py...as hdf' line. Change back if you encouter a problem with this. 
    # don't forget to also change the last line to include the hdf.close() again
    #with h5py.File('swarm_data.h5', 'a') as hdf:
    while True:
            args = output.get()
            if args:
                param_groupname, datanames, data = args
                for j,name in enumerate(datanames):
                    if param_groupname+'/'+name not in hdf:
                        #print('creating parameter set group '+param_groupname+'/'+name)
                        #print(np.shape(data[j]))
                        hdf.require_dataset(param_groupname+'/'+name,
                                        data=data[j],shape=[1]+list(np.shape(data[j])),chunks=True,dtype='f',maxshape=(None,None,None))
                    else:   
                        #print('appending to set '+param_groupname)
                        hdf[param_groupname+'/'+name].resize((hdf[param_groupname+'/'+name].shape[0]+1),axis=0)
                        hdf[param_groupname+'/'+name][-1]=data[j]
            else:
                break
            
    hdf.close()


def Parallel_Simulation(inqueue, output):
    for state,d,n,nphi,npos,w,st,reposition in iter(inqueue.get, sentinel):  
        i=0
        while i<5:
            swarm=Swarm(setup=state,n=n,dist=d,noise_phi=nphi,noise_pos=npos,w=w,l=0.)
            ok=swarm.calc_visfield(reposition=reposition) 
            print('calculated visual field')
            if ok:    
                        print('worked ok')
                        "FUNCTION"
                        #swarm.calc_only_vis_angle()
                        hxhy=np.vstack([np.cos(swarm.phi),np.sin(swarm.phi)])
                        #data=[swarm.pos,hxhy,swarm.ang_area,swarm.md_center,swarm.vis_angles]
                        data=[swarm.pos,hxhy,swarm.ang_area,swarm.md_center]
                        param_groupname='/'+state+'/N%i'%n+'/w%1.2f'%w+'/noisePos%1.3f'%npos+'/noisePhi%1.5f'%nphi+'/dist%1.3f'%d
                        data_name_list=['positions','hxhy','angularArea','metricDistances']#,'angularAreaNoOcclusions']
                        output.put([param_groupname,data_name_list,data])
                        i=5                
            else:
                i+=1
            if i==5:
                print('tried 5 times',d,n,nphi,npos,w,st)
   


def Simple_Simulation(params):
    # This is used for generating swarms without multiprocessing
    # It is largely identical to the parallel version, just the data is
    # handled differently
    # It would be nice to write the parts that are used in both in an extra function
    # but I haven't gotten around to it
    state,d,n,nphi,npos,w,st,reposition =params
    i=0
    while i<5:
        try:
            swarm=Swarm(setup=state,n=n,dist=d,noise_phi=nphi,noise_pos=npos,w=w,l=0.)
            ok=swarm.calc_visfield(reposition=reposition) 
            print('calculated visual field')
            if ok:    
                    print('worked ok')
                    #swarm.calc_only_vis_angle()
                    hxhy=np.vstack([np.cos(swarm.phi),np.sin(swarm.phi)])
                    #data=[swarm.pos,hxhy,swarm.ang_area,swarm.md_center,swarm.vis_angles]
                    data=[swarm.pos,hxhy,swarm.ang_area,swarm.md_center]
                    print(data[1])
                    param_groupname='/'+state+'/N%i'%n+'/w%1.2f'%w+'/noisePos%1.3f'%npos+'/noisePhi%1.5f'%nphi+'/dist%1.3f'%d
                    data_name_list=['positions','hxhy','angularArea','metricDistances']#,'angularAreaNoOcclusions']
                    nosuccess=False
                    i=5
                    return [param_groupname,data_name_list,data]
        except:
            i+=1
        if i==5:
            print('tried 5 times',d,n,nphi,npos,w,st)



    
if __name__=="__main__":
    parallel=True
    reposition=False # whether to shift the ellipses to avoid overlaps, not eliminating overlaps will cause errors in the visual field calculations at high density
    num_processors=7

    # Set parameters as lists, all combinations are calculated
    states=['grid']
    dist=np.array([20,10,5,2.5,2,1.5,1.])
    number=np.array([49])
    noisephi=[0.4]
    noisepos=[0.1]
    aspect=[0.4]
    stats=np.arange(10) # how many configurations for each set of parameters

    # Set up multiprocessing
    processes=len(dist)*len(number)*len(noisephi)*len(noisephi)*len(aspect)*len(stats)
    if processes<num_processors:
        num_processors=processes
    print('number of processes ',num_processors)
    paramlist= it.product(states,dist,number,noisephi,noisepos,aspect,stats,[reposition])

    # Multiprocessing that writes into one HDF5 file. Because writing into the file can not be done in parallel by many processes, the results need to be queued before
    if parallel:
        output = mp.Queue()
        inqueue = mp.Queue()
        jobs = []
        proc = mp.Process(target=handle_output, args=(output, ))
        proc.start()
        for i in range(num_processors):
            p = mp.Process(target=Parallel_Simulation, args=(inqueue, output))
            jobs.append(p)
            p.start()
        for i in paramlist:
            inqueue.put(i)
        for i in range(num_processors):
            # Send the sentinal to tell Simulation to end
            inqueue.put(sentinel)
        for p in jobs:
            p.join()
        output.put(None)
        proc.join()
    # Version without parallel processing and saving each swarm to a txt file. I have only used this for debugging and you would have to change it to save to a SINGLE hdf5, if you want to run code on a single core only (without multiprocessing) and then use the process_swarm_data.py for processing (or simply use the parallel version above, which already generates the right hdf5 file)
    else:
        data=[]
        for params in paramlist:
            print(params)
            data.append(Simple_Simulation(params))
        #print(np.shape(data))
        for dat in data:
            #print(dat)
            for k,da in enumerate(dat[1]):
                #print(da)
                print('saving to '+dat[0].replace('/','_')+'_'+da+'.txt')
                #print(dat[2][k])
                np.savetxt(dat[0].replace('/','_')+'_'+da+'.txt',dat[2][k])
