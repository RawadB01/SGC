from __future__ import division
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 11:53:39 2018

SGC

@authors: Rawad Bitar, Mary Wootters and Salim El Rouayheb
"""
#%%
import numpy as np
import pandas as pd
from collections import Counter
#from matplotlib2tikz import save as tikz_save
import math



#%%
def genGauData( weight, mean, variance, dataPoints):
    """
    This function geenrates noisy data using Gaussian distribution
    Input: The original model represented by the weight vector, the noise mean and variance, and the number of data points
    Output: the data set x, the labels vector y  

    weight represent the orginal model
    mean and variance: the mean and variance of a gaussian noise that will be added to the target labels
    dataPoints: number of data points
    """
            
    numFeatures = len (weight)    
    x   = np.zeros(shape = (dataPoints, numFeatures))
    y_t = np.zeros(shape = (dataPoints))    
    
    for i in range(dataPoints):
        """
        Un-comment the following to create data vectors with different norms
        that will lead to d_i's that are different by choice
        """
#        vec = np.random.normal(0, 100, numFeatures)
#        if   i < dataPoints / 4:
#            x[i] = (math.sqrt(1) /np.linalg.norm(vec,2)) * vec
#        elif i < dataPoints / 2:
#            x[i] = (math.sqrt(3) /np.linalg.norm(vec,2)) * vec
#        elif i < 3 * dataPoints / 4:
#            x[i] = (math.sqrt(6) /np.linalg.norm(vec,2)) * vec
#        else:
#            x[i] = (math.sqrt(10) /np.linalg.norm(vec,2)) * vec
        
        x[i] = np.random.normal(0, 100, numFeatures) # Generate data vectros using N(0,100)
        y_t[i] = np.random.normal(np.dot(x[i],weight)+mean, variance) # create noisy labels y = N(x*model + noise_mean, noise_var)
    
    
    return x, y_t

#%%
def genData( weight, mean, variance, dataPoints):
    """
    This function geenrates noisy data using uniform distribution over integers
    Input: The original model represented by the weight vector, the noise mean and variance, and the number of data points
    Output: the data set x, the labels vector y  

    weight represent the orginal model
    mean and variance: the mean and variance of a gaussian noise that will be added to the target labels
    dataPoints: number of data points
    """
            
    numFeatures = len (weight)    
    x   = np.zeros(shape = (dataPoints, numFeatures))
    y   = np.zeros(shape = (dataPoints))
    y_t = np.zeros(shape = (dataPoints))
    

    for i in range(dataPoints):
        for j in range (numFeatures):
            x[i][j]=np.random.randint(0,10)
    y_t = (x*weight).sum(axis=1)
    noise = np.random.normal(mean,variance,( dataPoints))
    y = y_t + noise
    
    
    return x, y

#%%
# Some functions that will help the main code look nice
    
def compute_sigma(X, d):
    sigma = (X.shape[0] * d) / int(round(np.linalg.norm(X,'fro') ** 2))
    return sigma

def compute_di(sigma, x):
    di = max (1, int( round( (np.linalg.norm(x, 2) ** 2) * sigma ) ) )
    return di


def compute_alpha(norm, itera):    
        epsilon = 10 ** -50
        newi = (itera + 1) ** power
        alpha = 7 * (math.log(1 / (epsilon ** 2) ) ) / ( newi * norm )
        
        return alpha
    

def L2_cost(X, Y, weight):
    num_points = X.shape[0]
    cost = (np.linalg.norm(np.dot(X, weight) - Y ,2) ** 2)/(2 * num_points)
    
    return cost
        

def wstar_cost(w, wstar):
    cost = (np.linalg.norm(w - wstar, 2))
    return cost

#%%
"""
The main function that includes all computations
"""
def SGC(X, Y, System_SGD, wSystem, dvec, System_EHD, System, num_workers, point_per_server, straggler_prob, straggler_tolerance, initial_weight, max_num_it = 1000, strag_it = 0, alpha = 0.5):
    
    """
    This function runs all approximate gradient descent algorithms discussed in the paper. In addition it runs centralized batch SGD to compare it with distributed algorithms
    
    INPUTS:
        X: data set
        Y: labels vector
        System_SGD: an array of length n, row i presents the identity of data vectors assigned to worker i for SGD
        w_System: an array of length n, row i presents the identity of data vectors assigned to worker i for SGC
        dvec: a vector containig the values of all the di's
        System_EHD: an array of length n, row i presents the identity of data vectors assigned to worker i for ErasureHead
        System: an array of length n, row i presents the identity of data vectors assigned to worker i for BGC
        num_worker: number of workers present in the system
        point_per_server: average replication factor d
        straggler_prob: the probability of workers being stragglers
        straggler_tolerance: a parameter for ErasureHead used to decide how many stragglers the algorithm can tolerate
        initial_weight: the vector used to initialize gradient descent algorithms
        max_num_it: number of iterations the algorithm will run for (default = 1000)
        alpha: step size (default 0.5)
        stragg_it: parameter to desing the number of iterations after which the identity of the stragglers change (default 0) 
        
        
    OUTPUTS:
        The value of the loss function at all iterations and the l_2 distance between the reached model and the best model for all considered algorithms
        
    """
    
    
    num_points      = X.shape[0]

    weight_SGC         = initial_weight
    weight_EHD         = initial_weight
    weight_BGC         = initial_weight
    weight_SGD_dedup   = initial_weight
    weight_SGD         = initial_weight
    weight_SGD_ind     = initial_weight      

    cost_SGC           = np.zeros((max_num_it,))
    cost_BGC           = np.zeros((max_num_it,))
    cost_EHD           = np.zeros((max_num_it,))
    cost_SGD_dedup     = np.zeros((max_num_it,))
    cost_SGD           = np.zeros((max_num_it,))
    cost_SGD_ind       = np.zeros((max_num_it,))
    
    cost_SGC_w           = np.zeros((max_num_it,))
    cost_BGC_w           = np.zeros((max_num_it,))
    cost_EHD_w           = np.zeros((max_num_it,))
    cost_SGD_w           = np.zeros((max_num_it,))
    cost_SGD_ind_w       = np.zeros((max_num_it,))
    cost_SGD_dedup_w     = np.zeros((max_num_it,))

    n               = num_workers               #% number of workers
    d               = point_per_server           #% average replication factor
    p               = straggler_prob          #% probability of workers being stragglers
    str_tol         = straggler_tolerance


    "Compute normalization factors for gradients"
    avgpoint = (1-p) * d  #% number of times each point is available using SGC
    avgpoint_SGD = (1-p) * n * System_SGD.shape[1]
    avgpoint_SGD_ind = min(d * (1-p), 1)
    avgpoint_SGD_dedup = 0
    
    for di in dvec:
        avgpoint_SGD_dedup += 1 - (p ** di)
        
    pts = int(round(avgpoint_SGD_dedup))
    weight0 = np.dot(np.linalg.pinv(X),Y)

   
    i = 0
    
    normX2 = np.linalg.norm( np.dot(np.transpose(X),X) ,2)
    #sigma = compute_sigma(X, d)

    while i < max_num_it:

        if (strag_it == 0) or (i % strag_it == 0):

            failure_pattern = np.random.binomial(1, p, n)
            servers_SGD = System_SGD[np.where(failure_pattern == 0)]
            servers_SGC = wSystem[np.where(failure_pattern == 0)]
            servers_BGC = System[np.where(failure_pattern == 0)]
            nbservers = n - failure_pattern.sum()  

        if nbservers > 0:
            
            if (strag_it == 0) or (i % strag_it == 0): # Check condition od dependency between stragglers
                
                "Get the points received at the master using SGC, SGD and BGC"
                Count_SGD = Counter(servers_SGD[0])
                Count_SGC = Counter(servers_SGC[0])
                Count_BGC = Counter(servers_BGC[0])
                for s in range(1,nbservers):
                    Count_SGD = Count_SGD + Counter(servers_SGD[s])
                    Count_SGC = Count_SGC + Counter(servers_SGC[s])
                    Count_BGC = Count_BGC + Counter(servers_BGC[s])
                    
                SGD_points = list(Count_SGD.keys())
                SGD_points.sort()
                
                SGC_points = list(Count_SGC.keys())
                SGC_points.sort()

                BGC_points = list(Count_BGC.keys())
                BGC_points.sort()
                
            "Get the points received at the master using ErasureHead"
            red = int(n / (str_tol + 1))
            
            EHD_points = []
            for group in range(red):
                for worker in range(group * (str_tol+1), (group + 1) * (str_tol + 1),1):
                    if failure_pattern[worker] == 0:
                        EHD_points = EHD_points + list(System_EHD[worker])
                        break
                    


            "Draw points uniformly at random to compare distributed algorithms with centralized SGD"
            SGD_points_ind = np.random.choice(X.shape[0], pts, replace = False)


                
            X_SGD = X[SGD_points]
            X_SGD_ind = X[SGD_points_ind]
            X_SGC = X[SGC_points]
            X_BGC = X[BGC_points]
            X_EHD = X[EHD_points]
            X_SGD_dedup = X[SGC_points]

            
            loss_SGD = np.dot(X_SGD, weight_SGD ) - Y[SGD_points]     
            loss_SGC = np.dot(X_SGC, weight_SGC ) - Y[SGC_points] 
            loss_BGC = np.dot(X_BGC, weight_BGC ) - Y[BGC_points] 
            loss_EHD = np.dot(X_EHD, weight_EHD ) - Y[EHD_points]  
            loss_SGD_ind = np.dot(X_SGD_ind, weight_SGD_ind ) - Y[SGD_points_ind]            
            loss_SGD_dedup = np.dot(X_SGD_dedup, weight_SGD_dedup ) - Y[SGC_points]
            
            
            alpha_SGD = compute_alpha(normX2, i)
            alpha_SGC = alpha_SGD 
            
            index1 = 0
            index2 = 0
            for j in range(num_points):
                
                if j in BGC_points:
                    X_BGC[index2] = Count_BGC[j] * X[j]
                    index2 +=1
                    
                
                if j in SGC_points:
                    "normalize data points by di"
                    dj = dvec[j]
                    X_SGC[index1] = X[j] * (Count_SGC[j] / dj)
                    
                    X_SGD_dedup[index1] = X[j] / (1 - (f ** dj))
                    
                    index1 +=1
                    
              
                
            """
            Compute gradients and update
            
            Note: replace alpha_SGD by alpha to use a constant step size taken as input
            """
            gradient_SGD = np.dot(np.transpose(X_SGD), loss_SGD) 
            weight_SGD  = weight_SGD - alpha_SGD * gradient_SGD / ( avgpoint_SGD )
            cost_SGD_w[i] = wstar_cost(weight_SGD,weight0)
            cost_SGD[i] = (L2_cost(X,Y,weight_SGD))


            gradient_SGD_ind = np.dot(np.transpose(X_SGD_ind),  loss_SGD_ind)
            weight_SGD_ind = weight_SGD_ind - (alpha_SGD * gradient_SGD_ind ) / (avgpoint_SGD_ind * num_points)
            cost_SGD_ind_w[i] = wstar_cost(weight_SGD_ind,weight0)
            cost_SGD_ind[i] = (L2_cost(X, Y, weight_SGD_ind))
            
            
            gradient_SGC = np.dot(np.transpose(X_SGC),  loss_SGC)
            weight_SGC = weight_SGC - (alpha_SGC * gradient_SGC) / ( (1-f) * num_points )
            cost_SGC_w[i] = wstar_cost(weight_SGC,weight0)
            cost_SGC[i] = (L2_cost(X, Y, weight_SGC))
            
            gradient_BGC = np.dot(np.transpose(X_BGC),  loss_BGC)
            weight_BGC = weight_BGC - (alpha_SGC * gradient_BGC) / ( avgpoint * num_points )
            cost_BGC_w[i] = wstar_cost(weight_BGC,weight0)
            cost_BGC[i] = (L2_cost(X, Y, weight_BGC))
            
            gradient_EHD = np.dot(np.transpose(X_EHD),  loss_EHD)
            """
            Un-comment the follwoing to use regularized gradient descent with constant step size,
            pick eta = 10 and alpha = 0.1 to have same parameters used by ErasureHead simulations
            """
            #weight_EHD = (1 - 2 * alpha * eta) * weight_EHD - (eta * gradient_EHD) / ( num_points ) 
            weight_EHD = weight_EHD - (alpha_SGC * gradient_EHD) / ( num_points )
            cost_EHD_w[i] = wstar_cost(weight_EHD, weight0)
            cost_EHD[i] = (L2_cost(X, Y, weight_EHD))
            
            
            gradient_SGD_dedup = np.dot(np.transpose(X_SGD_dedup),  loss_SGD_dedup)
            weight_SGD_dedup   = weight_SGD_dedup - (alpha_SGD * gradient_SGD_dedup ) / (num_points)
            cost_SGD_dedup_w[i] =  wstar_cost(weight_SGD_dedup,weight0)
            cost_SGD_dedup[i] =  (L2_cost(X, Y, weight_SGD_dedup))
            
            i+=1
    
        
    return cost_SGC, cost_SGD, cost_SGD_ind, cost_SGD_dedup,\
           cost_SGC_w, cost_SGD_w, cost_SGD_ind_w, cost_SGD_dedup_w, cost_BGC_w, cost_BGC, cost_EHD_w, cost_EHD



#%%
def GD(X, Y, initial_weight, max_num_it = 1000, alpha = 0.5):
    
        
    """
    This function runs Gradient Descent algorithm
    
    INPUTS:
        X: data set
        Y: labels vector
        initial_weight: the vector used to initialize gradient descent algorithms
        max_num_it: number of iterations the algorithm will run for (default = 1000)
        alpha: step size (default 0.5)
        
    OUTPUTS:
        The value of the loss function at all iterations and the l_2 distance between the reached model and the best model    
    """
    
    num_points      = X.shape[0]
    weight          = initial_weight    
    cost            = np.zeros((max_num_it,))
    cost_w          = np.zeros((max_num_it,))

    
    normX2 = np.linalg.norm( np.dot(np.transpose(X),X) ,2)
    weight0 = np.dot(np.linalg.pinv(X),Y)
    
    for i in range(max_num_it):

        alpha_var = compute_alpha(normX2, i)
        loss1 = np.dot(X, weight ) - Y   
        gradient = np.dot(np.transpose(X),loss1)
        weight = weight - alpha_var * gradient / num_points
        cost_w[i] = wstar_cost(weight,weight0)
        cost[i] = (L2_cost(X, Y, weight))
        


    return cost, cost_w
#%%
#### Data generation parameters 
noise_variance = 1
noise_mean = 0
datapoints = 1000
data_dim = 100


original_weight = np.random.randint(1,10,data_dim)
#%%
"""
Un-comment the following to generate the data with parameters defined above
"""
#X1, Y1 = genGauData(original_weight, noise_mean, noise_variance, datapoints)


#dfx1 = pd.DataFrame(X1)
#dfy1 = pd.DataFrame(Y1)

#dfx1.to_csv('X1-sig200-noisevar='+str(noise_variance)+',noisemean='+str(noise_mean)+',datapoints='+str(datapoints)+',dim='+str(data_dim)+'.csv')
#dfy1.to_csv('Y1-sig200-noisevar='+str(noise_variance)+',noisemean='+str(noise_mean)+',datapoints='+str(datapoints)+',dim='+str(data_dim)+'.csv')

#%%
"""
Read a pre-saved data set
"""
X1 = pd.DataFrame.from_csv('X1-sig100-noisevar='+str(noise_variance)+',noisemean='+str(noise_mean)+',datapoints='+str(datapoints)+',dim='+str(data_dim)+'.csv')
X1 = np.array(X1)
#
Y1 = pd.DataFrame.from_csv('Y1-sig100-noisevar='+str(noise_variance)+',noisemean='+str(noise_mean)+',datapoints='+str(datapoints)+',dim='+str(data_dim)+'.csv')
Y1 = np.array(Y1)
Y1 = Y1[:,0]

#%%
"""
Optimization parameters
"""
initial_weight = np.zeros(len(original_weight))
n = 100

"Change this value to create dependency between stragglers across iterations"
straggling_iteration = 0

"Here fail reflects the probability of workers being stragglers p"
fail = np.arange(0,1,0.1)


"Number of iterations"
itera = 5000

"The step size is proportional to (1/t)^(power)"
power = 0.7

"Number of times you want to repeat the experiment and average the results over"
average_over = 10

"Change this value and modify algorithms accordingly to have constant step size"
alpha = 0.1

iteration = list (range(itera))
#%%

"""
The main function where all variables are initialized and all functions are called
"""
darray=np.array([2])

            #print(sigma)
for d in darray:
    print('Starting code for n = '+str(n)+' and d = '+str(d)+', pow = '+str(power))
    avg_pps = datapoints * d / n
    sigma = compute_sigma(X1, d)
    
    sum_SGD = np.zeros(fail.shape[0])
    sum_SGC = np.zeros(fail.shape[0])
    sum_BGC = np.zeros(fail.shape[0])
    sum_EHD = np.zeros(fail.shape[0])
    sum_SGD_dedup  = np.zeros(fail.shape[0])
    sum_SGD_ind = np.zeros(fail.shape[0])
    sum_GD = np.zeros(fail.shape[0])
    
    sum_SGD_w = np.zeros(fail.shape[0])
    sum_SGC_w = np.zeros(fail.shape[0])
    sum_BGC_w = np.zeros(fail.shape[0])
    sum_EHD_w = np.zeros(fail.shape[0])
    sum_SGD_dedup_w  = np.zeros(fail.shape[0])
    sum_SGD_ind_w = np.zeros(fail.shape[0])
    sum_GD_w = np.zeros(fail.shape[0])

    
    cost_SGD = np.zeros(shape=(fail.shape[0],itera))
    cost_SGD_dedup = np.zeros(shape=(fail.shape[0],itera))
    cost_SGD_ind = np.zeros(shape=(fail.shape[0],itera))
    cost_SGC = np.zeros(shape=(fail.shape[0],itera))
    cost_BGC = np.zeros(shape=(fail.shape[0],itera))
    cost_EHD = np.zeros(shape=(fail.shape[0],itera))
    cost_GD = np.zeros(shape=(fail.shape[0],itera))
    
    cost_SGD_w = np.zeros(shape=(fail.shape[0],itera))
    cost_SGD_dedup_w = np.zeros(shape=(fail.shape[0],itera))
    cost_SGD_ind_w = np.zeros(shape=(fail.shape[0],itera))
    cost_SGC_w = np.zeros(shape=(fail.shape[0],itera))
    cost_BGC_w = np.zeros(shape=(fail.shape[0],itera))
    cost_EHD_w = np.zeros(shape=(fail.shape[0],itera))
    cost_GD_w = np.zeros(shape=(fail.shape[0],itera))
    

    
    dvec = np.zeros(datapoints)
    
    #% Data placement
    System_SGD = []
    System = []
    wSys = []
    nbpps = int(datapoints / n)
        
        
    for i in range(n):
        start = nbpps * i
        end = start + nbpps
        System_SGD.append(np.arange(start, end, 1))
        System.append([])
        wSys.append([])
                
    for i in range(datapoints):
        a = np.random.choice(n, d, replace = False)
        for j in a:
            System[j].append(i)
                
        di = compute_di(sigma, X1[i])
        dvec[i] = di
        #print(di)
        b = np.random.choice(n, di, replace = False)
        for j in b:
            wSys[j].append(i)
            
    print('dvec avg = ' + str(np.average(dvec)))
                
    System_SGD = np.array(System_SGD)
    System   = np.array(System)
    wSys   = np.array(wSys)
        
    s = d-1

    System_EHD = []
    red = int(n / (s+1))
    for variable in range(n):
        start = (variable // (s+1) ) * int(datapoints / red)
        end =   (variable // (s+1) + 1) * int(datapoints / red)
        System_EHD.append(np.arange(start, end, 1))
                
    System_EHD = np.array(System_EHD)
    
    
    for repeat_exp in range(average_over):
        
        ind = 0
        print('Experiment number ' +str(repeat_exp + 1) + ' out of '+str(average_over))
        

        
        for f in fail:
            print('Probability of failure f = '+str(f))

            
            cost_SGC1, cost_SGD1, cost_SGD_ind1, cost_SGD_dedup1, \
            cost_SGC1_w, cost_SGD1_w, cost_SGD_ind1_w, cost_SGD_dedup1_w, cost_BGC1_w, cost_BGC1, cost_EHD1_w, cost_EHD1 =\
            SGC(X1, Y1, System_SGD, wSys, dvec, System_EHD, System, n, d, f, s, initial_weight, itera, straggling_iteration, alpha)
    
            cost_SGD[ind]  += cost_SGD1
            cost_SGC[ind]  += cost_SGC1
            cost_BGC[ind]  += cost_BGC1
            cost_EHD[ind]  += cost_EHD1
            cost_SGD_ind[ind]  += cost_SGD_ind1
            cost_SGD_dedup[ind] += cost_SGD_dedup1
            
            
            sum_SGD[ind]       += cost_SGD1[-1]
            sum_SGD_ind[ind]   += cost_SGD_ind1[-1]
            sum_SGC[ind]       += cost_SGC1[-1]
            sum_BGC[ind]       += cost_BGC1[-1]
            sum_EHD[ind]       += cost_EHD1[-1]
            sum_SGD_dedup[ind]  += cost_SGD_dedup1[-1]
            
            cost_SGD_w[ind]  += cost_SGD1_w
            cost_SGC_w[ind]  += cost_SGC1_w
            cost_BGC_w[ind]  += cost_BGC1_w
            cost_EHD_w[ind]  += cost_EHD1_w
            cost_SGD_ind_w[ind]  += cost_SGD_ind1_w
            cost_SGD_dedup_w[ind] += cost_SGD_dedup1_w
            
            
            sum_SGD_w[ind]       += cost_SGD1_w[-1]
            sum_SGD_ind_w[ind]   += cost_SGD_ind1_w[-1]
            sum_SGC_w[ind]       += cost_SGC1_w[-1]
            sum_BGC_w[ind]       += cost_BGC1_w[-1]
            sum_EHD_w[ind]       += cost_EHD1_w[-1]
            sum_SGD_dedup_w[ind]  += cost_SGD_dedup1_w[-1]        
      
            ind += 1
            
        cost_GD1, cost_GD1_w = GD(X1, Y1, initial_weight, itera, alpha) 
        
        cost_GD[0] += cost_GD1
        sum_GD[0]  += cost_GD1[-1]   
        
        cost_GD_w[0] += cost_GD1_w
        sum_GD_w[0]  += cost_GD1_w[-1]
    
    
        
    print('Averaging results')
    
    ## Costs for L2 norm
    cost_SGD = cost_SGD/average_over
    cost_SGC = cost_SGC/average_over
    cost_BGC = cost_BGC/average_over
    cost_EHD = cost_EHD/average_over
    cost_SGD_ind = cost_SGD_ind/average_over
    cost_SGD_dedup = cost_SGD_dedup/average_over
    
    
    sum_SGD = sum_SGD/average_over
    sum_SGD_dedup = sum_SGD_dedup/average_over
    sum_SGD_ind = sum_SGD_ind/average_over
    sum_SGC = sum_SGC/average_over
    sum_BGC = sum_BGC/average_over
    sum_EHD = sum_EHD/average_over


    cost_GD[0] = cost_GD[0]/average_over
    sum_GD[0] = sum_GD[0]/average_over
    
    cost_GD = [cost_GD[0]]*fail.shape[0]
    sum_GD = [sum_GD[0]]*fail.shape[0]
    
    
    ## Costs for Wstar norm
    cost_SGD_w = cost_SGD_w/average_over
    cost_SGC_w = cost_SGC_w/average_over
    cost_BGC_w = cost_BGC_w/average_over
    cost_EHD_w = cost_EHD_w/average_over
    cost_SGD_ind_w = cost_SGD_ind_w/average_over
    cost_SGD_dedup_w = cost_SGD_dedup_w/average_over
    
    
    sum_SGD_w = sum_SGD_w/average_over
    sum_SGD_dedup_w = sum_SGD_dedup_w/average_over
    sum_SGD_ind_w = sum_SGD_ind_w/average_over
    sum_SGC_w = sum_SGC_w/average_over
    sum_BGC_w = sum_BGC_w/average_over
    sum_EHD_w = sum_EHD_w/average_over
    
    
    cost_GD_w[0] = cost_GD_w[0]/average_over
    sum_GD_w[0] = sum_GD_w[0]/average_over
    
    cost_GD_w = [cost_GD_w[0]]*fail.shape[0]
    sum_GD_w = [sum_GD_w[0]]*fail.shape[0]


    
    failures = list (fail)
    Savings = 100*(1-1/avg_pps)
    
    
    
    # Save the results in a CSV file
    from collections import OrderedDict
    DF1 = []
    DF2 = []
    DF1w = []
    DF2w = []

    DF1 = OrderedDict([('Iterations', np.repeat(itera, fail.shape[0])),\
                   ('f' , fail),\
                   ('SGC conv' , sum_SGC),\
                   ('BGC conv' , sum_BGC),\
                   ('EHD conv' , sum_EHD),\
                   ('SGD conv' , sum_SGD),\
                   ('SGD ind conv' , sum_SGD_ind),\
                   ('SGD dedup conv' , sum_SGD_dedup),\
                   ('GD conv' , sum_GD)]
            )
    
    DF1w = OrderedDict([('Iterations', np.repeat(itera, fail.shape[0])),\
                   ('f' , fail),\
                   ('SGC conv' , sum_SGC_w),\
                   ('BGC conv' , sum_BGC_w),\
                   ('EHD conv', sum_EHD_w),\
                   ('SGD conv' , sum_SGD_w),\
                   ('SGD ind conv' , sum_SGD_ind_w),\
                   ('SGD dedup conv' , sum_SGD_dedup_w),\
                   ('GD conv' , sum_GD_w)]
            )
    
    for index in range(len(fail)):
        DF2.append(cost_SGC[index])
        DF2.append(cost_BGC[index])
        DF2.append(cost_EHD[index])
        DF2.append(cost_SGD[index])   
        DF2.append(cost_SGD_ind[index])  
        DF2.append(cost_SGD_dedup[index])
        DF2.append(cost_GD[index])
        
        DF2w.append(cost_SGC_w[index])
        DF2w.append(cost_BGC_w[index])
        DF2w.append(cost_EHD_w[index])
        DF2w.append(cost_SGD_w[index])   
        DF2w.append(cost_SGD_ind_w[index])  
        DF2w.append(cost_SGD_dedup_w[index])
        DF2w.append(cost_GD_w[index])


    
    df1 = pd.DataFrame(DF1)
    df2 = pd.DataFrame(DF2)
    df1w = pd.DataFrame(DF1w)
    df2w = pd.DataFrame(DF2w)
    
    df1.to_csv('Figures/avg-over-'+str(average_over)+'-aaa-SGD-l2-varalpha-n='+str(n)+',d='+str(d)+',noise='+str(noise_variance)+',datapoints='+str(datapoints)+',dim='+str(data_dim)+',iter='+str(itera)+'str='+str(straggling_iteration)+'.csv', header = 1)
    df2.to_csv('Figures/avg-over-'+str(average_over)+'-bbb-SGD-l2-varalpha-n='+str(n)+',d='+str(d)+',noise='+str(noise_variance)+',datapoints='+str(datapoints)+',dim='+str(data_dim)+',iter='+str(itera)+'str='+str(straggling_iteration)+'.csv')

    df1w.to_csv('Figures/avg-over-'+str(average_over)+'-aaa-SGD-wstar-varalpha-n='+str(n)+',d='+str(d)+',noise='+str(noise_variance)+',datapoints='+str(datapoints)+',dim='+str(data_dim)+',iter='+str(itera)+'str='+str(straggling_iteration)+'.csv', header = 1)
    df2w.to_csv('Figures/avg-over-'+str(average_over)+'-bbb-SGD-wstar-varalpha-n='+str(n)+',d='+str(d)+',noise='+str(noise_variance)+',datapoints='+str(datapoints)+',dim='+str(data_dim)+',iter='+str(itera)+'str='+str(straggling_iteration)+'.csv')


#%%
"L2 Plots: Plot the value of the loss function as function of iterations and other parameters and save the graphs in a pdf"
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

pp = PdfPages('Figures/avg-over-'+str(average_over)+'-SGD-varalpha-l2-n='+str(n)+',d='+str(d)+',noise='+str(noise_variance)+',datapoints='+str(datapoints)+',dim='+str(data_dim)+',iter='+str(itera)+'str='+str(straggling_iteration)+'.pdf')

fig = plt.figure()
plt.semilogy(failures, sum_SGC,linestyle = '-', marker = 's', label='SGC')
plt.semilogy(failures, sum_BGC, marker = 's', label='BGC')
plt.semilogy(failures, sum_EHD,linestyle = '-.', marker = 's', label='EHD')
plt.semilogy(failures, sum_SGD,linestyle = '--', marker = 'o', label='SGD')
plt.semilogy(failures, sum_SGD_ind,linestyle = '--', marker = '^', label='SGD ind')
plt.semilogy(failures, sum_SGD_dedup,linestyle = '--', marker = '*', label='SGD dedup')
plt.semilogy(failures, sum_GD,linestyle = '-', label='GD')

plt.legend()
plt.xlabel('Failure rate')
plt.ylabel('Error')
plt.title('n = '+str(n)+', d = '+str(d)+', $\sigma^2$ = '+str(noise_variance)+', iter = '+str(itera)+', dim = '+str(data_dim)+', nb pts = '+str(datapoints)+', avg_pps = '+str(d*datapoints/n)+', Saving = '+str(Savings))
"Un-comment the following command to save values in TIKZ files"
#tikz_save('Figures/avg-over-'+str(average_over)+'convergence-l2-n='+str(n)+',d='+str(d)+',noise='+str(noise_variance)+',datapoints='+str(datapoints)+',dim='+str(data_dim)+',iter='+str(itera)+'pow='+str(power)+'str='+str(straggling_iteration)+'.tex', figureheight='\\figureheight', figurewidth='\\figurewidth')

plt.rcParams.update({'font.size': 8})# change font size
pp.savefig(fig)
plt.show()



nbiterations = itera

ind = 0
for f in fail:
    fig0 = plt.figure() # inches)
    plt.semilogy(iteration[:nbiterations], cost_SGC[ind][:nbiterations], label='SGC')
    plt.semilogy(iteration[:nbiterations], cost_BGC[ind][:nbiterations], label='BGC s = '+ str(s))
    plt.semilogy(iteration[:nbiterations], cost_EHD[ind][:nbiterations], label='EHD s = '+ str(s))
    plt.semilogy(iteration[:nbiterations], cost_SGD[ind][:nbiterations], label='SGD')
    plt.semilogy(iteration[:nbiterations], cost_SGD_ind[ind][:nbiterations], label='SGD ind')
    plt.semilogy(iteration[:nbiterations], cost_SGD_dedup[ind][:nbiterations], label='SGD dedup')
    plt.semilogy(iteration[:nbiterations], cost_GD[ind][:nbiterations], label='GD')
    
    plt.legend()
    plt.xlabel('Nb of iterations')
    plt.ylabel('Error')
    plt.title('N = '+str(n)+', d = '+str(d)+', f= '+str(f)+', $\sigma^2$ = '+str(noise_variance))

    # change font size
    plt.rcParams.update({'font.size': 8})
    pp.savefig(fig0)
    
    "Un-comment the following command to save values in TIKZ files"
    #tikz_save('Figures/avg-over-'+str(average_over)+'pb='+str(f)+'-l2-n='+str(n)+',d='+str(d)+',noise='+str(noise_variance)+',datapoints='+str(datapoints)+',dim='+str(data_dim)+',iter='+str(itera)+'str='+str(straggling_iteration)+'.tex', figureheight='\\figureheight', figurewidth='\\figurewidth')

    plt.show()
    ind += 1


pp.close()

#%%
"WSTAR plots: plot the value of the loss function as function of iterations and other parameters"
ppw = PdfPages('Figures/avg-over-'+str(average_over)+'-SGD-varalpha-wstar-n='+str(n)+',d='+str(d)+',noise='+str(noise_variance)+',datapoints='+str(datapoints)+',dim='+str(data_dim)+',iter='+str(itera)+'str='+str(straggling_iteration)+'.pdf')

fig = plt.figure()
plt.semilogy(failures, sum_SGC_w,linestyle = '-', marker = 's', label='SGC')
plt.semilogy(failures, sum_BGC_w, marker = 's', label='BGC')
plt.semilogy(failures, sum_EHD_w,linestyle = '-.', marker = 's', label='EHD')
plt.semilogy(failures, sum_SGD_w,linestyle = '--', marker = 'o', label='SGD')
plt.semilogy(failures, sum_SGD_ind_w,linestyle = '--', marker = '^', label='SGD ind')
plt.semilogy(failures, sum_SGD_dedup_w,linestyle = '--', marker = '*', label='SGD dedup')
plt.semilogy(failures, sum_GD_w,linestyle = '-', label='GD')

plt.legend()
plt.xlabel('Failure rate')
plt.ylabel('Error')
plt.title('ws: n = '+str(n)+', d = '+str(d)+', $\sigma^2$ = '+str(noise_variance)+', iter = '+str(itera)+', dim = '+str(data_dim)+', nb pts = '+str(datapoints)+', avg_pps = '+str(d*datapoints/n)+', Saving = '+str(Savings))
"Un-comment the following command to save values in TIKZ files"
#tikz_save('Figures/avg-over-'+str(average_over)+'convergence-wstar-n='+str(n)+',d='+str(d)+',noise='+str(noise_variance)+',datapoints='+str(datapoints)+',dim='+str(data_dim)+',iter='+str(itera)+'pow='+str(power)+'str='+str(straggling_iteration)+'.tex', figureheight='\\figureheight', figurewidth='\\figurewidth')
plt.rcParams.update({'font.size': 8}) # change font size
ppw.savefig(fig)
plt.show()




nbiterations = itera

ind = 0
for f in fail:
    fig0 = plt.figure()
    plt.semilogy(iteration[:nbiterations], cost_SGC_w[ind][:nbiterations], label='SGC')
    plt.semilogy(iteration[:nbiterations], cost_BGC_w[ind][:nbiterations], label='BGC s = '+ str(s))
    plt.semilogy(iteration[:nbiterations], cost_EHD_w[ind][:nbiterations], label='EHD s = '+ str(s))
    plt.semilogy(iteration[:nbiterations], cost_SGD_w[ind][:nbiterations], label='SGD')
    plt.semilogy(iteration[:nbiterations], cost_SGD_ind_w[ind][:nbiterations], label='SGD ind')
    plt.semilogy(iteration[:nbiterations], cost_SGD_dedup_w[ind][:nbiterations], label='SGD dedup')
    plt.semilogy(iteration[:nbiterations], cost_GD_w[ind][:nbiterations], label='GD')
    
    plt.legend()
    plt.xlabel('Nb of iterations')
    plt.ylabel('Error')
    plt.title('ws: N = '+str(n)+', d = '+str(d)+', f= '+str(f)+', $\sigma^2$ = '+str(noise_variance))

    # change font size
    plt.rcParams.update({'font.size': 8})
    ppw.savefig(fig0)
    "Un-comment the following command to save values in TIKZ files"
    #tikz_save('Figures/avg-over-'+str(average_over)+'pb='+str(f)+'-wstar-n='+str(n)+',d='+str(d)+',noise='+str(noise_variance)+',datapoints='+str(datapoints)+',dim='+str(data_dim)+',iter='+str(itera)+'str='+str(straggling_iteration)+'.tex', figureheight='\\figureheight', figurewidth='\\figurewidth')
    
    plt.show()
    ind += 1


ppw.close()
