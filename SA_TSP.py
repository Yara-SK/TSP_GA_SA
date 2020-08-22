import random
import math
import copy
import numpy
import matplotlib.pyplot as plt
from math import sqrt
from scipy.spatial import distance


data = [        [10,10], [15,30], [45,77],  [20,50],  [60,50], 
                [60,15], [90,25], [90,79],  [54,18],  [21,10],
                [63,91], [16,71], [26,29],  [58,39],  [33,15],
                [33,55], [80,90], [100,79], [100,59], [22,80],
                [33,81], [48,88], [50,77],  [90,76],  [20,56],
                [80,60], [30,5],  [20,10],  [0,30],   [100,80],
                [11,54], [11,90], [11,18],  [80,43],  [43,81],
                [76,67], [57,50], [99,21],  [99,88],  [5,31]
        ]


########################### CREATE INPUT ################################

# FUNCTION THAT CREATES RANDOM SEQUENCE
def random_sequence():
    ran_seq = []
    r = 0

    while len(ran_seq) < 40:  #creating a list of integers <39 in random order
        r = random.randint(1,40)
        if r not in ran_seq:
            ran_seq.append(r)
            
    return ran_seq    


# FUNCTION THAT USES RANDOM SEQUENCE TO BUILD A COORDINATE LIST
def coords_indx_res():
    
    coords_indx = []
    count1 = 0
    count2 = 0

    restriction_1 = [5,10,15,20,25,30,35,40]     # only visit 3 of these
    restriction_2 = [2,4,6,8,10,20,22,32,33,35]  # only visit 5 of these
    
    ran_seq = random_sequence()    
    
    for i in ran_seq: #setting up restrictions
       
       #if a coordinate is in both restrictions
        if i in restriction_1 and i in restriction_2:
            if count1 != 3 and count2 != 5:
                coords_indx.append(i)
                count1 += 1
                count2 += 1

        #if a coordinate is in restriction_1    
        elif i in restriction_1:
            if count1 != 3:
                coords_indx.append(i)
                count1 += 1

        #if a coordinate is in restriction_2        
        elif i in restriction_2:
            if count2 != 5:
                coords_indx.append(i)
                count2 += 1
                
        else:
            coords_indx.append(i)
    
    return(ran_seq, coords_indx, len(coords_indx))


# OUTPUTTING INDICIES OF THE CHOSEN COORDINATES
ran_seq, coords_indx, lc = coords_indx_res()
print(ran_seq, "\nLength is: ", len(ran_seq))
print(coords_indx, "\nLength is: ", len(coords_indx))


# LOOK AT THE INDICIES IN THE coords_indx, AND EXTRACT CORRESPONDING COORDINATES 
def extract(coords_list_in, coords_indx):
    
    # first add the depot coordinate
    coords_list = [(0,0)]
    coords_indx_sort = sorted(coords_indx)
    
    # outputing coordinates
    for i in coords_indx_sort:
        coords_list.append(coords_list_in[i-1])

    #finish at the depot coordinate
    coords_list.append([0,0])
        
    return coords_list, coords_indx_sort

# OUTPUTTING COORDINATES TO coords_list
coords_list, sorty = extract(data, coords_indx)



############################## SIMULATED ANNEALING #######################################

# DEFINING THE ALGORITHM
def TSP_SA(data,n):
    
    customers = data.copy()
    #choosing random route
    route = random.sample(range(n),n)

    #THE ALGORITHM
    #setting up temperature, 200K in this case 
    for temperature in numpy.logspace(0,5,num=200000)[::-1]:
        #make a new route by randomly swapping two customers in 'route'
        [i,j] = sorted(random.sample(range(n),2))
        new_Route =  route[:i] + route[j:j+1] +  route[i+1:j] + route[i:i+1] + route[j+1:]
        
        #fourmulas
        oldDistances =  sum([ math.sqrt(sum([(customers[route[(k+1) % n]][d] - customers[route[k % n]][d])**2 for d in [0,1] ])) for k in [j,j-1,i,i-1]])
        newDistances =  sum([ math.sqrt(sum([(customers[new_Route[(k+1) % n]][d] - customers[new_Route[k % n]][d])**2 for d in [0,1] ])) for k in [j,j-1,i,i-1]])
        
        #if if condition is satisfied, old route will be replace by new route
        if math.exp( ( oldDistances - newDistances) / temperature) > random.random():
            route = copy.copy(new_Route)
            
    
    #setting up parameters to plot the result
    xs = [customers[route[i % n]][0] for i in range(n+1)]
    ys = [customers[route[i % n]][1] for i in range(n+1)]
    
    total_dist = 0
    for k in range(len(xs)-1):
        total_dist += sqrt((xs[k] - xs[k+1])**2 + (ys[k] - ys[k+1])**2)
    
    plt.plot(xs, ys, 'ob-')
    plt.show()
    
    print("Total Distance: ", total_dist)
    print(customers)


TSP_SA(coords_list,len(coords_list))