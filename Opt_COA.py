import numpy  as np
import math
import random
import os

# Function
def target_function():
    return

# Function: Initialize Variables
def initial_position(swarm_size = 5, min_values = [-5,-5], max_values = [5,5], target_function = target_function):
    position = np.zeros((swarm_size, len(min_values) + 1))
    for i in range(0, swarm_size):
        for j in range(0, len(min_values)):
             position[i,j] = random.uniform(min_values[j], max_values[j])
        position[i,-1] = target_function(position[i,0:position.shape[1]-1])
    return position

# Function: Initialize Food Position
def food_position(dimension = 2, target_function = target_function):
    food = np.zeros((1, dimension+1))
    for j in range(0, dimension):
        food[0,j] = 0.0
    food[0,-1] = target_function(food[0,0:food.shape[1]-1])
    return food

# Function: Updtade Food Position by Fitness
def update_food(position, food):
    for i in range(0, position.shape[0]):
        if (food[0,-1] > position[i,-1]):
            for j in range(0, position.shape[1]):
                food[0,j] = position[i,j]
    return food

# Function: Updtade Position
def update_position(position, food, c1 = 1, min_values = [-5,-5], max_values = [5,5], target_function = target_function):
    for i in range(0, position.shape[0]):
        if (i <= position.shape[0]/2):
            for j in range (0, len(min_values)):
                c2 = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
                c3 = int.from_bytes(os.urandom(8), byteorder = "big") / ((1 << 64) - 1)
                if (c3 >= 0.5): #c3 < 0.5
                    position[i,j] = np.clip((food[0,j] + c1*((max_values[j] - min_values[j])*c2 + min_values[j])), min_values[j],max_values[j])
                else:
                    position[i,j] = np.clip((food[0,j] - c1*((max_values[j] - min_values[j])*c2 + min_values[j])), min_values[j],max_values[j])                       
        elif (i > position.shape[0]/2 and i < position.shape[0] + 1):
            for j in range (0, len(min_values)):
                position[i,j] = np.clip(((position[i - 1,j] + position[i,j])/2), min_values[j],max_values[j])             
        position[i,-1] = target_function(position[i,0:position.shape[1]-1])         
    return position

# COA Function
def coati_optimization_Algorithm(swarm_size = 5, min_values = [-5,-5], max_values = [5,5], iterations = 50, target_function = target_function):    
    count    = 0
    position = initial_position(swarm_size = swarm_size, min_values = min_values, max_values = max_values, target_function = target_function)
    food     = food_position(dimension = len(min_values), target_function = target_function)
    while (count <= iterations):
        c1       = 2*math.exp(-(4*(count/iterations))**2)
        food     = update_food(position, food)        
        position = update_position(position, food, c1 = c1, min_values = min_values, max_values = max_values, target_function = target_function)  
        count    = count + 1 
        
    return food

######################## Part 1 - Usage ####################################

# Function to be Minimized Solution ->  f(x1, x2) = -1.0316; x1 = 0.0898, x2 = -0.7126 or x1 = -0.0898, x2 = 0.7126
def coati(variables_values = [0, 0]):
    func_value = 4*variables_values[0]**2 - 2.1*variables_values[0]**4 + (1/3)*variables_values[0]**6 + variables_values[0]*variables_values[1] - 4*variables_values[1]**2 + 4*variables_values[1]**4
    return func_value

def rosenbrocks_valley(variables_values = [0,0]):
    func_value = 0
    last_x = variables_values[0]
    for i in range(1, len(variables_values)):
        func_value = func_value + (100 * math.pow((variables_values[i] - math.pow(last_x, 2)), 2)) + math.pow(1 - last_x, 2)
    return func_value
def optimizeweight():
    coa = coati_optimization_Algorithm(swarm_size = 15, min_values = [-5,-5], max_values = [5,5], iterations = 200, target_function = rosenbrocks_valley)
    return sum(coa[0])


