import csv
import itertools
import time
from scipy.optimize import curve_fit
import math
import numpy as np
import matplotlib.pyplot as plt
import random
import statistics as st

def exhaustive_search(N):
    test_cities = cities[0:N]
    min_dt = -1
    min_route = []
    start_time = time.time()
    for route in itertools.permutations(test_cities):
        total_dt = sum([float(relative_dt[route[i]][route[i+1]]) for i in range(N - 1)]) + float(relative_dt[route[-1]][route[0]])
        if min_dt > total_dt or min_dt == -1:
            min_dt = total_dt
            min_route = route
    time_taken = time.time() - start_time
    return [min_dt, min_route, time_taken]


#itertools.permutation has time complexity O(n!) so i use the function f(x) = m(x!)
def eq(x, m):
    return [m*math.factorial(xn) for xn in x]

#Gives the total travel distance for a given route
def travel_distance(route):
    return sum([float(relative_dt[route[i]][route[i+1]]) for i in range(len(route) - 1)]) + float(relative_dt[route[-1]][route[0]])

def hill_climbing(N):
    best_route = cities[0:N]
    random.shuffle(best_route)
    best_score = travel_distance(best_route)
    while True:
        current = best_score
        for i in range(N):
            for j in range(N):
                if i != j:
                    test_route = best_route
                    i_val = best_route[i]
                    j_val = best_route[j]
                    test_route[i] = j_val
                    test_route[j] = i_val
                    test_score = travel_distance(test_route)
                    if test_score < best_score:
                        best_route = test_route
                        best_score = test_score
        if current == best_score:
            return [best_score, best_route]

def bench_alg(N, iterations):    
    best = hill_climbing(N)
    worst = best
    scores = []
    for i in range(iterations):
        res = hill_climbing(N)
        scores.append(res[0])
        if res[0] < best[0]:
            best = res
        elif res[0] > worst[0]:
            worst = res
    print("Testing for N=%i" %N)
    print("Best route distance: %f" %best[0])
    print("Worst route distance: %f" %worst[0])
    print("Mean route distance: %f" %(sum(scores)/len(scores)))
    print("Standard deviation of route distances: %f\n" %st.stdev(scores))


def init(N, pop_size): # N = number of cities included
    return [random.sample(cities[:N], N) for i in range(pop_size)]

#Gives the total travel distance for a given route
def travel_distance(route):
    return sum([float(relative_dt[route[i]][route[i+1]]) for i in range(len(route) - 1)]) + float(relative_dt[route[-1]][route[0]])

def ranking(routes):
    ranking = {i:travel_distance(routes[i]) for i in range(len(routes))} #key = list index, value = score/travel distance
    return sorted(ranking.items(), key=lambda x: x[1], reverse=False)
    
def mutate(route):
    if random.uniform(0.0, 1.0) <= mutate_prob:
        indx_1 = random.randint(0, len(route) - 1)
        indx_2 = indx_1
        while indx_2 == indx_1:
            indx_2 = random.randint(0, len(route) - 1)
        mutate = route; val_1 = route[indx_1]; val_2 = route[indx_2]
        mutate[indx_1] = val_2
        mutate[indx_2] = val_1
        return mutate
    return route
    
def crossover(parent_1, parent_2):
    #Partially mapped crossover(PMX)  
    sub_str_start = random.randint(0, (len(parent_1)//2) - 1)
    sub_str_end = sub_str_start + random.randint(1, len(parent_1)//2)
    sub_str = parent_1[sub_str_start : sub_str_end + 1]
    child = parent_2[:sub_str_start] + sub_str + parent_2[sub_str_end+1:]
    for i in range(len(parent_1)):
        if i not in list(range(sub_str_start, sub_str_end + 1)):
            for element in parent_2:
                if element not in child:
                    child[i] = element
    return child

def generate_children(routes, rank, c_num, p_num):
    parent_pool = [routes[key] for key,value in rank[:p_num]] #Im taking the best parents up to p_num
    children = []
    while len(children) < c_num:
        parent_1 = random.randint(0, len(parent_pool) - 1)
        parent_2 = random.choice([i for i in range(len(parent_pool)) if i != parent_1])
        child = crossover(parent_pool[parent_1], parent_pool[parent_2])
        child = mutate(child)
        children.append(child)
    return children

def next_evolution(routes, c_num, p_num, pop_size): 
    #c_num - how many children in next population
    #p_num - how many parents in next population
    global current_best, generation, current_avg
    generation += 1
    rank = ranking(routes)
    current_avg = sum([value for key, value in rank]) / len(rank)
    if rank[0][1] < current_best[0] or current_best[0] == -1:
        current_best = [rank[0][1], routes[rank[0][0]]]
    children = generate_children(routes, rank, c_num, pop_size//10)
    return [routes[key] for key, value in rank[:p_num]] + children

def bench_alg2(N, iterations, pop_size):
    global routes, current_best, current_avg, generation, runtime
    prev_best = []
    start_time = time.time()
    for i in range(iterations):
        routes = init(N, pop_size)
        current_best = [-1, None]
        generation = 0
        gen_averages = []
        #Termination condition is set to when standard deviation of last 10 averages is less than 40.0
        while len(gen_averages) < 20 or st.stdev(gen_averages[-10:]) > 40.0:
            routes = next_evolution(routes, int(pop_size*0.3), int(pop_size*0.7), pop_size)
            gen_averages.append(current_avg)
        prev_best.append(current_best)
    prev_best = sorted(prev_best, key=lambda x: x[0]) #Sort by distance low to high
    prev_best_dt = [route[0] for route in prev_best]
    best = prev_best[0]; worst = prev_best[-1]
    best_mean = sum(prev_best_dt)/len(prev_best_dt)
    std = 0
    if iterations > 1:
        std = st.stdev(prev_best_dt)
    print("\nTesting for N=%d, population size=%d and %d iterations\n" %(N, pop_size, iterations))
    print("The best/shortest distance: %f" %best[0])
    print("The worst/longest distance: %f" %worst[0])
    print("The mean distance: %f" %best_mean)
    print("The standard deviation: %f" %std)
    print("Total runtime: %fs" % (time.time() - start_time))
    return best, worst, best_mean, std, gen_averages #gen_averages is an average for every gen of last iteration


if __name__ == '__main__':
    #These are used for all my solutions later on
    cities = []
    relative_dt = {} #Look up table for relative distances
    with open("european_cities.csv", "r") as f:
        data = list(csv.reader(f, delimiter=';'))
        cities = data[0]
        data = data[1:]
        for i in range(len(data)):
            relative_dt[cities[i]] = {}
            for j in range(len(data)):
                relative_dt[cities[i]][cities[j]] = data[i][j]

    #Here i attempt to calculate an approximantion of n=24 using non-linear regression
    runtime = [exhaustive_search(n+1)[2] for n in range(10)] #runtime values from 0 > n <= 10
    x = np.array([i+1 for i in range(10)]) #x-values
    x_fit = np.array([i+1 for i in range(24)]) #x-values for best fit curve up to n=24

    #Using scipy.curve_fit to find optimal value for m
    m, cov = curve_fit(eq, x, runtime, p0=[1],maxfev=1000)
    curve = eq(x_fit, *m) #best fit curve y-values

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_ylabel('Runtime in seconds')
    ax1.set_xlabel('N destinations')
    ax1.set_title('Best fit curve for runtime / N')
    ax1.scatter(x, runtime, color='red', label="meassured values")
    ax1.plot(x_fit[:11], curve[:11], label="best fit curve")
    ax1.legend()
    print("\nEstimated runtime for n = 24: %i hours or %i years" %((curve[-1]/60/60), 
                                                                   (curve[-1]/60/60/24/365)))
    #Testing hill-climbing alg
    bench_alg(6, 20)
    bench_alg(24, 20)

    #parameters for EA solution
    mutate_prob = 0.3
    generation = 0
    routes = [] #population
    current_best = [-1, None] #current best route while algorithm is running [score, route]
    current_avg = 0 #current average distance of all routes in the population

    #Testing EA solution
    stats100 = bench_alg2(24, 20, 100)
    stats1000 = bench_alg2(24, 20, 1000)
    stats10000 = bench_alg2(24, 20, 10000)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_ylabel('Average distance')
    ax1.set_xlabel('generation N')
    ax1.set_title('average distance / generation nr. for N=24')
    ax1.plot([i for i in range(len(stats100[-1]))], stats100[-1], label='population=100')
    ax1.plot([i for i in range(len(stats1000[-1]))], stats1000[-1], color='green', label='population=1000')
    ax1.plot([i for i in range(len(stats10000[-1]))], stats10000[-1], color='red', label='population=10000')
    ax1.legend()
