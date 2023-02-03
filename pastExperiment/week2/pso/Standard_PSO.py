# particle swarm optimization article - restarts PSO 
import random
import numpy as np
import math
import functions

# Hyper parameters
w = 0.729844  # Inertia weight to prevent velocities becoming too large
c1 = 2.05 * w  # Scaling co-efficient on the Social component
c2 = 2.05 * w  # Scaling co-efficient on the Cognitive component
dimension = 30  # Size of the problem
evalspPerDim = 10000
swarmsize = 50  # population size
maxevals = evalspPerDim * dimension
iteration = math.floor(maxevals / swarmsize)


# This class contains the code of the Particles in the swarm
class Particle:

    def __init__(self, fun_num, RangeF=100):
        # PSO parameters -- D. Bratton and J. Kennedy, "Defining a standard for particle swarm optimization,
        # " IEEE SIS, 2007, pp. 120â€“127.
        self.pos = np.random.uniform(low=-1 * RangeF, high=RangeF, size=dimension)
        self.pbest = self.pos
        self.velocity = np.array([
                                     0] * dimension)  # zero for initial velocity -- A. Engelbrecht, "Particle Swarm
        # Optimization: Velocity Initialization," IEEE CEC, 2012, pp. 70-77.
        self.Vmin, self.Vmax = -1 * RangeF, RangeF
        self.Xmin, self.Xmax = -1 * RangeF, RangeF
        self.f = fun_num
        self.function = functions.CEC_functions(30)
        self.val = self.eval(self.pos)
        self.pbest_val = self.val
        self.category = 0
        return

    # This method needs to be changed when changing objective function
    def eval(self, x):
        arr_x = np.array(x).astype(float)
        return self.function.Y(arr_x, self.f)

    def update_velocities(self, Xlbest):
        r1 = np.random.uniform(size=dimension)
        r2 = np.random.uniform(size=dimension)
        social = c1 * r1 * (Xlbest - self.pos)
        cognitive = c2 * r2 * (self.pbest - self.pos)
        self.velocity = (w * self.velocity) + social + cognitive
        self.velocity = np.clip(self.velocity, self.Vmin, self.Vmax)
        return

    def update_position(self):
        self.pos = self.pos + self.velocity
        for i in range(dimension):
            # clamp position: all positions need to be inside of the boundary, reflect them in cased they are out of
            # the boundary based on: S. Helwig, J. Branke, and S. Mostaghim, "Experimental Analysis of Bound Handling
            # Techniques in Particle Swarm Optimization," IEEE TEC: 17(2), 2013, pp. 259-271
            if self.pos[i] < self.Xmin:  # Less than lower bounds
                self.pos[i] = self.Xmin + (self.Xmin - self.pos[i])  # reflect
                self.velocity[i] = 0  # set velocity for reflected particles to zero
            if self.pos[i] > self.Xmax:  # Higher than upper bounds
                self.pos[i] = self.Xmax + (self.Xmax - self.pos[i])  # reflect
                self.velocity[i] = 0  # set velocity for reflected particles to zero
        self.val = self.eval(self.pos)
        return

    def update_pbest(self):
        # Update pbest
        if self.eval(self.pos) < self.pbest_val:
            self.pbest, self.pbest_val = self.pos, self.eval(self.pos)
        return


# This class contains the particle swarm optimization algorithm
class ParticleSwarmOptimizer:

    def __init__(self, fun_num):
        self.swarm = []
        self.fun_num = fun_num
        # Initial particles (position and velocity)
        for _ in range(swarmsize):
            p = Particle(fun_num)
            self.swarm.append(p)
        return

    def Lbest(self, pindex):
        adj_p = [self.swarm[(pindex - 1 + swarmsize) % swarmsize], self.swarm[pindex],
                 self.swarm[(pindex + 1) % swarmsize]]  # index wise adjacent particles
        adj_c = [adj_p[0].pbest_val, adj_p[1].pbest_val, adj_p[2].pbest_val]
        return adj_p[np.argmin(adj_c)]

    def Gbest(self):
        pbest_list = [p.pbest_val for p in self.swarm]
        gbest = np.min(pbest_list)
        gbest_pos = self.swarm[np.argmin(pbest_list)].pbest
        return gbest, gbest_pos

    def optimize(self):
        # gbest = []

        # Initialization of lbest for all particles
        lbest = []
        for j in range(swarmsize):
            lbest.append(self.Lbest(j).pbest)

        for ite in range(iteration):
            # Update of velocity and position
            for j, p in enumerate(self.swarm):
                p.update_velocities(lbest[j])
                p.update_position()
                p.update_pbest()

            # updating lbest
            lbest = []
            for j in range(swarmsize):
                p = self.Lbest(j)
                lbest.append(p.pbest)

            # Update gbest
            # gbest.append(self.Gbest())

        return self.Gbest()
