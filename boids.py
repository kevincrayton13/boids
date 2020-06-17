import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.spatial.distance import squareform, pdist, cdist
from numpy.linalg import norm

width, height = 640, 480

class Boids:
    #represents the simulation
    def __init__(self, N):
        #intial positions and velocities
        self.pos = [width / 2, height / 2] + 10 * np.random.rand(2 * N).reshape(N, 2)
        #normalized random velocities
        angles = 2 * math.pi * np.random.rand(N)
        self.vel = np.array(list(zip(np.sin(angles), np.cos(angles))))
        self.N = N
        #minimum distance of approach
        self.minDist = 25.0
        #max magnitude of velocity calculated by rules
        self.maxRuleVel = .03
        #max final magnitude of velocity
        self.maxVel = 2.0

    def tick(self, frameNum, pts, beak):
        #updates simulation
        #get pairwise distances
        self.distMatrix = squareform(pdist(self.pos))
        #apply the rules
        self.vel += self.applyRules()
        self.limit(self.vel, self.maxVel)
        self.pos += self.vel
        self.applyBC()
        #update data
        pts.set_data(self.pos.reshape(2 * self.N)[::2], self.pos.reshape(2 * self.N)[1::2])
        vec = self.pos + 10 * self.vel / self.maxVel
        beak.set_data(vec.reshape(2 * self.N)[::2], vec.reshape(2 * self.N)[1::2])

    def limitVec(self, vec, maxVal):
        #limit magnitude of the 2D vector
        mag = norm(vec)
        if mag > maxVal:
            vec[0], vec[1] = vec[0] * maxVal / mag, vec[1] * maxVal / mag

    def limit(self, X, maxVal):
        #limits magnitude of 2d vectors in array X to maxValue
        for vec in X:
            self.limitVec(vec, maxVal)

    def applyBC(self):
        #defines boundary conditions
        deltaR = 2.0
        for coord in self.pos:
            if coord[0] > width + deltaR:
                coord[0] = -deltaR
            if coord[0] < -deltaR:
                coord[0] = width + deltaR
            if coord[1] > height + deltaR:
                coord[1] = -deltaR
            if coord[1] < -deltaR:
                coord[1] = height + deltaR

    def applyRules(self):
        #rule number 1: seperation
        D = self.distMatrix < 25.0 #id's the rows that are too close
        vel = self.pos * D.sum(axis = 1).reshape(self.N, 1) - D.dot(self.pos)
        self.limit(vel, self.maxRuleVel)

        #distance threshold of alignment
        D = self.distMatrix < 50

        #apply rule 2: alignment
        vel2 = D.dot(self.vel)
        self.limit(vel2, self.maxRuleVel)
        vel += vel2

        #apply rule 3: cohesion
        vel3 = D.dot(self.pos) - self.pos
        self.limit(vel3, self.maxRuleVel)
        vel += vel3

        return vel

    def buttonPress(self, event):
        #event handler for button presses
        #guardian code if mouse is out of bounds
        if event.xdata == None or event.ydata == None: return
        #left click to add a boid
        if event.button == 1:
            self.pos = np.concatenate((self.pos, np.array([[event.xdata, event.ydata]])), axis = 0)
            #generate a random velocity
            angles = 2 * math.pi * np.random.rand(1)
            v = np.array(list(zip(np.sin(angles), np.cos(angles))))
            self.vel = np.concatenate((self.vel, v), axis = 0)
            self.N += 1

            #right click to scatter boids
        if event.button == 3:
            #adds scattering velocity
            self.vel += .1 * (self.pos - np.array([[event.xdata, event.ydata]]))

def tick(frameNum, pts, beak, boids):
    #print frameNum
    #update function for animation
    boids.tick(frameNum, pts, beak)
    return pts, beak

    #main function
def main():
    #set initial number of boids
    N = 100


    #create boids class
    boids = Boids(N)

    #set up the plot
    fig = plt.figure()
    ax = plt.axes(xlim = (0, width), ylim = (0, height))

    pts, = ax.plot([], [], markersize = 10, c = 'k', marker = 'o', ls = 'None')
    beak, = ax.plot([], [], markersize = 4, c = 'r', marker = 'o', ls = 'None')
    anim = animation.FuncAnimation(fig, tick, fargs = (pts, beak, boids), interval = 50)

    #adds buttonpress events handler
    cid = fig.canvas.mpl_connect('button_press_event', boids.buttonPress)
    ax.axes.tick_params(axis = 'both', color = '#FFFFFF', labelcolor = '#FFFFFF')
    plt.xlabel('Left click to add a boid, right click to create a disturbance')
    plt.title("Craig Reynold's Boids model")
    plt.show()

#call main
if __name__ == '__main__':
    main()
