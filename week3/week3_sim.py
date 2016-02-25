#!/usr/bin/env python

import numpy as np
import scipy.integrate

from mayavi import mlab
import matplotlib.mlab
from tvtk.tools import visual
from tvtk.api import tvtk

# Get matrices from our script
from week3_12_3 import H_X, f_X_u, h_X, get_Fx, get_Fu

"""

    Parameters and configuration

"""
# Track distance
L = 2

# Noise on actuators for x and y robots
action_noise = np.array([[.001],[.001]])
# z noise for x and y robots
z_noise = .001
# GPS noise
GPS_noise = .05
# Beta acquisition noise
beta_noise = .02


"""

    System matrices

"""
# State prediction starts at 0. for both robots 
X_hat_k = np.array([[0.],[0.]])
# State variance at start
P_k = np.ones((X_hat_k.size,X_hat_k.size))*1000.

Q = np.array([[ .0001 , 0],[ 0, .0001]])
# Measurements: z, beta, GPS
r_z = .001
r_beta = .25
r_gps = .5
R = np.array([[ r_z , 0, 0],[ 0, r_beta, 0.], [0, 0, r_gps]])

Fx = get_Fx()
Fu = get_Fu()



"""

    Simulates system:

    dot_x = [u v]'

"""
def simulate(t, state, (u,v)):    
    dxdt = np.zeros_like(state)
    
    dxdt[0] = u
    dxdt[1] = v

    return dxdt
    

    
"""

    Simulation

"""
dt = 0.01
t_max = 10.
X_init = np.array([[0.],[1.]])
u_cmd = 0.
v_cmd = 0.

odey = scipy.integrate.ode(simulate).set_integrator('dopri5')
odey.set_initial_value(X_init, 0.0)

t = np.zeros(np.ceil(t_max/dt)+1)
x = np.zeros((np.ceil(t_max/dt)+1,X_hat_k.size,1))
x_hat = np.zeros((np.ceil(t_max/dt)+1,X_hat_k.size,1))
P = np.zeros((np.ceil(t_max/dt)+1,X_hat_k.size,X_hat_k.size))
z = np.zeros((np.ceil(t_max/dt)+1,3,1))
z_pred = np.zeros((np.ceil(t_max/dt)+1,3,1))

idx = 0
while odey.successful() and odey.t < t_max:   
    """ Control the robots here """
    if odey.t > 1.:
        u_cmd = 1.
        v_cmd = .6
    else:
        u_cmd = 0.
        v_cmd = 0.
        
        
    """ Robot y kidnapping """
    if odey.t > 4 and odey.t < 4.2:
        odey.y[1] = 4.

        
    # time step simulation
    odey.set_f_params((u_cmd, v_cmd))
    odey.integrate(odey.t+dt)
    
    """ Your Extended Kalman Filter here """    
    # Action measurement
    U_k = np.array([[u_cmd*dt], [v_cmd*dt]]) + np.random.normal(0, action_noise, (2,1))
    
    # Noisy exteropeceptive measurements
    z_meas = np.sqrt((odey.y[1,0]-odey.y[0,0])**2 + L**2) + np.random.normal(0, z_noise, 1)
    beta_meas = np.arctan2(L,odey.y[1,0]-odey.y[0,0]) + np.random.normal(0, beta_noise, 1)
    GPS_meas = odey.y[0,0] + np.random.normal(0, GPS_noise, 1)
    
    Z_k = np.array([z_meas,beta_meas,GPS_meas])    
    
    # Predict/Action step    
    X_hat_k_prior = f_X_u(X_hat_k, U_k )
    P_k_prior = Fx.dot(P_k).dot(Fx.T) + Fu.dot(Q).dot(Fu.T)

    # Update/Perception step
    H = H_X(X_hat_k_prior, L)
    X_hat_k = X_hat_k_prior + P_k_prior.dot(H.T).dot( np.linalg.inv(H.dot(P_k_prior).dot(H.T) + R) ).dot(Z_k - h_X(X_hat_k_prior, L))
    P_k = P_k_prior - P_k_prior.dot(H.T).dot(np.linalg.inv( H.dot(P_k_prior).dot(H.T) + R)).dot(H).dot(P_k_prior)

    # Save data for animation
    t[idx] = odey.t
    x[idx,:] = odey.y    
    x_hat[idx,:] = X_hat_k
    z[idx,:] = Z_k
    z_pred[idx,:] = h_X(X_hat_k_prior, L)
    P[idx,:] = P_k
    idx += 1
    
    
"""

    Plots

"""

import matplotlib.pyplot as plt

# Plot states and predicted states
fig = plt.figure()
plt.subplot(211)
plt.errorbar(t, x[:,0], yerr=P[:,0,0])
plt.plot(t, x_hat[:,0]);
plt.legend(['x_real','x_hat']);
plt.xlim([0, t_max])
plt.title('State, x and y')

plt.subplot(212)
plt.errorbar(t, x[:,1], yerr=P[:,1,1])
plt.plot(t, x_hat[:,1]);
plt.legend(['y_real','y_hat']);
plt.xlim([0, t_max])

# Plot sensor readings and predicted sensor readings
fig = plt.figure()
plt.subplot(311)
plt.plot(t, z[:,0], t, z_pred[:,0]);
plt.legend(['z meas','z pred']);
plt.xlim([0, t_max])

plt.subplot(312)
plt.plot(t, z[:,1], t, z_pred[:,1]);
plt.legend(['beta meas','beta pred']);
plt.xlim([0, t_max])


plt.subplot(313)
plt.plot(t, z[:,2], t, z_pred[:,2], t,x[:,0]);
plt.legend(['x meas','x pred', 'x real']);
plt.xlim([0, t_max])



plt.figure()
plt.subplot(311)
plt.plot(t, x[:,1]);
plt.legend(['y real']);
plt.xlim([0, t_max])

plt.subplot(312)
plt.plot(t, P[:,0,0],t, P[:,0,1]);
plt.xlim([0, t_max])

plt.subplot(313)
plt.plot(t, P[:,1,0],t, P[:,1,1]);
plt.xlim([0, t_max])


plt.show()

"""

    Animation

"""
# Create a figure
f = mlab.figure(size=(1000,1000))
# Tell visual to use this as the viewer.
visual.set_viewer(f)
robot_radius = .15
robot_height = robot_radius/5.0

# Elements in animation
robot_x_axis = (0, 0, 1.0)
robot_x_pos = (x[0,0,0], -L/2., 0)
robot_x = visual.cylinder(pos=robot_x_pos, axis=robot_x_axis,radius=robot_radius, length=robot_height, color=(0.0,1.0,0.0))

robot_y_axis = (0, 0, 1.0)
robot_y_pos = (x[0,1,0], L/2., 0)
robot_y = visual.cylinder(pos=robot_y_pos, axis=robot_y_axis,radius=robot_radius, length=robot_height, color=(1.0,0.0,0.0))

vtext = tvtk.VectorText()
vtext.text = str(t[0]) + ' s'
text_mapper = tvtk.PolyDataMapper(input=vtext.get_output())
p2 = tvtk.Property(color=(0.3, 0.3, 0.3))
text_actor = tvtk.Follower(mapper=text_mapper, property=p2)
text_actor.position = (0, -L/4., 0)
f.scene.add_actor(text_actor)

# start line
mlab.plot3d([0, 0],[-L/2., L/2.],[0, 0],figure=f)
# tracks
mlab.plot3d([0, 10.],[-L/2., -L/2.],[0, 0],figure=f, line_width=1)
mlab.plot3d([0, 10.],[L/2., L/2.],[0, 0],figure=f, line_width=1)

# distance between robots
line = tvtk.LineSource(point1=robot_x_pos, point2=robot_y_pos)
line_mapper = tvtk.PolyDataMapper(input=line.output)
line_actor = tvtk.Actor(mapper=line_mapper)
f.scene.add_actor(line_actor)

# Location of the robot
delta = 0.1
surf_length = 5.
surf_width = .5

X_x, Y_x = np.meshgrid(np.arange(x_hat[0,0,0]-surf_length, x_hat[0,0,0]+surf_length, delta), np.arange(robot_x.pos[1]-surf_width, robot_x.pos[1]+surf_width, delta))
Z_x = matplotlib.mlab.bivariate_normal(X_x.T, Y_x.T, P[0,0,0], .1, x_hat[0,0,0], robot_x.pos[1])
surface_x = mlab.surf(X_x.T, Y_x.T, Z_x, representation='wireframe')
ms_x = surface_x.mlab_source

X_y, Y_y = np.meshgrid(np.arange(x_hat[0,1,0]-surf_length, x_hat[0,1,0]+surf_length, delta), np.arange(robot_y.pos[1]-surf_width, robot_y.pos[1]+surf_width, delta))
Z_y = matplotlib.mlab.bivariate_normal(X_y.T, Y_y.T, P[0,1,1], .1, x_hat[0,1,0], robot_y.pos[1])
surface_y = mlab.surf(X_y.T, Y_y.T, Z_y, representation='wireframe')
ms_y = surface_y.mlab_source

@mlab.show
@mlab.animate(delay=10)
def anim():
    idx = 0
    idx_steps = x.shape[0]
    while 1:
        robot_x.pos = (x[idx,0,0], -L/2., robot_height)
        robot_y.pos = (x[idx,1,0], L/2., robot_height)
                
        vtext.text = str(t[idx]) + ' s'
        
        line.point1 = robot_x.pos
        line.point2 = robot_y.pos
        
        X_x, Y_x = np.meshgrid(np.arange(x_hat[idx,0,0]-surf_length, x_hat[idx,0,0]+surf_length, delta), np.arange(robot_x.pos[1]-surf_width, robot_x.pos[1]+surf_width, delta))
        Z_x = matplotlib.mlab.bivariate_normal(X_x.T, Y_x.T, P[idx,0,0], .1, x_hat[idx,0,0], robot_x.pos[1])
        ms_x.set(x=X_x.T, y=Y_x.T, s=Z_x)
        
        X_y, Y_y = np.meshgrid(np.arange(x_hat[idx,1,0]-surf_length, x_hat[idx,1,0]+surf_length, delta), np.arange(robot_y.pos[1]-surf_width, robot_y.pos[1]+surf_width, delta))
        Z_y = matplotlib.mlab.bivariate_normal(X_y.T, Y_y.T, P[idx,1,1], .1, x_hat[idx,1,0], robot_y.pos[1])
        ms_y.set(x=X_y.T, y=Y_y.T, s=Z_y)
        

        idx += 1
        if idx > idx_steps-5:
            idx = 0
        yield

# Run the animation.
anim()