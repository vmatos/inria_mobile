#!/usr/bin/env python

import numpy as np
import sympy as sym

"""
Robot state
X = [x, y]'
"""
x_k, y_k = sym.symbols('x_k y_k')
X_k = sym.Matrix([[x_k],[y_k]])

"""
Robot inputs
U = [u, v]'
"""
u_k, v_k = sym.symbols('u_k v_k')
U_k = sym.Matrix([[u_k],[v_k]])

"""
Parameters and other variables:
L- distance between lines
z- measured distance between the robots
beta- Angle of the robot_y as seen by robot_x
"""
L = sym.symbols('L')
z = sym.symbols('z')
beta = sym.symbols('beta')


"""
    Exercise 1.

 Provide the analytical expression of the function f, which characterizes the 
link between the state and the proprioceptive measurements 
(i.e., Xk+1=f(Xk, uk))
"""
# ---- Write your function f here
f = sym.Matrix([[ 0 ], [ 0 ]])

# ---- 

F_x = f.jacobian(X_k)
F_u = f.jacobian(U_k)


"""
    Exercise 2.

 Provide the analytical expression of the function h that characterizes the link
 between the state and the range measurements (i.e., z=h(X)) 
"""
# ---- Write your h_z here
h_z = sym.Matrix([[ 0 ]])

# ----

H_z = h_z.jacobian(X_k)


"""
    Exercise 3.

 Provide the analytical expression of the function h_beta that characterizes 
 the link between the state and the bearing measurements (i.e., beta=h_beta(X))
"""
# ---- Write your h_beta here
h_beta = sym.Matrix([[ 0 ]])

# ---- 

H_beta = h_beta.jacobian(X_k)


"""
    Exercise 4.

 Provide the analytical expression of the function hGPS that characterizes the 
 link between the state and the GPS measurements
"""
# ---- Write your h_GPS here
h_GPS = sym.Matrix([[ 0 ]])

# ---- 

H_GPS = h_GPS.jacobian(X_k)

"""
Print results
"""
sym.init_printing()

sym.pprint("\nh_z:")
sym.pprint(h_z, use_unicode=True)

sym.pprint("\nh_beta:", use_unicode=True)
sym.pprint(h_beta, use_unicode=True)

sym.pprint("\nh_GPS:", use_unicode=True)
sym.pprint(h_GPS, use_unicode=True)

sym.pprint("\nH_z:")
sym.pprint(H_z, use_unicode=True)

sym.pprint("\nH_beta:", use_unicode=True)
sym.pprint(H_beta, use_unicode=True)

sym.pprint("\nH_GPS:", use_unicode=True)
sym.pprint(H_GPS, use_unicode=True)

h = h_z.col_join(h_beta).col_join(h_GPS)
H = H_z.col_join(H_beta).col_join(H_GPS)

""" Get matrices """
def H_X(X_k_1, L_track):
    return np.array( H.subs([(x_k,X_k_1[0,0]),(y_k,X_k_1[1,0]),(L,L_track), ('pi',np.pi)]) )
    
def f_X_u(X_k_1, u):
    return np.array( f.subs([(x_k,X_k_1[0,0]), (y_k,X_k_1[1,0]), (u_k,u[0,0]),(v_k, u[1,0])])) 
    
def get_Fx():
    return np.array( F_x ) 
    
def get_Fu():
    return np.array( F_u ) 
    
def h_X(X_k, L_track):
    return np.array( h.subs([(x_k,X_k[0,0]),(y_k,X_k[1,0]),(L,L_track), ('pi',np.pi)]) )