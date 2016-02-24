#!/usr/bin/env python

import sympy as sym

"""
Robot state
X = [x, y, L]'
"""
x_k, y_k, L = sym.symbols('x_k y_k L')
X_E = sym.Matrix([[x_k],[y_k],[L]])

"""
Robot inputs
U = [u, v]'
"""
u_k, v_k = sym.symbols('u_k v_k')
U_k = sym.Matrix([[u_k],[v_k]])

"""
Parameters and other variables:
z- measured distance between the robots
beta- Angle of the robot_y as seen by robot_x
"""
z = sym.symbols('z')
beta = sym.symbols('beta')


"""
    Exercise 5.

 Provide the analytical expressions of the three Jacobian matrices of the 
 previous three exteroceptive observations with respect to the new extended 
 state
"""
# ---- Write your function f here
f = sym.Matrix([[ 0 ], [ 0 ], [ 0 ]])

# ---- 

F_x = f.jacobian(X_E)
F_u = f.jacobian(U_k)


# ---- Write your h_z here
h_z = sym.Matrix([[ 0 ]])

# ---- Write your h_beta here
h_beta = sym.Matrix([[ 0 ]])

# ---- Write your h_GPS here
h_GPS = sym.Matrix([[ 0 ]])

# ----


H_z = h_z.jacobian(X_E)
H_beta = h_beta.jacobian(X_E)
H_GPS = h_GPS.jacobian(X_E)

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