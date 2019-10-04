# -*- coding: utf-8 -*-
from numpy import linspace, sqrt, newaxis, zeros, exp
import matplotlib.pyplot as plt
import os
import time
from glob import glob
from mayavi.mlab import mesh, surf
import numpy as np
import sympy as sym
#from scitools.std import *


def scheme_ijn(u_1, u_2, q, b, dx, dy, dt,
               f, x, y, t_1, i, j, im1, ip1, jm1, jp1, I, V):
    """
    Right-hand side of finite difference at point [i,j].
    im1, ip1 denote i-1, i+1, resp. Similar for jm1, jp1.
    t_1 corresponds to u_1 (previous time level relative to u).

    """
#    u_1 = u[i, j, n]
#    u_2 = u[i, j, n-1]
    dt_2 = dt*dt
    dx_2 = dx*dx
    dy_2 = dy*dy

    res = (b*dt*dx_2*dy_2*u_2[i, j] + 2.0*dt_2*dx_2*dy_2*f(i, j, t_1) - 2.0*dt_2*dx_2*q(i, j - dy/2)*u_1[i, j] + 2.0*dt_2*dx_2*q(i, j - dy/2)*u_1[i, jm1] - 2.0*dt_2*dx_2*q(i, j + dy/2)*u_1[i, j] + 2.0*dt_2*dx_2*q(i, j + dy/2)*u_1[i, jp1] - 2.0*dt_2*dy_2*q(i - dx/2, j)*u_1[i, j] + 2.0*dt_2*dy_2*q(i - dx/2, j)*u_1[im1, j] - 2.0*dt_2*dy_2*q(i + dx/2, j)*u_1[i, j] + 2.0*dt_2*dy_2*q(i + dx/2, j)*u_1[ip1, j] + 4.0*dx_2*dy_2*u_1[i, j] - 2.0*dx_2*dy_2*u_2[i, j])/(dx_2*dy_2*(b*dt + 2.0))
    
    return res


def scheme_ij1(u_1, u_2, q, b, dx, dy, dt,
               f, x, y, t_1, i, j, im1, ip1, jm1, jp1, I, V):
    """
    Right-hand side of finite difference at point [i,j] for n=1.
    im1, ip1 denote i-1, i+1, resp. Similar for jm1, jp1.
    t_1 corresponds to u_1 (previous time level relative to u).

    """
#    u_1 = u[i, j, n]
    dt_2 = dt*dt
    dx_2 = dx*dx
    dy_2 = dy*dy
    res = (0.5*dt_2*dx_2*(-I(i, j)*q(i, j - dy/2) - I(i, j)*q(i, j + dy/2) + q(i, j - dy/2)*u_1[i, jm1] + q(i, j + dy/2)*u_1[i, jp1]) + 0.5*dt_2*dy_2*(-1.0*I(i, j)*q(i - dx/2, j) - I(i, j)*q(i + dx/2, j) + q(i - dx/2, j)*u_1[im1, j] + q(i + dx/2, j)*u_1[ip1, j]) + dx_2*dy_2*(-0.5*b*dt_2*V(i, j) + 0.5*dt_2*f(i, j, t_1) + 1.0*dt*I(i, j) + 1.0*I(i, j)))/(dx_2*dy_2)
    return res


def scheme_scalar_mesh(u, u_1, u_2, q, b, dx, dy, dt,
                       f, x, y, t_1, Nx, Ny, n, bc, I, V):
    if n == 1:
        scheme_ij = scheme_ij1
    elif n > 1:
        scheme_ij = scheme_ijn
    else:
        raise ValueError("n must be larger than 0!")

    Ix = range(0, u.shape[0])
    Iy = range(0, u.shape[1])

    # Interior points
    for i in Ix[1:-1]:
        for j in Iy[1:-1]:
            im1 = i-1; ip1 = i+1; jm1 = j-1; jp1 = j+1

            u[i, j] = scheme_ij(u_1, u_2, q, b, dx, dy, dt,
               f, x, y, t_1, i, j, im1, ip1, jm1, jp1, I, V)
    # Boundary points
    i = Ix[0]
    ip1 = i+1
    im1 = ip1
    if bc['W'] is None:
        for j in Iy[1:-1]:
            jm1 = j-1; jp1 = j+1
            u[i, j] = scheme_ij(u_1, u_2, q, b, dx, dy, dt,
               f, x, y, t_1, i, j, im1, ip1, jm1, jp1, I, V)
    else:
        for j in Iy[1:-1]:
            u[i, j] = bc['W'](x[i], y[j])
    i = Ix[-1]
    im1 = i-1
    ip1 = im1
    if bc['E'] is None:
        for j in Iy[1:-1]:
            jm1 = j-1; jp1 = j+1
            u[i, j] = scheme_ij(u_1, u_2, q, b, dx, dy, dt,
               f, x, y, t_1, i, j, im1, ip1, jm1, jp1, I, V)
    else:
        for j in Iy[1:-1]:
            u[i, j] = bc['E'](x[i], y[j])
    j = Iy[0]
    jp1 = j+1
    jm1 = jp1
    if bc['S'] is None:
        for i in Ix[1:-1]:
            im1 = i-1; ip1 = i+1
            u[i, j] = scheme_ij(u_1, u_2, q, b, dx, dy, dt,
               f, x, y, t_1, i, j, im1, ip1, jm1, jp1, I, V)
    else:
        for i in Ix[1:-1]:
            u[i, j] = bc['S'](x[i], y[j])
    j = Iy[-1]
    jm1 = j-1
    jp1 = jm1
    if bc['N'] is None:
        for i in Ix[1:-1]:
            im1 = i-1; ip1 = i+1
            u[i, j] = scheme_ij(u_1, u_2, q, b, dx, dy, dt,
               f, x, y, t_1, i, j, im1, ip1, jm1, jp1, I, V)
    else:
        for i in Ix[1:-1]:
            u[i, j] = bc['N'](x[i], y[j])

    # Corner points
    i = j = Iy[0]
    ip1 = i+1; jp1 = j+1
    im1 = ip1; jm1 = jp1
    if bc['S'] is None:
        u[i, j] = scheme_ij(u_1, u_2, q, b, dx, dy, dt,
               f, x, y, t_1, i, j, im1, ip1, jm1, jp1, I, V)
    else:
        u[i, j] = bc['S'](x[i], y[j])

    i = Ix[-1]; j = Iy[0]
    im1 = i-1; jp1 = j+1
    ip1 = im1; jm1 = jp1
    if bc['S'] is None:
        u[i, j] = scheme_ij(u_1, u_2, q, b, dx, dy, dt,
               f, x, y, t_1, i, j, im1, ip1, jm1, jp1, I, V)
    else:
        u[i, j] = bc['S'](x[i], y[j])

    i = Ix[-1]; j = Iy[-1]
    im1 = i-1; jm1 = j-1
    ip1 = im1; jp1 = jm1
    if bc['N'] is None:
        u[i, j] = scheme_ij(u_1, u_2, q, b, dx, dy, dt,
               f, x, y, t_1, i, j, im1, ip1, jm1, jp1, I, V)
    else:
        u[i, j] = bc['N'](x[i], y[j])

    i = Ix[0]; j = Iy[-1]
    ip1 = i+1; jm1 = j-1
    im1 = ip1; jp1 = jm1
    if bc['N'] is None:
        u[i, j] = scheme_ij(u_1, u_2, q, b, dx, dy, dt,
               f, x, y, t_1, i, j, im1, ip1, jm1, jp1, I, V)
    else:
        u[i, j] = bc['N'](x[i], y[j])

    return u


def scheme_vectorized_mesh(u, u_1, u_2, q, b, dx, dy, dt,
                       f, x, y, t_1, Nx, Ny, n, bc, I, V):
    # TODO:
    # - vx, vy?
    # -Ix, Iy
    if n == 1:
        scheme_ij = scheme_ij1
    elif n>1:
        scheme_ij = scheme_ijn
    else:
        raise ValueError("n must be larger than 0!")
    # Interior points
    i = slice(1, Nx)
    j = slice(1, Ny)
    im1 = slice(0, Nx-1)
    ip1 = slice(2, Nx+1)
    jm1 = slice(0, Ny-1)
    jp1 = slice(2, Ny+1)
    u[i, j] = scheme_ij(u_1, u_2, q, b, dx, dy, dt,
               f, x[i, :], y[i, :], t_1, i, j, im1, ip1, jm1, jp1, I, V)

    # Boundary points
    i = slice(1, Nx)
    ip1 = slice(2, Nx+1)
    im1 = ip1
    j = slice(1, Ny)
    jm1 = slice(0, Ny-1)
    jp1 = slice(2, Ny+1)
    if bc['W'] is None:
        u[i, j] = scheme_ij(u_1, u_2, q, b, dx, dy, dt,
               f, xv[i, :], yv[i, :], t_1, i, j, im1, ip1, jm1, jp1, I, V)
    else:
        u[i, j] = bc['W'](xv[i,:], yv[:,j])

    # The rest is not done yet.....
    i = Ix[-1]
    im1 = i-1
    ip1 = im1
    if bc['E'] is None:
        for j in Iy[1:-1]:
            jm1 = j-1; jp1 = j+1
            u[i, j] = scheme_ij(u_1, u_2, q, b, dx, dy, dt,
               f, x[i], y[i], t_1, i, j, im1, ip1, jm1, jp1, I, V)
    else:
        for j in Iy[1:-1]:
            u[i, j] = bc['E'](x[i], y[j])
    j = Iy[0]
    jp1 = j+1
    jm1 = jp1
    if bc['S'] is None:
        for i in Ix[1:-1]:
            im1 = i-1; ip1 = i+1
            u[i, j] = scheme_ij(u_1, u_2, q, b, dx, dy, dt,
               f, x[i], y[i], t_1, i, j, im1, ip1, jm1, jp1, I, V)
    else:
        for i in Ix[1:-1]:
            u[i, j] = bc['S'](x[i], y[j])
    j = Iy[-1]
    jm1 = j-1
    jp1 = jm1
    if bc['N'] is None:
        for i in Ix[1:-1]:
            im1 = i-1; ip1 = i+1
            u[i, j] = scheme_ij(u_1, u_2, q, b, dx, dy, dt,
               f, x[i], y[i], t_1, i, j, im1, ip1, jm1, jp1, I, V)
    else:
        for i in Ix[1:-1]:
            u[i, j] = bc['N'](x[i], y[j])

    # Corner points
    i = j = Iy[0]
    ip1 = i+1; jp1 = j+1
    im1 = ip1; jm1 = jp1
    if bc['S'] is None:
        u[i, j] = scheme_ij(u_1, u_2, q, b, dx, dy, dt,
               f, x[i], y[i], t_1, i, j, im1, ip1, jm1, jp1, I, V)
    else:
        u[i, j] = bc['S'](x[i], y[j])

    i = Ix[-1]; j = Iy[0]
    im1 = i-1; jp1 = j+1
    ip1 = im1; jm1 = jp1
    if bc['S'] is None:
        u[i, j] = scheme_ij(u_1, u_2, q, b, dx, dy, dt,
               f, x[i], y[i], t_1, i, j, im1, ip1, jm1, jp1, I, V)
    else:
        u[i, j] = bc['S'](x[i], y[j])

    i = Ix[-1]; j = Iy[-1]
    im1 = i-1; jm1 = j-1
    ip1 = im1; jp1 = jm1
    if bc['N'] is None:
        u[i, j] = scheme_ij(u_1, u_2, q, b, dx, dy, dt,
               f, x[i], y[i], t_1, i, j, im1, ip1, jm1, jp1, I, V)
    else:
        u[i, j] = bc['N'](x[i], y[j])

    i = Ix[0]; j = Iy[-1]
    ip1 = i+1; jm1 = j-1
    im1 = ip1; jp1 = jm1
    if bc['N'] is None:
        u[i, j] = scheme_ij(u_1, u_2, q, b, dx, dy, dt,
               f, x[i], y[i], t_1, i, j, im1, ip1, jm1, jp1, I, V)
    else:
        u[i, j] = bc['N'](x[i], y[j])

    return u


def solver(I, V, f, c, b, bc, Lx, Ly, Nx, Ny, dt, T,
           user_action=None, version='scalar',
           verbose=True):
    """
    Solve the 2D wave equation u_tt = u_xx + u_yy + f(x,t) on (0,L) with
    u = bc(x,y, t) on the boundary and initial condition du/dt = 0.

    Nx and Ny are the total number of grid cells in the x and y
    directions. The grid points are numbered as (0,0), (1,0), (2,0),
    ..., (Nx,0), (0,1), (1,1), ..., (Nx, Ny).

    dt is the time step. If dt<=0, an optimal time step is used.
    T is the stop time for the simulation.

    I, f, bc are functions: I(x,y), f(x,y,t), bc(x,y,t)

    user_action: function of (u, x, xv, y, yv, t, n) called at each time
    level (x and y are one-dimensional coordinate vectors).
    This function allows the calling code to plot the solution,
    compute errors, etc.

    verbose: true if a message at each time step is written,
    false implies no output during the simulation.
    """
    # TODO:
#    - find good c
#    - b should be an argument (?)
#    -

    x = linspace(0, Lx, Nx+1)  # mesh points in x dir
    y = linspace(0, Ly, Ny+1)  # mesh points in y dir
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    xv = x[:,newaxis]          # for vectorized function evaluations
    yv = y[newaxis,:]
    if dt <= 0:                # max time step?
        dt = (1/float(c))*(1/sqrt(1/dx**2 + 1/dy**2))
    Nt = int(round(T/float(dt)))
    t = linspace(0, T, Nt+1)   # mesh points in time
#    Cx2 = (c*dt/dx)**2;  Cy2 = (c*dt/dy)**2    # help variables
#    dt2 = dt**2

    q = c * 2
    u   = zeros((Nx+1, Ny+1))   # solution array
    u_1 = zeros((Nx+1, Ny+1))   # solution at t-dt
    u_2 = zeros((Nx+1, Ny+1))   # solution at t-2*dt

    Ix = range(0, Nx+1)
    Iy = range(0, Ny+1)
    It = range(0, Nt+1)

    # Load initial condition into u_1
    for i in Ix:
        for j in Iy:
            u_1[i, j] = I(x[i], y[j])

    if user_action is not None:
        user_action(u_1, x, xv, y, yv, t, 0)

    # Special formula for first time step
    if version == 'scalar':
        u = scheme_scalar_mesh(u, u_1, u_2, q, b, dx, dy, dt,
                       f, x, y, t[0], Nx, Ny, 1, bc, I, V)

    if user_action is not None:
        user_action(u, x, xv, y, yv, t, 1)

    u_2[:,:] = u_1; u_1[:,:] = u

    for n in It[1:-1]:
        if version == 'scalar':
            u = scheme_scalar_mesh(u, u_1, u_2, q, b, dx, dy, dt,
                                   f, x, y, t[n], Nx, Ny, n, bc, I, V)

        if user_action is not None:
            if user_action(u, x, xv, y, yv, t, n+1):
                break

        # Update data structures for next step
        #u_2[:] = u_1; u_1[:] = u  # slower
        u_2, u_1, u = u_1, u, u_2

    return dt  # dt might be computed in this function


def test_Gaussian(plot_u=1, version='scalar'):
    """
    Initial Gaussian bell in the middle of the domain.
    plot: not plot: 0; mesh: 1, surf: 2.
    """
    # Clean up plot files
    for name in glob('tmp_*.png'):
        os.remove(name)

    Lx = 10
    Ly = 10
    c = 1.0
    b = 1.0
    def I(x, y):
        return exp(-(x-Lx/2.0)**2/2.0 -(y-Ly/2.0)**2/2.0)
    def f(x, y, t):
        return 0.0
    def V(x, y):
        return 0.0

    def q(x, y):
        return 0.0
    
    bc = dict(N=None, W=None, E=None, S=None)

    def action(u, x, xv, y, yv, t, n):
        #print 'action, t=',t,'\nu=',u, '\Nx=',x, '\Ny=', y
        if t[n] == 0:
            time.sleep(2)
        if plot_u == 1:
            mesh(x, y, u, title='t=%g' % t[n], zlim=[-1,1],
                 caxis=[-1,1])
        elif plot_u == 2:
            surf(xv, yv, u, title='t=%g' % t[n], zlim=[-1, 1],
                 colorbar=True, colormap=hot(), caxis=[-1,1])
        if plot_u > 0:
            time.sleep(0) # pause between frames
            filename = 'tmp_%04d.png' % n
            #savefig(filename)  # time consuming...

    Nx = 40; Ny = 40; T = 15
    dt = solver(I, V, f, c, q, b, bc, Lx, Ly, Nx, Ny, 0, T,
                user_action=action, version='scalar')


def test_1D(plot=1, version='scalar'):
    """
    1D test problem with exact solution.
    """
    Lx = 10
    Ly = 10
    c = 1.0
    b = 1.0

    
    def I(x, y):
        """Plug profile as initial condition."""
        if abs(x-Lx/2.0) > 0.1:
            return 0
        else:
            return 1
        
    def f(x, y, t):
        """Return 0, but in vectorized mode it must be an array."""
        if isinstance(x, (float,int)):
            return 0
        else:
            return zeros(x.size)
    
    V = I
    def q(x, y):
        return 0
    

    bc=dict(N=None, E=None, S=None, W=None)


    def action(u, x, xv, y, yv, t, n):
        #print 'action, t=',t,'\nu=',u, '\Nx=',x, '\Ny=', y
        if plot:
            mesh(xv, yv, u, title='t=%g')
            time.sleep(0.2) # pause between frames

    Nx = 40; Ny = 40; T = 700
    dt = solver(I, V, f, c, q, b, bc,
                Lx, Ly, Nx, Ny, 0, T,
                user_action=action,
                version='scalar')


def test_const(plot=1, version='scalar'):
    """
    Test problem with constant solution.
    """
    Lx = 10
    Ly = 10
    c = 1.0
    C = 1.2
    b = 1
    def I(x,y):
        """Plug profile as initial condition."""
        return C
    def V(x, y):
        return 0
    
    def q(x, y):
        return 0

    def f(x, y, t):
        """Return 0, but in vectorized mode it must be an array."""
        if isinstance(x, (float,int)):
            return 0
        else:
            return zeros(x.size)

    u0 = lambda x, y, t=0: C
    bc=dict(N=u0, E=u0, S=u0, W=u0)

    def action(u, x, xv, y, yv, t, n):
        print(t)
        print(u)

    Nx = 4; Ny = 3; T = 5
    dt = solver(I, V, f, c, q, b, bc, Lx, Ly, Nx, Ny, 0, T, action, 'scalar')


if __name__ == '__main__':
    #test_Gaussian()
    # (Lx=1,Ly=1,Nx=4,Ny=4,dt=0.25,T=1,b=1,c=4,u_EXACT=5)
    #test_1D()
    #test_const()
#    import sys
#
#    if len(sys.argv) < 2:
#        print("""Usage [...] function arg1 arg2 arg3 ...""")
#        print(sys.argv[0])
#        sys.exit(0)
#    print("hi")
#    cmd = '%s(%s)' % (sys.argv[1], ', '.join(sys.argv[2:]))
#    print(cmd)
#    eval(cmd)