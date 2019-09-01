import sympy as sym
import numpy as np
V, t, I, w, dt = sym.symbols('V t I w dt')  # global symbols
a, b = sym.symbols('a b') # For subparts d, and e
f = None  # global variable for the source term in the ODE

def ode_source_term(u):
    """Return the terms in the ODE that the source term
    must balance, here u'' + w**2*u.
    u is symbolic Python function of t."""
    return sym.diff(u(t), t, t) + w**2*u(t)

def residual_discrete_eq(u):
    """Return the residual of the discrete eq. with u inserted."""
    R = DtDt(u, dt) + w**2*u(t) - f
    return sym.simplify(R)

def residual_discrete_eq_step1(u):
    """Return the residual of the discrete eq. at the first
    step with u inserted."""
    #R = (f(t_0) - \omega^2 I) \frac{\Delta t^2}{2} + V\Delta t + I - u_1
    R = 0.5*(f.subs(t, 0) - w**2 * I)*dt**2 + V*dt + I - u(t + dt)
    R = R.subs(t, 0)
    return sym.simplify(R)

def DtDt(u, dt):
    """Return 2nd-order finite difference for u_tt.
    u is a symbolic Python function of t.
    """
    return (u(t - dt) - 2*u(t) + u(t + dt))/(dt**2)

def main(u):
    """
    Given some chosen solution u (as a function of t, implemented
    as a Python function), use the method of manufactured solutions
    to compute the source term f, and check if u also solves
    the discrete equations.
    """
    print ('=== Testing exact solution: %s ===' % u)
    print ("Initial conditions u(0)=%s, u'(0)=%s:" % (u(t).subs(t, 0), sym.diff(u(t), t).subs(t, 0)))

    # Method of manufactured solution requires fitting f
    global f  # source term in the ODE
    f = sym.simplify(ode_source_term(u))

    # Residual in discrete equations (should be 0)
    print ('residual step1:', residual_discrete_eq_step1(u))
    print ('residual:', residual_discrete_eq(u))

def linear():
    main(lambda t: V*t + I)

def quadratic():
    main(lambda t: b*t**2 + V*t + I)

def poly_d3():
    main(lambda t: a*t**3 + b*t**2 + V*t + I)


def solver(I, V, w, f, dt, T):
    """
    Solve u'' + w**2*u = f(t) for t in (0,T],
    u(0)=I and u'(0)=V

    This function is based on the solver function in the lecture for IN5270
    """
    dt = float(dt)  # avoid integer div.
    Nt = int(round(T / dt))
    T = Nt * dt
    u = np.zeros(Nt + 1)
    t = np.linspace(0, T, Nt + 1)

    u[0] = I    # the initial condition
    u[1] = 0.5*(f(t[0]) - w**2 * I)*dt**2 + V*dt + I
    
    for n in range(1, Nt):
        u[n+1] = (f(t[n]) - w**2 *u[n]) * dt**2 + 2*u[n] - u[n-1]
    return u, t

def nose_test():
    global I, V, w, b
    u_e = lambda t: b * t ** 2 + V * t + I  # quadratic
        
    I = 1.0; V = 1.0; w = 1.0; b = 2.0 # initial values

    global f, t
    f = sym.simplify(ode_source_term(u_e))
    f = sym.lambdify (t, f)
    
    dt = 1e-2
    T = dt * 10
    u, t = solver(I, V, w, f, dt, T)
    
    err = np.abs(u - u_e(t))
    
    print(f"maximum error: {np.max(err)}, T = {T}, dt={dt}")


 

if __name__ == '__main__':
    linear()
    quadratic()
    poly_d3()
    nose_test()
