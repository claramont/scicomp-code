import numpy as np
from scipy.sparse import lil_matrix, eye
from scipy.sparse.linalg import spsolve

class SphericalHeatCondSolver:
    def __init__(self, N, dt, Bi, omega, theta, t_end):
        self.N = N
        self.dt = dt
        self.Bi = Bi
        self.omega = omega
        self.theta = theta
        self.t_end = t_end

        self.dr = 1 / (self.N-1)
        self.r = np.linspace(0.0, 1.0, N)
        self.alpha = 1.0 - self.dr / 2.0

        self.A = self.build_matrix()
    
    def build_matrix(self):
        N = self.N
        dr = self.dr
        r = self.r
        Bi = self.Bi
        alpha = self.alpha


        # lil_matrix: fast at setting values row-by-row -> later convert to csr_matrix
        A = lil_matrix((N,N))

        # centre row r_1=0
        A[0,0] = -6.0 / (dr**2)
        A[0,1] = 6.0 / (dr**2)

        # interior rows j = 2, ..., N-1
        for i in range(1, N-1):
            rj = r[i]
            c_alpha = ((rj + dr/2.0)**2) / (rj**2)
            c_beta = ((rj - dr/2.0)**2) / (rj**2)

            A[i, i] = -(c_alpha + c_beta) / dr**2
            A[i, i-1] = c_beta / dr**2
            A[i, i+1] = c_alpha / dr**2

        # boundary
        A[N-1, N-2] = (3.0 *alpha**2) / (dr * (1.0 - alpha**3))
        A[N-1, N-1] = - 3.0 * (Bi * dr + alpha**2) / (dr * (1.0 - alpha**3))

        return A
    
    def b(self, t):
        b = np.zeros(self.N)

        b[-1] = 3 * self.Bi * np.sin(self.omega * t) * self.dr / (self.dr * (1.0 - self.alpha**3))
        return b

    
    def solve(self):
        N = self.N
        dt = self.dt
        theta = self.theta
        A = self.A

        nsteps = int(self.t_end / dt) # 0, 1, ..., Nsteps
        times = np.linspace(0.0, nsteps*dt, nsteps +1)

        U = np.zeros((nsteps+1, N))

        # initial condition: u(r_j, 0) = 0 for all j
        u = np.zeros(N)
        U[0,:] = u

        I = eye(N, format= "csr")

        M_left = I - dt * theta * A
        M_right = I + (1-theta) * dt * A

        for n in range(nsteps):
            tn= times[n]
            tn_next = times[n+1]
            bn = self.b(tn)
            bn_next = self.b(tn_next)

            rhs = (M_right @ u + dt * ((1-theta) * bn + theta* bn_next) )

            if theta == 0:
                u = rhs
            else:
                u = spsolve(M_left, rhs)

            U[n+1, :] = u

        return times, self.r, U 