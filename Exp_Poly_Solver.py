import numpy as np
from scipy.sparse import lil_matrix, eye
from scipy.linalg import expm
from scipy.sparse.linalg import spsolve

class SphericalExpRKSolver:
    def __init__(self, N, dt, Bi, omega, t_end):
        self.N = N
        self.dt = dt
        self.Bi = Bi
        self.omega = omega
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
        A = self.A.toarray()

        nsteps = int(self.t_end / dt) # 0, 1, ..., Nsteps
        times = np.linspace(0.0, nsteps*dt, nsteps +1)

        U = np.zeros((nsteps+1, N))

        # initial condition: u(r_j, 0) = 0 for all j
        u = np.zeros(N)
        U[0,:] = u

        H = dt * A
        I = np.eye(N)

        # E = exp(hA)
        E = expm(H)

        for n in range(nsteps):
            tn = times[n]
            tn_next = times[n+1]

            k1 = self.b(tn)
            k2 = self.b(tn_next)
            
            # compute r1, r2
            r1 = (E - I) @ k1
            r2 = (E - I - H) @ (k2-k1)

            # s1: H @ s1  = r1
            s1 = np.linalg.solve(H, r1)

            # s2: H @ q = r2, H@s2 = q
            q = np.linalg.solve(H, r2)
            s2 = np.linalg.solve(H, q)

            u = E @ u + dt * (s1 + s2)
            U[n+1, :] = u
        
        return times, self.r, U


