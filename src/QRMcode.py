import numpy as np
import scipy as sp
from .utils import bin_wt

class QuantumReedMuller:

    def __init__(self, rx, rz, n, distance=None, code_type=None, **kwargs):
        self._n = n
        self._N = 2**n
        self._rx = rx
        self._rz = rz
        self._k = np.sum(list(sp.special.binom(n, i) for i in range(rx+1, n-rz)))
        self._dx = 2**(rx+1)
        self._dz = 2**(rz+1) 
        if rx==rz:
            self._d = 2**(rx+1)
        else:
            self._d = None

        self._E, self._SX, self._SZ, self._LX, self._LZ = self._generate_code_properties()

        self._G = None

    def _generate_code_properties(self):
        N=2**self.n
        F = np.array([[1,0],[1,1]])
        E = F
        for i in range(self.n-1):
            E = sp.linalg.kron(E, F)

        bin_wt = lambda i: bin(i)[2:].count('1')

        LX = []
        LZ = []
        SX = []
        SZ = []
        for row in range(N):
            if bin_wt(row) == self._rx:         #Something is wrong
                LX.append(E[row])
            if bin_wt(row) <= self._rx:         #Something is wrong
                SX.append(E[row])
            if bin_wt(row) == self._n-self._rz-1:
                LZ.append(E[row])
            if bin_wt(row) >= self._n-self._rz: 
                SZ.append(E[row])
        return E, SX, SZ, LX, LZ

    def gauge_operators(self, polynomials):
        all_checks= list(range(int(self._N**(1/2))))
        return None

    @property
    def N(self):
        return self._N
    
    @property
    def n(self):
        return self._n

    @property
    def k(self):
        return self.k

    @property
    def dz(self):
        return self._dz

    @property
    def dx(self):
        return self._dx
    
    @property
    def d(self):
        return self._d
    
    @property
    def E(self):
        return self._E
    

    @property
    def LX(self):
        return self._LX

    @property
    def LZ(self):
        return self._LZ

    @property
    def SX(self):
        return self._SX

    @property
    def SZ(self):
        return self._SZ
    
    def is_stabilizer(self, stabilizer):
        propagated_stabilizer = list(stabilizer)
        n=int(np.log2(len(stabilizer)))
        for i in reversed(range(n)):
            for j in reversed(range(2**(n-i-1))):
                for k in reversed(range(2**i)):
                    if propagated_stabilizer[k+j*2**(i+1)]==1:
                        propagated_stabilizer[2**i+k+j*2**(i+1)]=(propagated_stabilizer[2**i+k+j*2**(i+1)]+1)%2
        for position in range(64):
            if bin_wt(position)>self._rx and propagated_stabilizer[position]!=0:
                return False
        return True



