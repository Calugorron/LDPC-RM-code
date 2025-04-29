import numpy as np
import scipy as sp
import math
from .utils import bin_wt, get_layout, flatten, int2bin, rank
import itertools as iter

class QuantumReedMuller:

    def __init__(self, rx, rz, n, distance=None, code_type=None, **kwargs):
        self._n = n
        self._N = 2**n
        self._rx = rx
        self._rz = rz
        list_for_k = list(math.comb(n, i) for i in range(rx+1, n-rz))
        self._k = sum(list_for_k)
        self._dx = 2**(rx+1)
        self._dz = 2**(rz+1) 
        if rx==rz:
            self._d = 2**(rx+1)
        else:
            self._d = None

        self._E, self._SX, self._SZ, self._LX, self._LZ = self._generate_code_properties()

        self._G = None
        self._GS = None
        self._aggregate_matrix = None

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
        if self._G == None:
            self._G = []
        if self._GS == None:
            self._GS = []


        #Function to generate operator from polynomial
        def gen_pattern(poly, check):
            pattern = list(range(2**self._n))
            for i, number in enumerate(pattern):
                    binary = int2bin(number, self._n)
                    if binary[poly[0]-1]==check[0] and binary[poly[1]-1]==check[1] and binary[poly[2]-1]==check[2]:
                        pattern[i]=1
                    else:
                        pattern[i]=0
            return pattern
        
        #Gauge logical
        saved_poly = []
        saved_check = []
        complementary_polynomials = []
        all_checks= list(int2bin(i, 3) for i in range(int(self._N**(1/2))))
        for poly in polynomials:
            for check in all_checks:
                possible_gauge=gen_pattern(poly, check)
                for j, L in enumerate(self._LZ):
                    if L.tolist()==possible_gauge:
                        complementary_polynomials.append(list(range(1, self._n+1)))
                        for x in poly:
                            complementary_polynomials[-1].remove(x)
                        self._G.append(L)
                        self._GS.append(L)
                        self._LZ.pop(j)
                        saved_poly.append(poly)
                        saved_check.append(check)
            if poly not in saved_poly:
                print(f"The polynomial {poly} doesn't result in a logical operator to gauge")
        #remove complementary logicals
        for poly in complementary_polynomials:
            for check in all_checks:
                possible_logical_to_remove=gen_pattern(poly, check)
                for j, L in enumerate(self._LZ):
                    if L.tolist()==possible_logical_to_remove:
                        self._LZ.pop(j)

        #Create group of gauge that spans stabilizer group + gauge group
        new_GS = []
        for i, poly in enumerate(polynomials):
            for j, check in enumerate(all_checks):
                if not (poly == saved_poly[i] and check == saved_check[i]):
                    new_gauge = gen_pattern(poly, check)
                    for gauge in self._G:
                        new_GS = self._GS[:]
                        new_GS.append(new_gauge)
                        if self.is_stabilizer(new_gauge+gauge) and rank(self._GS)+1==rank(new_GS):
                            self._GS.append(new_gauge)
                            break
        if rank(self._GS) == len(self._SZ)+len(self._G):
            print("All stabilizers have weight reduced from the gauges")

        #Create aggregate matrix to rebuild stabilizers from gauge
        if self._aggregate_matrix == None:
            self._aggregate_matrix = []
        for stab in self._SZ:
            weight_stab = sum(stab)
            weight_gauge = self._d
            for comb in iter.combinations(range(len(self._GS)), int(weight_stab/weight_gauge)):
                summed_gauge = [0]*len(stab)
                for element in comb:
                    summed_gauge = np.bitwise_xor(summed_gauge, self._GS[element])
                if not np.any(np.bitwise_xor(summed_gauge, stab)):
                    row = [0]*len(self._GS)
                    for element in comb:
                        row[element]=1
                    self._aggregate_matrix.append(row)
                    break


        

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
    
    @property
    def G(self):
        return self._G
    
    @property
    def GS(self):
        return self._GS
    
    @property
    def aggregate_matrix(self):
        return self._aggregate_matrix
    
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



