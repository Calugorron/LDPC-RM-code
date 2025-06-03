import numpy as np
import scipy as sp
import math
from .utils import bin_wt, get_layout, flatten, int2bin, rank, distribute, show_group
import itertools as iter
from collections import deque

class QuantumReedMuller:

    def __init__(self, rx, rz, m, distance=None, code_type=None, **kwargs):
        self._m = m
        self._N = 2**m
        self._rx = rx
        self._rz = rz
        list_for_k = list(math.comb(m, i) for i in range(rx+1, m-rz))
        self._k = sum(list_for_k)
        self._dx = 2**(rx+1)
        self._dz = 2**(rz+1)
        if rx==rz:
            self._d = 2**(rx+1)
        else:
            self._d = None

        self._E, self._SX, self._SZ, self._LX, self._LZ = self._generate_code_properties()

        self._GX = None
        self._GZ = None
        self._GSX = None
        self._GSZ = None
        self._aggregate_matrix_X = None
        self._aggregate_matrix_Z = None
        #self._extra_aggregate_matrix_X = None
        #self._extra_aggregate_matrix_Z = None
        #self._overchecked_SX = None
        #self._overchecked_SZ = None


    def _generate_code_properties(self):
        N=2**self._m
        F = np.array([[1,0],[1,1]])
        E = F
        for i in range(self._m-1):
            E = sp.linalg.kron(E, F)
        bin_wt = lambda i: bin(i)[2:].count('1')
        ET=np.transpose(E)
        LX = []
        LZ = []
        SX = []
        SZ = []
        for row in range(N):
            
            if bin_wt(N-row-1) == self._rx+1:
                LX.append(E[row])
            if bin_wt(N-row-1) <= self._rx:         #Something is wrong
                SX.append(E[row])
            if bin_wt(N-row-1) == self._m-self._rz-1:
                LZ.append(ET[row])
            if bin_wt(N-row-1) >= self._m-self._rz:
                SZ.append(ET[row])
        return E, SX, SZ, LX, LZ
    

    def gauge_operators(self, polynomials, all=True, generators_size=None, aggregate_method="simple"):
        if self._GX == None:
            self._GX = []
        if self._GSX == None:
            self._GSX = []
        if self._GZ == None:
            self._GZ = []
        if self._GSZ == None:
            self._GSZ = []


        #Function to generate operator from polynomial
        def gen_pattern(poly, check):
            pattern = list(range(2**self._m))
            for i, number in enumerate(pattern):
                    binary = int2bin(number, self._m)
                    if binary[poly[0]-1]==check[0] and binary[poly[1]-1]==check[1] and binary[poly[2]-1]==check[2]:
                        pattern[i]=1
                    else:
                        pattern[i]=0
            return pattern
        
        #Gauge logical
        print("Gauging X logical")
        saved_poly = []
        saved_check = []
        complementary_polynomials = []
        all_checks= list(int2bin(i, 3) for i in range(int(self._N**(1/2))))
        for poly in polynomials:
            for check in all_checks:
                possible_gauge=gen_pattern(poly, check)
                for j, L in enumerate(self._LX):
                    if L.tolist()==possible_gauge:
                        complementary_polynomials.append(list(range(1, self._m+1)))
                        for x in poly:
                            complementary_polynomials[-1].remove(x)
                        self._GX.append(L)
                        self._LX.pop(j)
                        saved_poly.append(poly)
                        saved_check.append(check)
            if poly not in saved_poly:
                print(f"The polynomial {poly} doesn't result in a logical operator to gauge")

        print("Removing complementary logicals")
        for poly in complementary_polynomials:
            for check in all_checks:
                possible_logical_to_remove=gen_pattern(poly, check)
                for j, L in enumerate(self._LX):
                    if L.tolist()==possible_logical_to_remove:
                        self._LX.pop(j)

        print("Creating group of gauge that spans stabilizer group + gauge group")
        if all:
            for poly in polynomials:
                for check in all_checks:
                    possible_gauge=gen_pattern(poly, check)
                    for gauge in self._GX:
                        if self.is_stabilizer(gauge+possible_gauge) or not np.any(possible_gauge-gauge):
                            self._GSX.append(possible_gauge)
                            break
        else:
            self._GSX = self._GX[:]
            new_GSX = []
            for i, poly in enumerate(reversed(polynomials)):
                for j, check in enumerate(reversed(all_checks)):
                    if not (poly == saved_poly[i] and check == saved_check[i]):
                        new_gauge = gen_pattern(poly, check)
                        for gauge in self._GX:
                            new_GSX = self._GSX[:]
                            new_GSX.append(new_gauge)
                            if self.is_stabilizer(new_gauge+gauge) and rank(self._GSX)+1==rank(new_GSX):
                                self._GSX.append(new_gauge)
                                break
                            else:
                                new_GSX.pop(-1)
            if rank(self._GSX) == len(self._SX)+len(self._GX):
                print("All stabilizers have weight reduced from the gauges")

            #print("Adding the remaining gauge in order to create a generator group of size generators_size")
            #def add_remaining_gauge(poly, call):
            #    added = False
            #    print(call)
            #    for check in reversed(all_checks):
            #        if not (poly == saved_poly[i] and check == saved_check[i]):
            #            new_gauge = gen_pattern(poly, check)
            #            for gauge in self._GSX:
            #                if self.is_stabilizer(new_gauge+gauge) and np.any(new_gauge+gauge):
            #                    self._GSX.append(new_gauge)
            #                    added=True
            #                    break
            #            if added:
            #                added=False
            #                break
            #            
            #    if len(self._GSX) < generators_size and call < len(poly)*len(all_checks):
            #        add_remaining_gauge(polynomials[(call+1)%len(polynomials)], call+1)
            #    elif call == len(poly)*len(all_checks):
            #        print("Couldn't reach the desired number of gauge operators")
            #    else:
            #        print(f"Reached {generators_size} gauge for the generators")
#
            #add_remaining_gauge(polynomials[0], 0)

        print("Creating aggregate matrix to rebuild stabilizers from gauge")
        if self._aggregate_matrix_X == None:
            self._aggregate_matrix_X = []
        gauges = deque(list(range(len(self._GSX))))
        for i, stab in enumerate(self._SX):
            print(f"Stab number {i}")
            weight_stab = sum(stab)
            weight_gauge = self._d
            for comb in iter.combinations(gauges, int(weight_stab/weight_gauge)):
                summed_gauge = [0]*len(stab)
                for element in comb:
                    summed_gauge = np.bitwise_xor(summed_gauge, self._GSX[element])
                if not np.any(np.bitwise_xor(summed_gauge, stab)):
                    row = [0]*len(self._GSX)
                    for element in comb:
                        row[element]=1
                    self._aggregate_matrix_X.append(row)
                    gauges.rotate(1)
                    break
        #if aggregate_method == "extra" and all:
        #    for i, row in enumerate(np.transpose(self._aggregate_matrix_X)):
        #        if not np.any(row):
        #            print(f"    Gauge not contributing in the aggregate matrix {i}")
        #            for j, gauge in enumerate(self._GSX):
        #                if self.is_stabilizer(np.bitwise_xor(gauge, self._GSX[i])) and np.any(np.bitwise_xor(gauge, self._GSX)):
        #                    row = [0]*len(self._GSX)
        #                    row[i]=1
        #                    row[j]=1
        #                    self._aggregate_matrix_X.append(row)
        #                    break
        #elif aggregate_method == "complete" and all:
        #    stabilizers = deque(self._SX)
        #    self._overchecked_SX = []
        #    self._extra_aggregate_matrix_X = []
        #    for i, row in enumerate(np.transpose(self._aggregate_matrix_X)):
        #        if not np.any(row):
        #            print(f"    Gauge not contributing in the aggregate matrix {i}")
        #            stop = False
        #            gauge = self._GSX[i]
        #            gauges = list(range(len(self._GSX)))
        #            gauges.pop(i)
        #            for num_elements in range(1, 9):
        #                print(f"    Number of elements {num_elements}")
        #                for comb in iter.combinations(gauges, num_elements):
        #                    summed_gauge = gauge[:]
        #                    for element in comb:
        #                        summed_gauge = np.bitwise_xor(summed_gauge, self._GSX[element])
        #                    for j, stab in enumerate(stabilizers):
        #                        if not np.any(np.bitwise_xor(summed_gauge, stab)) and np.any(np.bitwise_xor(gauge, self._GSX)):
        #                            row = [0]*len(self._GSX)
        #                            row[i]=1
        #                            for element in comb:
        #                                row[element]=1
        #                            self._extra_aggregate_matrix_X.append(row)
        #                            self._overchecked_SX.append(j)
        #                            stabilizers.rotate(1)
        #                            stop = True
        #                            break
        #                    if stop:
        #                        break
        #                if stop:
        #                    break
                
            

        
        #inverse_aggregrate = []
        #elements_of_the_generators = []
        #for i, gauge in enumerate(self._GSX):
        #    stop = False
        #    print(f"Gauge number {i}")
        #    for num_elements in range(1, 9):
        #        print(f"number of elements {num_elements}")
        #        for comb in iter.combinations(range(len(self._GX)+len(self._SX)), num_elements):
        #            summed_gauge = [0]*len(gauge)
        #            for element in comb:
        #                if element < len(self.SX):
        #                    summed_gauge = np.bitwise_xor(summed_gauge, self._SX[element])
        #                else:
        #                    summed_gauge = np.bitwise_xor(summed_gauge, self._GX[element-len(self.SX)])
        #            if not np.any(np.bitwise_xor(summed_gauge, gauge)):
        #                row = [0]*(len(self._GX)+len(self._SX))
        #                for element in comb:
        #                    row[element]=1
        #                print(row)
        #                if len(comb) == 1:
        #                    elements_of_the_generators.append(element)
        #                    print(f"generator {element}")
        #                inverse_aggregrate.append(row)
        #                stop = True
        #                break
        #        if stop:
        #            break
        #print(inverse_aggregrate)
        #self._aggregate_matrix_X = np.linalg.pinv(inverse_aggregrate)
        #print(self._aggregate_matrix_X)
        #print(self._aggregate_matrix_X%2)

            



        #####################################################3Perfomr the same for Z
        print("Gauging Z logical")
        saved_poly = []
        saved_check = []
        complementary_polynomials = []
        all_checks= list(int2bin(i, 3) for i in range(int(self._N**(1/2))))
        for poly in polynomials:
            for check in all_checks:
                possible_gauge=gen_pattern(poly, check)
                for j, L in enumerate(self._LZ):
                    if L.tolist()==possible_gauge:
                        complementary_polynomials.append(list(range(1, self._m+1)))
                        for x in poly:
                            complementary_polynomials[-1].remove(x)
                        self._GZ.append(L)
                        self._LZ.pop(j)
                        saved_poly.append(poly)
                        saved_check.append(check)
            if poly not in saved_poly:
                print(f"The polynomial {poly} doesn't result in a logical operator to gauge")

        print("Removing complementary logicals")
        for poly in complementary_polynomials:
            for check in all_checks:
                possible_logical_to_remove=gen_pattern(poly, check)
                for j, L in enumerate(self._LZ):
                    if L.tolist()==possible_logical_to_remove:
                        self._LZ.pop(j)

        print("Creating group of gauge that spans stabilizer group + gauge group")

        if all:
            for poly in polynomials:
                for check in all_checks:
                    possible_gauge=gen_pattern(poly, check)
                    for gauge in self._GZ:
                        if self.is_stabilizer(gauge+possible_gauge) or not np.any(possible_gauge-gauge):
                            self._GSZ.append(possible_gauge)
                            break                
        else:
            self._GSZ = self._GZ[:]
            new_GSZ = []
            for i, poly in enumerate(reversed(polynomials)):
                for j, check in enumerate(reversed(all_checks)):
                    if not (poly == saved_poly[i] and check == saved_check[i]):
                        new_gauge = gen_pattern(poly, check)
                        for gauge in self._GZ:
                            new_GSZ = self._GSZ[:]
                            new_GSZ.append(new_gauge)
                            if self.is_stabilizer(new_gauge+gauge) and rank(self._GSZ)+1==rank(new_GSZ):
                                self._GSZ.append(new_gauge)
                                break
                            else:
                                new_GSZ.pop(-1)
            if rank(self._GSZ) == len(self._SZ)+len(self._GZ):
                print("All stabilizers have weight reduced from the gauges")
            #print("Adding the remaining gauge in order to create a generator group of size generators_size")
            #def add_remaining_gauge(poly, call):
            #    added = False
            #    print(call)
            #    for check in reversed(all_checks):
            #        if not (poly == saved_poly[i] and check == saved_check[i]):
            #            new_gauge = gen_pattern(poly, check)
            #            for gauge in self._GSZ:
            #                if self.is_stabilizer(new_gauge+gauge) and np.any(new_gauge+gauge):
            #                    self._GSZ.append(new_gauge)
            #                    added=True
            #                    break
            #            if added:
            #                added=False
            #                break      
            #    if len(self._GSZ) < generators_size and call < len(poly)*len(all_checks):
            #        add_remaining_gauge(polynomials[(call+1)%len(polynomials)], call+1)
            #    elif call == len(poly)*len(all_checks):
            #        print("Couldn't reach the desired number of gauge operators")
            #    else:
            #        print(f"Reached {generators_size} gauge for the generators")
            #add_remaining_gauge(polynomials[0], 0)

        #print("Creating aggregate matrix to rebuild stabilizers from gauge")
        #if self._aggregate_matrix_Z == None:
        #    self._aggregate_matrix_Z = []
        #gauges = deque(reversed(list(range(len(self._GSZ)))))
        #for i, stab in enumerate(self._SZ):
        #    print(f"Stab number {i}")
        #    weight_stab = sum(stab)
        #    weight_gauge = self._d
        #    for comb in iter.combinations(gauges, int(weight_stab/weight_gauge)):
        #        summed_gauge = [0]*len(stab)
        #        for element in comb:
        #            summed_gauge = np.bitwise_xor(summed_gauge, self._GSZ[element])
        #        if not np.any(np.bitwise_xor(summed_gauge, stab)):
        #            row = [0]*len(self._GSZ)
        #            for element in comb:
        #                row[element]=1
        #            self._aggregate_matrix_Z.append(row)
        #            gauges.rotate(1)
        #            break
        #if aggregate_method == "extra" and all:
        #    for i, row in enumerate(np.transpose(self._aggregate_matrix_Z)):
        #        if not np.any(row):
        #            print(f"    Gauge not contributing in the aggregate matrix {i}")
        #            for j, gauge in enumerate(self._GSZ):
        #                if self.is_stabilizer(np.bitwise_xor(gauge, self._GSZ[i])) and np.any(np.bitwise_xor(gauge, self._GSZ)):
        #                    row = [0]*len(self._GSZ)
        #                    row[i]=1
        #                    row[j]=1
        #                    self._aggregate_matrix_Z.append(row)
        #                    break
        #elif aggregate_method == "complete" and all:
        #    stabilizers = deque(self._SZ)
        #    self._overchecked_SZ = []
        #    self._extra_aggregate_matrix_Z = []
        #    for i, row in enumerate(np.transpose(self._aggregate_matrix_Z)):
        #        if not np.any(row):
        #            print(f"    Gauge not contributing in the aggregate matrix {i}")
        #            stop = False
        #            gauge = self._GSZ[i]
        #            gauges = list(range(len(self._GSZ)))
        #            gauges.pop(i)
        #            for num_elements in range(1, 9):
        #                print(f"    Number of elements {num_elements}")
        #                for comb in iter.combinations(gauges, num_elements):
        #                    summed_gauge = gauge[:]
        #                    for element in comb:
        #                        summed_gauge = np.bitwise_xor(summed_gauge, self._GSZ[element])
        #                    for j, stab in enumerate(stabilizers):
        #                        if not np.any(np.bitwise_xor(summed_gauge, stab)) and np.any(np.bitwise_xor(gauge, self._GSZ)):
        #                            row = [0]*len(self._GSZ)
        #                            row[i]=1
        #                            for element in comb:
        #                                row[element]=1
        #                            self._extra_aggregate_matrix_Z.append(row)
        #                            self._overchecked_SZ.append(j)
        #                            stabilizers.rotate(1)
        #                            stop = True
        #                            break
        #                    if stop:
        #                        break
        #                if stop:
        #                    break


            
        

    @property
    def N(self):
        return self._N
    
    @property
    def m(self):
        return self._m

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
    def GZ(self):
        return self._GZ
    
    @property
    def GSZ(self):
        return self._GSZ
    
    @property
    def GX(self):
        return self._GX
    
    @property
    def GSX(self):
        return self._GSX
    
    @property
    def aggregate_matrix_X(self):
        return self._aggregate_matrix_X
    
    @property
    def aggregate_matrix_Z(self):
        return self._aggregate_matrix_Z
    

    @property
    def extra_aggregate_matrix_X(self):
        return self._extra_aggregate_matrix_X

    @property
    def extra_aggregate_matrix_Z(self):
        return self._extra_aggregate_matrix_Z

    @property
    def overchecked_SX(self):
        return self._overchecked_SX

    @property
    def overchecked_SZ(self):
        return self._overchecked_SZ
    

    def is_stabilizer(self, stabilizer):
        propagated_stabilizer = list(stabilizer)
        m=int(np.log2(len(stabilizer)))
        for i in reversed(range(m)):
            for j in reversed(range(2**(m-i-1))):
                for k in reversed(range(2**i)):
                    if propagated_stabilizer[k+j*2**(i+1)]==1:
                        propagated_stabilizer[2**i+k+j*2**(i+1)]=(propagated_stabilizer[2**i+k+j*2**(i+1)]+1)%2
        for position in range(64):
            if bin_wt(position)>self._rx and propagated_stabilizer[position]!=0:
                return False
        return True



