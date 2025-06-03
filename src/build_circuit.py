import numpy as np
import scipy as sp
import stim
from typing import List, FrozenSet, Dict
from scipy.sparse import csc_matrix
import networkx as nx

from src.utils import edge_coloring_bipartite


def build_QRM_circuit(QRM, p, num_repeat, z_basis=True, use_both=False, HZH=False, aggregate_method="simple"):
    circuit = stim.Circuit()
    
    #edge_coloring, colors = edge_coloring_bipartite(np.concatenate((QRM.GSZ, QRM.GSX)))
    
    data_size = QRM.N
    stab_size = len(QRM.GSX)
    #Data and ancilla initialization
    for i in range(QRM.N):
        circuit.append("R" if z_basis else "RX", i)
        circuit.append("X_ERROR" if z_basis else "Z_ERROR", i, p)
    for i in range(stab_size):
        circuit.append("R", data_size+i)
        circuit.append("X_ERROR", data_size+i, p)
        circuit.append("RX", data_size+stab_size+i)
        circuit.append("Z_ERROR", data_size+stab_size+i, p)
    circuit.append("TICK")

    #Z stab SE first round
    Z_stab_SE = ""
    X_stab_SE = ""
    if aggregate_method=="complete":
        GSZT = list(np.transpose(np.concatenate((QRM.GSZ, QRM.GSX))))
        while np.any(GSZT):
            linked_GSZ = list(range(2*stab_size))
            for i, row in enumerate(GSZT):
                for j, element in enumerate(row):
                    if element == 1 and j in linked_GSZ and i<=32:
                        GSZT[i][j]=0
                        linked_GSZ.remove(j)
                        Z_stab_SE += f"CNOT {i} {j+data_size}\n"
                        Z_stab_SE += f"DEPOLARIZE2({p}) {i} {j+data_size}\n"
                        break
                    elif element == 1 and j in linked_GSZ and i>32:
                        GSZT[i][j]=0
                        linked_GSZ.remove(j)
                        Z_stab_SE += f"CNOT {j+data_size+stab_size} {i}\n"
                        Z_stab_SE += f"DEPOLARIZE2({p}) {j+data_size+stab_size} {i}\n"
                        break
            Z_stab_SE += f"TICK \n"
        circuit += stim.Circuit(Z_stab_SE)
    #circuit += stim.Circuit(X_stab_SE)

    #for color in range(colors+1):
    #    for edge in edge_coloring[color]:
    #        if edge[0]<48:
    #            circuit.append("CNOT", [int(edge[1]), int(edge[0])+data_size])
    #        else:
    #            circuit.append("CNOT", [int(edge[0])+data_size, int(edge[1])])
    #    #linked_edges = X_tanner_graph.edges(edge[1], "color")
    #    #max_color_value = max(edge_tuple[2] for edge_tuple in linked_edges)
    #    #if color == max_color_value:
    #    #    circuit.append("Z_ERROR", n + len(GZ)+int(edge[1][1:]), p)
    #    #    circuit.append("MX", n + len(GZ)+int(edge[1][1:]))
                
    circuit.append("TICK")
    ###########ROUNDS
    round_circuit = stim.Circuit()

    #Data idling
    for i in range(QRM.N):
        round_circuit.append("DEPOLARIZE1", i, p)


    #Ancilla measurment and initialization
    X_ancilla_measurment = ""
    Z_ancilla_measurment = ""
    for i in range(stab_size):
        Z_ancilla_measurment += f"X_ERROR({p}) {data_size+i}\n"
        Z_ancilla_measurment += f"MR {data_size+i}\n"
        Z_ancilla_measurment += f"X_ERROR({p}) {data_size+i}\n"
        X_ancilla_measurment += f"Z_ERROR({p}) {data_size+stab_size+i}\n"
        X_ancilla_measurment += f"MRX {data_size+stab_size+i}\n"
        X_ancilla_measurment += f"Z_ERROR({p}) {data_size+stab_size+i}\n"
    round_circuit += stim.Circuit(Z_ancilla_measurment)
    round_circuit += stim.Circuit(X_ancilla_measurment)
    round_circuit.append("TICK")

    
    ######
    circuit += (num_repeat-1)*round_circuit


    #Final ancilla measurment
    final_X_ancilla_measurment = ""
    final_Z_ancilla_measurment = ""
    for i in range(stab_size):
        final_Z_ancilla_measurment += f"X_ERROR({p}) {data_size+i}\n"
        final_Z_ancilla_measurment += f"M {data_size+i}\n"
        final_X_ancilla_measurment += f"Z_ERROR({p}) {data_size+stab_size+i}\n"
        final_X_ancilla_measurment += f"MX {data_size+stab_size+i}\n"
    circuit += stim.Circuit(final_Z_ancilla_measurment)
    circuit += stim.Circuit(final_X_ancilla_measurment)
    
    
    #detector_str = ""
    #for round_num in range(num_repeat):
    #    if round_num==0:
    #        if z_basis:
    #            for i, row in enumerate(QRM.aggregate_matrix_Z):
    #                detector_str += "DETECTOR "
    #                for j, value in enumerate(row):
    #                    if value == 1:
    #                        detector_str += f"rec[{-((num_repeat-round_num)*stab_size*2)+j}] "
    #                for l in QRM.overchecked_SZ:
    #                    if l==i:
    #                        for k, element in enumerate(QRM.extra_aggregate_matrix_Z[QRM.overchecked_SZ.index(i)]):
    #                            if value == 1:
    #                                detector_str += f"rec[{-((num_repeat-round_num)*stab_size*2)+k}] "
    #                detector_str += "\n"
    #        else:
    #            for i, row in enumerate(QRM.aggregate_matrix_X):
    #                detector_str += "DETECTOR "
    #                for j, value in enumerate(row):
    #                    if value == 1:
    #                        detector_str += f"rec[{-((num_repeat-round_num)*stab_size*2)+j}] "
    #                if i in QRM.overchecked_SX:
    #                    for k, element in enumerate(QRM.extra_aggregate_matrix_X[QRM.overchecked_SX.index(i)]):
    #                        if value == 1:
    #                            detector_str += f"rec[{-((num_repeat-round_num)*stab_size*2)+k}] "
    #                detector_str += "\n"
    #    else:
    #        if z_basis:
    #            for i, row in enumerate(QRM.aggregate_matrix_Z):
    #                detector_str += "DETECTOR "
    #                for j, value in enumerate(row):
    #                    if value == 1:
    #                        detector_str += f"rec[{-((num_repeat-round_num+1)*stab_size*2)+j}] rec[{-((num_repeat-round_num)*stab_size*2)+j}] "
    #                if i in QRM.overchecked_SZ:
    #                    for k, element in enumerate(QRM.extra_aggregate_matrix_Z[QRM.overchecked_SZ.index(i)]):
    #                        if value == 1:
    #                            detector_str += f"rec[{-((num_repeat-round_num+1)*stab_size*2)+k}] rec[{-((num_repeat-round_num)*stab_size*2)+k}] "
    #                detector_str += "\n"
    #        else:
    #            for i, row in enumerate(QRM.aggregate_matrix_X):
    #                detector_str += "DETECTOR "
    #                for j, value in enumerate(row):
    #                    if value == 1:
    #                        detector_str += f"rec[{-((num_repeat-round_num+1)*stab_size*2)+j+stab_size}] rec[{-((num_repeat-round_num)*stab_size*2)+j+stab_size}] "
    #                for l in QRM.overchecked_SZ:
    #                    if l==i:
    #                        for k, element in enumerate(QRM.extra_aggregate_matrix_X[QRM.overchecked_SX.index(i)]):
    #                            if value == 1:
    #                                detector_str += f"rec[{-((num_repeat-round_num+1)*stab_size*2)+k+stab_size}] rec[{-((num_repeat-round_num)*stab_size*2)+k+stab_size}] "
    #                detector_str += "\n"
    #detector_circuit = stim.Circuit(detector_str) 
    #circuit += detector_circuit


    for i in range(QRM.N):
        circuit.append("X_ERROR" if z_basis else "Z_ERROR", i, p)
        circuit.append("M" if z_basis else "MX", i)
    round_circuit.append("TICK")


    #detector_str = ""
    #if z_basis:
    #    for i, row in enumerate(QRM.SZ):
    #        detector_str += "DETECTOR "
    #        for k, value in enumerate(aggregate_matrix_Z[i]):
    #            if value == 1:
    #                detector_str += f"rec[{-(QRM.N+stab_size*2)+k}] "
    #        for j, value in enumerate(row):                                     
    #            if value==1:                                                    
    #                detector_str +=f"rec[{j-QRM.N}] "                               
    #        detector_str += "\n"
    #observable_str = ""
    #if z_basis:
    #    for i, row in enumerate(QRM.LZ):
    #        observable_str += f"OBSERVABLE_INCLUDE({i}) "
    #        for j, value in enumerate(row):
    #            if value == 1:
    #                observable_str += f"rec[{j-QRM.N}] "
    #        observable_str += "\n"
    #detector_circuit = stim.Circuit(detector_str) 
    #circuit += detector_circuit
    #    
    #observable_circuit = stim.Circuit(observable_str)
    #circuit+= observable_circuit

    return circuit



def get_syndrom_extraction_order(QRM):
    bipartite_X = nx.Graph()
    bipartite_Z = nx.Graph()
    
    for i in range(QRM.N):
        bipartite_X.add_node(f"Q{i}", bipartite=0)
        bipartite_Z.add_node(f"Q{i}", bipartite=0)
    
    for i in range(QRM.GSX):
        bipartite_Z.add_node(f"S{QRM.N+i}", bipartite=1)
        bipartite_X.add_node(f"S{QRM.N+i}", bipartite=1)


def dict_to_csc_matrix(elements_dict, shape):
    # Constructs a `scipy.sparse.csc_matrix` check matrix from a dictionary `elements_dict` 
    # giving the indices of nonzero rows in each column.
    nnz = sum(len(v) for k, v in elements_dict.items())
    data = np.ones(nnz, dtype=np.uint8)
    row_ind = np.zeros(nnz, dtype=np.int64)
    col_ind = np.zeros(nnz, dtype=np.int64)
    i = 0
    for col, v in elements_dict.items():
        for row in v:
            row_ind[i] = row
            col_ind[i] = col
            i += 1
    return csc_matrix((data, (row_ind, col_ind)), shape=shape)

def dem_to_check_matrices(dem: stim.DetectorErrorModel, return_col_dict=False):

    DL_ids: Dict[str, int] = {} # detectors + logical operators
    L_map: Dict[int, FrozenSet[int]] = {} # logical operators
    priors_dict: Dict[int, float] = {} # for each fault

    def handle_error(prob: float, detectors: List[int], observables: List[int]) -> None:
        dets = frozenset(detectors)
        obs = frozenset(observables)
        key = " ".join([f"D{s}" for s in sorted(dets)] + [f"L{s}" for s in sorted(obs)])

        if key not in DL_ids:
            DL_ids[key] = len(DL_ids)
            priors_dict[DL_ids[key]] = 0.0

        hid = DL_ids[key]
        L_map[hid] = obs
#         priors_dict[hid] = priors_dict[hid] * (1 - prob) + prob * (1 - priors_dict[hid])
        priors_dict[hid] += prob

    for instruction in dem.flattened():
        if instruction.type == "error":
            dets: List[int] = []
            frames: List[int] = []
            t: stim.DemTarget
            p = instruction.args_copy()[0]
            for t in instruction.targets_copy():
                if t.is_relative_detector_id():
                    dets.append(t.val)
                elif t.is_logical_observable_id():
                    frames.append(t.val)
            handle_error(p, dets, frames)
        elif instruction.type == "detector":
            pass
        elif instruction.type == "logical_observable":
            pass
        else:
            raise NotImplementedError()
    check_matrix = dict_to_csc_matrix({v: [int(s[1:]) for s in k.split(" ") if s.startswith("D")] 
                                       for k, v in DL_ids.items()},
                                      shape=(dem.num_detectors, len(DL_ids)))
    observables_matrix = dict_to_csc_matrix(L_map, shape=(dem.num_observables, len(DL_ids)))
    priors = np.zeros(len(DL_ids))
    for i, p in priors_dict.items():
        priors[i] = p

    if return_col_dict:
        return check_matrix, observables_matrix, priors, DL_ids
    return check_matrix, observables_matrix, priors

                    
