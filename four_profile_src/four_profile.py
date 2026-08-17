import networkx as nx
import itertools
import numpy as np

from util.four_profile_util import *

def four_profile(g: nx.Graph):
    """Computes the number of 4-node subgraphs that exists in a NetworkX Graph

    This is non-distributed implementation of algorithm from the paper
    "Distributed Estimation of Graph 4-Profiles" by E. Elenberg et al, 2016


    Returns a numpy vector of length 11, where component i is the count of F_i
    """
    two_hop_profile = two_hop_histogram(g)
    three_profile = three_profile_vertex(g)

    neighbor_sets = {}
    for v in g.nodes:
        neighbor_sets[v] = set(g.neighbors(v))
        g.nodes[v]['3_profile'] = vertex_3_data()

    four_profile_global = np.zeros(11)
    four_profile_local_equation = {}
    for v in g.nodes:
        four_profile_equation_data = np.zeros(17)
        four_profile_equation_data[0] = three_profile[v].n1_double
        four_profile_equation_data[1] = three_profile[v].n2c_double
        four_profile_equation_data[2] = three_profile[v].n2e_double
        four_profile_equation_data[3] = three_profile[v].n3_double
        four_profile_equation_data[4] = three_profile[v].n1_n2c
        four_profile_equation_data[5] = three_profile[v].n1_n2e
        four_profile_equation_data[6] = three_profile[v].n1_n3
        four_profile_equation_data[7] = three_profile[v].n2c_n2e
        four_profile_equation_data[8] = three_profile[v].n2c_n3
        four_profile_equation_data[9] = three_profile[v].n2e_n3
        four_profile_equation_data[10] = three_profile[v].num_disc_alone * g.degree(v)

        # (number of 2-paths to non-neighbours) choose 2
        four_profile_equation_data[11] = 0
        for a in set(g.nodes).difference(neighbor_sets[v], {v}):
            four_profile_equation_data[11] += (two_hop_profile[v][a] * (two_hop_profile[v][a] - 1)) / 2

        # Equation for F10 in the paper
        # count number of 4-cliques containing v
        # for each neighbour a of v, count number of triangles of abc where b and c are neighbours of v
        for a in neighbor_sets[v]:
            for (b,c) in three_profile[a].triangle_pairs:
                if b in neighbor_sets[v] and c in neighbor_sets[v]:
                    four_profile_equation_data[12] += 1
        four_profile_equation_data[12] = four_profile_equation_data[12] * 2
        # each clique at a vertex is counted three times once each for every incident edge, until fix matrix just scale this to 6x

        # 4th equation in (3) in the paper
        # For each neighbour a, sum (n3a - n3va), n3a is num triangles of a
        # n3va is number of shared triangles between v and a
        # sum of n3va will be total number of triangles containing v * 2
        for a in neighbor_sets[v]:
            four_profile_equation_data[13] += three_profile[a].num_triangles
        four_profile_equation_data[13] -= three_profile[v].n3

        # 5th equation in (3) in the paper
        # For each neighbour a, sum (n2e_a - n2c_va)
        for a in neighbor_sets[v]:
            four_profile_equation_data[14] += three_profile[a].n2e
        four_profile_equation_data[14] -= three_profile[v].n2c

        # Equation for F8 in the paper
        # for each neighbour a of v, count number of triangles abc where b and c are not neighbours of v
        # b and c must also be separate from v
        for a in neighbor_sets[v]:
            for (b,c) in three_profile[a].triangle_pairs:
                if b not in neighbor_sets[v] and c not in neighbor_sets[v] and b != v and c != v:
                    four_profile_equation_data[15] += 1

        four_profile_equation_data[16] = (g.number_of_nodes() - 1) * (g.number_of_nodes() - 2) * (g.number_of_nodes() - 3) / 6 # |V|-1 choose 3

        # solving for four-profile
        A0 = np.array([-6, -2, -6, -2, -3, -6, -3, -3, -2, -3, 0, 0, 0, 0, 0, 0, 6])
        A1 = np.array([1, 0, 0, -2, 0, 0, 0, 0, 0, -1, -1, -2, 0, 2, 1, -1, 0])
        A2 = np.array([0, 0, 0, 2, 0, 0, 0, 0, 0, 1, 1, 2, 0, -2, -1, 1, 0])
        A3 = np.array([0, 0, 0, 2, 0, 1, 0, 0, 0, 1, 0, 2, 0, -2, -1, 2, 0])
        A4 = np.array([0, 0, 0, 0, 2, 0, 0, -2, 0, 0, 0, 4, 1, -2, 0, 2, 0])
        A5 = np.array([0, 0, 0, -2, 0, 0, 0, 0, 0, -1, 0, -2, 0, 2, 1, -2, 0])
        A6 = np.array([0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, -4, -1, 2, 0, -2, 0])
        A7 = np.array([0, 0, 0, 0, 0, 0, 2, 0, 0, -2, 0, 0, -1, 2, 0, -2, 0])
        A8 = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0])
        A9 = np.array([0, 2, 0, 2, 0, 0, 0, 0, -1, 0, 0, 0, -1, 0, 0, 0, 0])
        A10 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 1, -2, 0, 2, 0])
        A11 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
        A12 = np.array([0, 0, 0, -2, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0])
        A13 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 1, -2, 0, 2, 0])
        A14 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 2, 0, -2, 0])
        A15 = np.array([0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0])
        A16 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])

        four_profile_local = np.zeros(17)
        four_profile_local[0] = np.dot(four_profile_equation_data, A0) / 6.
        four_profile_local[1] = np.dot(four_profile_equation_data, A1)
        four_profile_local[2] = np.dot(four_profile_equation_data, A2)
        four_profile_local[3] = np.dot(four_profile_equation_data, A3)
        four_profile_local[4] = np.dot(four_profile_equation_data, A4) / 4.
        four_profile_local[5] = np.dot(four_profile_equation_data, A5)
        four_profile_local[6] = np.dot(four_profile_equation_data, A6) / 2.
        four_profile_local[7] = np.dot(four_profile_equation_data, A7) / 4.
        four_profile_local[8] = np.dot(four_profile_equation_data, A8)
        four_profile_local[9] = np.dot(four_profile_equation_data, A9) / 6.
        four_profile_local[10] = np.dot(four_profile_equation_data, A10) / 4.
        four_profile_local[11] = np.dot(four_profile_equation_data, A11)
        four_profile_local[12] = np.dot(four_profile_equation_data, A12) / 2.
        four_profile_local[13] = np.dot(four_profile_equation_data, A13) / 2.
        four_profile_local[14] = np.dot(four_profile_equation_data, A14) / 4.
        four_profile_local[15] = np.dot(four_profile_equation_data, A15) / 2.
        four_profile_local[16] = np.dot(four_profile_equation_data, A16) / 6.

        four_profile_local_equation[v] = four_profile_local
        #debugging
        # print(four_profile_local[0])

        four_profile_global[0] += four_profile_local[0] / 4.
        four_profile_global[1] += four_profile_local[1] / 2.
        four_profile_global[2] += four_profile_local[2] / 4.
        four_profile_global[3] += four_profile_local[4]
        four_profile_global[4] += four_profile_local[6] / 2.
        four_profile_global[5] += four_profile_local[7] / 3.
        four_profile_global[6] += four_profile_local[9]
        four_profile_global[7] += four_profile_local[10] / 4.
        four_profile_global[8] += four_profile_local[11]
        four_profile_global[9] += four_profile_local[14] / 2.
        four_profile_global[10] += four_profile_local[16] / 4.

    # currently N0 is not being computed correct (A0 is off)
    # Since everything else is being counted correctly, count N0 from |V| choose 4 minus all other subgraph counts
    N0 = (g.number_of_nodes() * (g.number_of_nodes() - 1) * (g.number_of_nodes() - 2) * (g.number_of_nodes() - 3)) / 24
    for i in range(1,11):
        N0 -= four_profile_global[i]

    four_profile_global[0] = N0

    return four_profile_global

if __name__ == '__main__':
    # g = nx.gnp_random_graph(6, 1, seed=3)
    gnp150 = nx.gnp_random_graph(150, 0.3, seed=4)
    print(four_profile(gnp150))

    a = 0

