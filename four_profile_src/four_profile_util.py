import networkx as nx

def two_hop_histogram(g: nx.Graph):
    """For each node v in g, compute the number 2-paths from v to the other nodes of g

    Assumes that g is a simple graph with no loops.
    Returns a dictionary, keys are the nodes of g, and values are another dictionary that counts the number of 2-paths
    to other nodes. For example, two_hop_profile[v][u] is equal to the number of 2 paths from v to u.
    """
    two_hop_profile = {}

    neighbor_sets = {}
    # initialize tuples for each vertex
    for v in g.nodes:
        neighbor_sets[v] = set(g.neighbors(v))
        two_hop_profile[v] = dict()
        for u in g.nodes:
            two_hop_profile[v][u] = 0

            # below approach works well if instead of a dictionary, the 2-path counts were a vector
            # when a vector, it is easy to add just add two vectors together instead of neighbour checks in the next loop
            # if u in neighbor_sets[v]:
            #     two_hop_profile[v][u] = 1
            # else:
            #     two_hop_profile[v][u] = 0


    for e in g.edges:
        u = e[0]
        v = e[1]

        # for every neighbour a of v, if ua is not an edge, then uva is a 2-path
        # then, for every neighbour a of v that is not a neighbour of u, increase ua path count by 1
        for a in neighbor_sets[v].difference(neighbor_sets[u], {u}):
            two_hop_profile[u][a] += 1

        # same goes for u
        for a in neighbor_sets[u].difference(neighbor_sets[v], {v}):
            two_hop_profile[v][a] += 1

    return two_hop_profile


class vertex_3_data:
    __slots__ = ('num_triangles', 'num_wedges', 'num_discs', 'num_anti_triangle', 'num_wedge_center', 'num_wedge_leaf',
                 'num_disc_leaf', 'num_disc_alone',
                 'n1', 'n2c', 'n2e', 'n3',
                 'n1_double', 'n2c_double', 'n2e_double', 'n3_double',
                 'n1_n2c', 'n1_n2e', 'n1_n3', 'n2c_n2e', 'n2c_n3', 'n2e_n3',
                 'triangle_pairs'
                 )

    def __init__(self):
        self.num_triangles = 0
        self.num_wedges = 0
        self.num_discs = 0
        self.num_anti_triangle = 0

        self.num_wedge_center = 0
        self.num_wedge_leaf = 0
        self.num_disc_leaf = 0
        self.num_disc_alone = 0

        # 3-profile algebraic variables
        self.n1 = 0 # number of disc leafs
        self.n2c = 0 # number of wedge center (scaled by 2)
        self.n2e = 0 # number of wedge leafs
        self.n3 = 0 # number of triangles (scaled by 2)

        # 4-profile equation variables
        self.n1_double = 0
        self.n2c_double = 0
        self.n2e_double = 0
        self.n3_double = 0
        self.n1_n2c = 0
        self.n1_n2e = 0
        self.n1_n3 = 0
        self.n2c_n2e = 0
        self.n2c_n3 = 0
        self.n2e_n3 = 0

        self.triangle_pairs = set() # set of 2-tuples that form triangles with v


def three_profile_vertex(g: nx.Graph):
    """Computes the 3-profile of a graph g, the number of subgraphs of each graph on 3 vertices.

    For each vertex v, three_profile_vertex computes the number of each 3 node graph v is in.
    For example, three_profile_vertex counts the number of different triangles contained in each vertex.

    Returns a dict where the keys are the nodes of g, and the values are vertex_3_data objects, that contain the counts of different 3-vertex subgraphs.
    """
    vertex_profile = {}

    neighbor_sets = {}
    for v in g.nodes:
        neighbor_sets[v] = set(g.neighbors(v))
        g.nodes[v]['3_profile'] = vertex_3_data()

    for e in g.edges:
        u = e[0]
        v = e[1]
        num_u_neighbors = len(neighbor_sets[u])
        num_v_neighbors = len(neighbor_sets[v])
        common_neighbors = neighbor_sets[u].intersection(neighbor_sets[v])
        n3_a = len(common_neighbors)

        g.nodes[v]['3_profile'].n3 += n3_a
        g.nodes[u]['3_profile'].n3 += n3_a

        g.nodes[v]['3_profile'].n3_double += (n3_a * (n3_a - 1)) / 2
        g.nodes[u]['3_profile'].n3_double += (n3_a * (n3_a - 1)) / 2

        for neighbor in common_neighbors:
            g.nodes[v]['3_profile'].triangle_pairs.add(tuple(sorted((u,neighbor)))) # we sort since each triangle will be added twice, triangle auv will be added for u on edge au and uv
            g.nodes[u]['3_profile'].triangle_pairs.add(tuple(sorted((v,neighbor))))

        # to count wedges, split up wedges centered at u and centered at v
        # to count wedges centered at u containing uv, count all neighbors of u, except for v, and subtract common neighbors
        n2c_ua = num_u_neighbors - n3_a - 1
        g.nodes[u]['3_profile'].n2c += n2c_ua
        g.nodes[v]['3_profile'].n2e += n2c_ua

        g.nodes[u]['3_profile'].n2c_double += (n2c_ua * (n2c_ua - 1)) / 2
        g.nodes[v]['3_profile'].n2e_double += (n2c_ua * (n2c_ua - 1)) / 2

        # to count wedges centered at v containing uv, count all neighbors of v, except for u, and subtract common neighbors
        n2c_va = num_v_neighbors - n3_a - 1
        g.nodes[v]['3_profile'].n2c += n2c_va
        g.nodes[u]['3_profile'].n2e += n2c_va

        g.nodes[v]['3_profile'].n2c_double += (n2c_va * (n2c_va - 1)) / 2
        g.nodes[u]['3_profile'].n2e_double += (n2c_va * (n2c_va - 1)) / 2

        # to get disc counts, number of nodes that aren't adjacent to u or v. |V(g)| - num of distinct neighbors of u union distinct neighbors of v
        # to calculate number of distinct neighbors, take both neighbor sets combined, and substract the common ones (they're counted twice)
        n1_a = g.number_of_nodes() - num_u_neighbors - num_v_neighbors + n3_a
        g.nodes[v]['3_profile'].n1 += n1_a
        g.nodes[u]['3_profile'].n1 += n1_a

        g.nodes[v]['3_profile'].n1_double += (n1_a * (n1_a - 1)) / 2
        g.nodes[u]['3_profile'].n1_double += (n1_a * (n1_a - 1)) / 2

        # compute calculated variable for 4-profile
        g.nodes[u]['3_profile'].n1_n2c += n1_a * n2c_ua
        g.nodes[u]['3_profile'].n1_n2e += n1_a * n2c_va
        g.nodes[u]['3_profile'].n1_n3 += n1_a * n3_a
        g.nodes[u]['3_profile'].n2c_n2e += n2c_ua * n2c_va
        g.nodes[u]['3_profile'].n2c_n3 += n2c_ua * n3_a
        g.nodes[u]['3_profile'].n2e_n3 += n2c_va * n3_a

        g.nodes[v]['3_profile'].n1_n2c += n1_a * n2c_va
        g.nodes[v]['3_profile'].n1_n2e += n1_a * n2c_ua
        g.nodes[v]['3_profile'].n1_n3 += n1_a * n3_a
        g.nodes[v]['3_profile'].n2c_n2e += n2c_ua * n2c_va
        g.nodes[v]['3_profile'].n2c_n3 += n2c_va * n3_a
        g.nodes[v]['3_profile'].n2e_n3 += n2c_ua * n3_a


        # for vertex in e:
            # g.nodes[vertex]['3_profile'].n1_double += (n1_additions * (n1_additions - 1)) / 2
            # g.nodes[vertex]['3_profile'].n2c_double += (g.nodes[vertex]['3_profile'].n2c * g.nodes[vertex][
            #     '3_profile'].n2c - 1) / 2
            # g.nodes[vertex]['3_profile'].n2e_double += (g.nodes[vertex]['3_profile'].n2e * g.nodes[vertex][
            #     '3_profile'].n2e - 1) / 2
            # g.nodes[vertex]['3_profile'].n3_double += (g.nodes[vertex]['3_profile'].n3 * g.nodes[vertex][
                # '3_profile'].n3 - 1) / 2
            #
            # g.nodes[vertex]['3_profile'].n1_n2c += g.nodes[vertex]['3_profile'].n1 * g.nodes[vertex]['3_profile'].n2c
            # g.nodes[vertex]['3_profile'].n1_n2e += g.nodes[vertex]['3_profile'].n1 * g.nodes[vertex]['3_profile'].n2e
            # g.nodes[vertex]['3_profile'].n1_n3 += g.nodes[vertex]['3_profile'].n1 * g.nodes[vertex]['3_profile'].n3
            # g.nodes[vertex]['3_profile'].n2c_n2e += g.nodes[vertex]['3_profile'].n2c * g.nodes[vertex]['3_profile'].n2e
            # g.nodes[vertex]['3_profile'].n2c_n3 += g.nodes[vertex]['3_profile'].n2c * g.nodes[vertex]['3_profile'].n3
            # g.nodes[vertex]['3_profile'].n2e_n3 += g.nodes[vertex]['3_profile'].n2e * g.nodes[vertex]['3_profile'].n3

    for v in g.nodes:
        g.nodes[v]['3_profile'].num_triangles = g.nodes[v]['3_profile'].n3 / 2
        g.nodes[v]['3_profile'].num_wedges = g.nodes[v]['3_profile'].n2e + g.nodes[v][
            '3_profile'].n2c / 2

        g.nodes[v]['3_profile'].num_disc_alone = g.number_of_edges() - g.nodes[v]['3_profile'].num_triangles - \
                                                 g.nodes[v]['3_profile'].n2e - g.degree(v)
        g.nodes[v]['3_profile'].num_discs = g.nodes[v]['3_profile'].n1 + g.nodes[v][
            '3_profile'].num_disc_alone

        g.nodes[v]['3_profile'].num_anti_triangle = (g.number_of_nodes() - 1) * (g.number_of_nodes() - 2) / 2 - \
                                                    g.nodes[v]['3_profile'].num_triangles - g.nodes[v][
                                                        '3_profile'].num_wedges - g.nodes[v]['3_profile'].num_discs

        vertex_profile[v] = g.nodes[v]['3_profile']

    return vertex_profile

# Test function for module
def _test():
    return 1

if __name__ == '__main__':
    _test()

