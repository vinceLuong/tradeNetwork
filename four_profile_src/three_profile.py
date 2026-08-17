import networkx as nx

def three_profile(g: nx.Graph):
    """Computes the number of 3-node subgraphs that exists in a NetworkX Graph
    Assumes no loops.

    Returns a 4-tuple: (number of empty subgraphs, number of discs, number of wedges, number of triangles)
    """
    neighbor_sets = {}
    for v in g.nodes:
        neighbor_sets[v] = set(g.neighbors(v))

    triangle_counts = 0  # 3 edges
    wedge_counts = 0  # 2 edges
    disc_counts = 0  # 1 edge
    empty_counts = 0  # 0 edges

    for e in g.edges:
        u = e[0]
        v = e[1]
        num_u_neighbors = len(neighbor_sets[u])
        num_v_neighbors = len(neighbor_sets[v])
        num_common_neighbors = len(neighbor_sets[u].intersection(neighbor_sets[v]))

        triangle_counts += num_common_neighbors

        # to count wedges, split up wedges centered at u and centered at v
        # to count wedges centered at u containing uv, count all neighbors of u, except for v, and subtract common neighbors
        wedge_counts += num_u_neighbors - num_common_neighbors - 1
        # to count wedges centered at v containing uv, count all neighbors of v, except for u, and subtract common neighbors
        wedge_counts += num_v_neighbors - num_common_neighbors - 1  # wedges centered at v containing uv

        # to get disc counts, number of nodes that aren't adjacent to u or v. |V(g)| - num of distinct neighbors of u union distinct neighbors of v
        # to calculate number of distinct neighbors, take both neighbor sets combined, and substract the common ones (they're counted twice)
        disc_counts += g.number_of_nodes() - num_u_neighbors - num_v_neighbors + num_common_neighbors

    wedge_counts = wedge_counts / 2  # each wedge counted twice, once per edge in wedge
    triangle_counts = triangle_counts / 3  # each triangle counted thrice, once per edge in triangle
    empty_counts = (g.number_of_nodes() * (g.number_of_nodes() - 1) * (
                g.number_of_nodes() - 2) / 6) - triangle_counts - wedge_counts - disc_counts

    return empty_counts, disc_counts, wedge_counts, triangle_counts