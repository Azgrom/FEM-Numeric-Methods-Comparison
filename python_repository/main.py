# Define Node and Element classes
class Node:
    def __init__(self, node_id, is_prescribed=False):
        self.id = node_id
        self.displacement = 0.0  # Unknown unless prescribed
        self.force = 0.0         # Applied force
        self.is_prescribed = is_prescribed  # Boundary condition

class Element:
    def __init__(self, elem_id, node1, node2, stiffness):
        self.id = elem_id
        self.node1 = node1  # Index of first node
        self.node2 = node2  # Index of second node
        self.stiffness = stiffness
        self.force = 0.0    # Force in the element

# Function to assemble global stiffness matrix
def assemble_global_stiffness_matrix(n_nodes, elements):
    K = [[0.0 for _ in range(n_nodes)] for _ in range(n_nodes)]
    for elem in elements:
        i = elem.node1
        j = elem.node2
        k = elem.stiffness
        K[i][i] += k
        K[i][j] -= k
        K[j][i] -= k
        K[j][j] += k
    return K

# Function to solve linear system using Gaussian elimination with partial pivoting
def solve_linear_system(A, b):
    n = len(b)
    # Create augmented matrix
    M = [A[i][:] + [b[i]] for i in range(n)]
    # Forward elimination
    for k in range(n):
        # Partial pivoting
        max_row = max(range(k, n), key=lambda i: abs(M[i][k]))
        if M[max_row][k] == 0:
            raise ValueError("Singular matrix!")
        # Swap rows if needed
        if max_row != k:
            M[k], M[max_row] = M[max_row], M[k]
        # Eliminate entries below pivot
        for i in range(k+1, n):
            factor = M[i][k] / M[k][k]
            for j in range(k, n+1):
                M[i][j] -= factor * M[k][j]
    # Back substitution
    x = [0.0 for _ in range(n)]
    for i in range(n-1, -1, -1):
        x[i] = M[i][n]
        for j in range(i+1, n):
            x[i] -= M[i][j] * x[j]
        x[i] /= M[i][i]
    return x

def main():
    # Number of nodes
    n_nodes = 5

    # Initialize nodes
    nodes = [Node(i) for i in range(n_nodes)]

    # Prescribed displacements at nodes 1 and 5 (indices 0 and 4)
    nodes[0].is_prescribed = True  # Node 1 fixed
    nodes[4].is_prescribed = True  # Node 5 fixed

    # Applied force at node 3 (index 2)
    nodes[2].force = 1000.0  # F3 = 1000 N

    # Initialize elements
    elements = []

    # Element definitions (node indices are zero-based)
    elements.append(Element(1, 0, 1, 500.0))  # Element 1: nodes 0 and 1, k = 500 N/mm
    elements.append(Element(2, 1, 3, 400.0))  # Element 2: nodes 1 and 3, k = 400 N/mm
    elements.append(Element(3, 2, 1, 600.0))  # Element 3: nodes 2 and 1, k = 600 N/mm
    elements.append(Element(4, 0, 2, 200.0))  # Element 4: nodes 0 and 2, k = 200 N/mm
    elements.append(Element(5, 2, 3, 400.0))  # Element 5: nodes 2 and 3, k = 400 N/mm
    elements.append(Element(6, 3, 4, 300.0))  # Element 6: nodes 3 and 4, k = 300 N/mm

    # Assemble global stiffness matrix K
    K_global = assemble_global_stiffness_matrix(n_nodes, elements)

    # Construct force vector F
    F_global = [node.force for node in nodes]

    # Identify free and prescribed DOFs
    free_dofs = [i for i, node in enumerate(nodes) if not node.is_prescribed]
    prescribed_dofs = [i for i, node in enumerate(nodes) if node.is_prescribed]

    # Reduce stiffness matrix and force vector for free DOFs
    n_free = len(free_dofs)
    K_reduced = [[0.0 for _ in range(n_free)] for _ in range(n_free)]
    F_reduced = [0.0 for _ in range(n_free)]

    for idx_i, i in enumerate(free_dofs):
        F_reduced[idx_i] = F_global[i]
        for idx_j, j in enumerate(free_dofs):
            K_reduced[idx_i][idx_j] = K_global[i][j]
        # Subtract contributions from prescribed displacements (which are zero here)
        for j in prescribed_dofs:
            F_reduced[idx_i] -= K_global[i][j] * nodes[j].displacement

    # Solve for displacements at free DOFs
    displacements = solve_linear_system(K_reduced, F_reduced)

    # Assign displacements to nodes
    for idx, dof in enumerate(free_dofs):
        nodes[dof].displacement = displacements[idx]

    # Compute reactions at prescribed DOFs
    reactions = []
    for i in prescribed_dofs:
        Ri = sum(K_global[i][j] * nodes[j].displacement for j in range(n_nodes)) - F_global[i]
        reactions.append((i, Ri))

    # Compute element forces
    for elem in elements:
        i = elem.node1
        j = elem.node2
        ui = nodes[i].displacement
        uj = nodes[j].displacement
        delta_u = uj - ui
        P = elem.stiffness * delta_u  # Positive P means tension
        elem.force = P

    # Output results
    print("Node Displacements:")
    for node in nodes:
        if node.is_prescribed:
            print(f"Node {node.id + 1}: Fixed (Displacement = {node.displacement:.4f} mm)")
        else:
            print(f"Node {node.id + 1}: Displacement = {node.displacement:.4f} mm")

    print("\nElement Forces (Positive=Tension, Negative=Compression):")
    for elem in elements:
        print(f"Element {elem.id}: Force = {elem.force:.2f} N")

    print("\nReactions at Prescribed Nodes:")
    for node_id, reaction in reactions:
        print(f"Node {node_id + 1}: Reaction = {reaction:.2f} N")

if __name__ == "__main__":
    main()
