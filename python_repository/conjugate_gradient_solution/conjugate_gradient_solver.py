class Node:
    def __init__(self, node_id):
        self.id = node_id
        self.displacement = 0.0  # Unknown unless prescribed
        self.force = 0.0         # Applied force
        self.is_displacement_prescribed = False

class Element:
    def __init__(self, element_id, node1_id, node2_id, stiffness):
        self.id = element_id
        self.node1_id = node1_id  # Reference to first node
        self.node2_id = node2_id  # Reference to second node
        self.stiffness = stiffness  # Stiffness k_e
        self.force = 0.0            # Force in the element

class FEMSystem:
    def __init__(self, n_nodes):
        self.nodes = [Node(i) for i in range(n_nodes)]
        self.elements = []

    def calculate_displacements(self):
        n_nodes = len(self.nodes)
        fem_metadata = FEMMetaData(n_nodes)

        # Assemble global stiffness matrix
        for e in self.elements:
            i = e.node1_id
            j = e.node2_id
            k = e.stiffness

            fem_metadata.stiffness_matrix[i][i] += k
            fem_metadata.stiffness_matrix[i][j] -= k
            fem_metadata.stiffness_matrix[j][i] -= k
            fem_metadata.stiffness_matrix[j][j] += k

        # Compile forces vector, free and prescribed DOFs
        for i, node in enumerate(self.nodes):
            fem_metadata.forces_vector[i] = node.force
            if node.is_displacement_prescribed:
                fem_metadata.prescribed_dofs.append(i)
            else:
                fem_metadata.free_dofs.append(i)

        # Solve the system using Conjugate Gradient method
        displacements = fem_metadata.solve_linear_system_with_conjugate_gradient_method()
        for displacement_index, node_index in enumerate(fem_metadata.free_dofs):
            self.nodes[node_index].displacement = displacements[displacement_index]

        return fem_metadata

class FEMMetaData:
    def __init__(self, n_nodes):
        self.stiffness_matrix = [[0.0 for _ in range(n_nodes)] for _ in range(n_nodes)]
        self.forces_vector = [0.0 for _ in range(n_nodes)]
        self.free_dofs = []
        self.prescribed_dofs = []

    def solve_linear_system_with_conjugate_gradient_method(self):
        # Extract the reduced stiffness matrix and force vector
        K_reduced = [[self.stiffness_matrix[i][j] for j in self.free_dofs] for i in self.free_dofs]
        F_reduced = [self.forces_vector[i] for i in self.free_dofs]

        # Conjugate Gradient method
        n = len(F_reduced)
        x = [0.0 for _ in range(n)]  # Initial guess

        r = vec_sub(F_reduced, mat_vec_mul(K_reduced, x))
        p = r.copy()
        rs_old = vec_dot(r, r)

        tol = 1e-6
        max_iter = 1000

        for _ in range(max_iter):
            Ap = mat_vec_mul(K_reduced, p)
            pAp = vec_dot(p, Ap)

            if abs(pAp) < 1e-10:
                break  # Avoid division by zero

            alpha = rs_old / pAp
            x = vec_add(x, vec_scalar_mul(p, alpha))
            r = vec_sub(r, vec_scalar_mul(Ap, alpha))

            rs_new = vec_dot(r, r)
            if rs_new ** 0.5 < tol:
                break

            beta = rs_new / rs_old
            p = vec_add(r, vec_scalar_mul(p, beta))
            rs_old = rs_new

        return x

# Helper functions for vector and matrix operations
def vec_dot(a, b):
    return sum(ai * bi for ai, bi in zip(a, b))

def vec_add(a, b):
    return [ai + bi for ai, bi in zip(a, b)]

def vec_sub(a, b):
    return [ai - bi for ai, bi in zip(a, b)]

def vec_scalar_mul(a, scalar):
    return [scalar * ai for ai in a]

def mat_vec_mul(A, x):
    return [vec_dot(row, x) for row in A]

def main():
    n_nodes = 5
    fem_system = FEMSystem(n_nodes)

    # Prescribed displacements at nodes 1 and 5 (indices 0 and 4)
    fem_system.nodes[0].is_displacement_prescribed = True  # Node 1 fixed
    fem_system.nodes[4].is_displacement_prescribed = True  # Node 5 fixed

    # Applied force at node 3 (index 2)
    fem_system.nodes[2].force = 1000.0  # F3 = 1000 N

    # Initialize elements
    fem_system.elements.append(Element(0, 0, 1, 500.0))  # Element 1
    fem_system.elements.append(Element(1, 1, 3, 400.0))  # Element 2
    fem_system.elements.append(Element(2, 2, 1, 600.0))  # Element 3
    fem_system.elements.append(Element(3, 0, 2, 200.0))  # Element 4
    fem_system.elements.append(Element(4, 2, 3, 400.0))  # Element 5
    fem_system.elements.append(Element(5, 3, 4, 300.0))  # Element 6

    fem_metadata = fem_system.calculate_displacements()

    # Compute reactions at prescribed DOFs
    reactions = []
    for i in fem_metadata.prescribed_dofs:
        Ri = 0.0
        for j in range(n_nodes):
            Ri += fem_metadata.stiffness_matrix[i][j] * fem_system.nodes[j].displacement
        Ri -= fem_metadata.forces_vector[i]
        reactions.append((i, Ri))

    # Compute element forces
    for e in fem_system.elements:
        i = e.node1_id
        j = e.node2_id
        ui = fem_system.nodes[i].displacement
        uj = fem_system.nodes[j].displacement
        delta_u = uj - ui
        e.force = e.stiffness * delta_u

    # Output results
    print("Node Displacements:")
    for node in fem_system.nodes:
        if node.is_displacement_prescribed:
            print(f"Node {node.id + 1}: Fixed (Displacement = {node.displacement:.4f} mm)")
        else:
            print(f"Node {node.id + 1}: u = {node.displacement:.4f} mm")

    print("\nElement Forces (Positive=Tension, Negative=Compression):")
    for e in fem_system.elements:
        print(f"Element {e.id + 1}: P = {e.force:.2f} N")

    print("\nReactions at Prescribed Nodes:")
    for node_id, reaction in reactions:
        print(f"Node {node_id + 1}: R = {reaction:.2f} N")

if __name__ == "__main__":
    main()
