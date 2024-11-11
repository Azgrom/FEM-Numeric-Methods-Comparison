import math


class Node:
    def __init__(self, id):
        self.id = id
        self.displacement = 0.0  # Unknown unless prescribed
        self.force = 0.0  # Applied force
        self.is_displacement_prescribed = False


class Element:
    def __init__(self, id, node1_id, node2_id, stiffness):
        self.id = id
        self.node1_id = node1_id  # Reference to first node
        self.node2_id = node2_id  # Reference to second node
        self.stiffness = stiffness  # Stiffness k_e
        self.force = 0.0  # Force in the element


class FEMMetaData:
    def __init__(self, nodes_count):
        self.stiffness_matrix = [[0.0 for _ in range(nodes_count)] for _ in range(nodes_count)]
        self.forces_vector = []
        self.free_dofs = []
        self.prescribed_dofs = []

    def solve_linear_system_with_gaussian_elimination_method(self):
        free_dofs_len = len(self.free_dofs)
        augmented_matrix = [[0.0 for _ in range(free_dofs_len + 1)] for _ in range(free_dofs_len)]

        for i, free_dof_i in enumerate(self.free_dofs):
            augmented_matrix[i][free_dofs_len] = self.forces_vector[free_dof_i]
            for j, free_dof_j in enumerate(self.free_dofs):
                augmented_matrix[i][j] = self.stiffness_matrix[free_dof_i][free_dof_j]

        # Gaussian elimination with partial pivoting
        for k in range(free_dofs_len):
            # Find the k-th pivot
            max_row = max(range(k, free_dofs_len), key=lambda i: abs(augmented_matrix[i][k]))
            if augmented_matrix[max_row][k] == 0:
                raise ValueError("Matrix is singular!")
            if max_row != k:
                # Swap rows
                augmented_matrix[k], augmented_matrix[max_row] = augmented_matrix[max_row], augmented_matrix[k]

            # Eliminate entries below pivot
            for i in range(k + 1, free_dofs_len):
                factor = augmented_matrix[i][k] / augmented_matrix[k][k]
                for j in range(k, free_dofs_len + 1):
                    augmented_matrix[i][j] -= factor * augmented_matrix[k][j]

        # Back substitution
        reactions = [0.0 for _ in range(free_dofs_len)]
        for i in range(free_dofs_len - 1, -1, -1):
            reactions[i] = augmented_matrix[i][free_dofs_len]
            for j in range(i + 1, free_dofs_len):
                reactions[i] -= augmented_matrix[i][j] * reactions[j]
            reactions[i] /= augmented_matrix[i][i]

        return reactions


class FEMSystem:
    def __init__(self, n_nodes):
        self.nodes = [Node(i) for i in range(n_nodes)]
        self.elements = []

    def calculate_displacements_with_gauss_elimination(self):
        n_nodes = len(self.nodes)
        fem_metadata = FEMMetaData(n_nodes)

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
            fem_metadata.forces_vector.append(node.force)
            if node.is_displacement_prescribed:
                fem_metadata.prescribed_dofs.append(i)
            else:
                fem_metadata.free_dofs.append(i)

        vec = fem_metadata.solve_linear_system_with_gaussian_elimination_method()
        for displacement_index, node_index in enumerate(fem_metadata.free_dofs):
            self.nodes[node_index].displacement = vec[displacement_index]

        return fem_metadata


def main():
    n_nodes = 5
    fem_system = FEMSystem(n_nodes)

    # Prescribed displacements at nodes 1 and 5 (indices 0 and 4)
    fem_system.nodes[0].is_displacement_prescribed = True  # Node 1 fixed
    fem_system.nodes[4].is_displacement_prescribed = True  # Node 5 fixed

    # Applied force at node 3 (index 2)
    fem_system.nodes[2].force = 1000.0  # F3 = 1000 N

    # Initialize elements
    # Element 1: nodes 0 and 1 (nodes 1 and 2), k = 500 N/mm
    fem_system.elements.append(Element(
        id=0,
        node1_id=0,
        node2_id=1,
        stiffness=500.0
    ))

    # Element 2: nodes 1 and 3 (nodes 2 and 4), k = 400 N/mm
    fem_system.elements.append(Element(
        id=1,
        node1_id=1,
        node2_id=3,
        stiffness=400.0
    ))

    # Element 3: nodes 2 and 1 (nodes 3 and 2), k = 600 N/mm
    fem_system.elements.append(Element(
        id=2,
        node1_id=2,
        node2_id=1,
        stiffness=600.0
    ))

    # Element 4: nodes 0 and 2 (nodes 1 and 3), k = 200 N/mm
    fem_system.elements.append(Element(
        id=3,
        node1_id=0,
        node2_id=2,
        stiffness=200.0
    ))

    # Element 5: nodes 2 and 3 (nodes 3 and 4), k = 400 N/mm
    fem_system.elements.append(Element(
        id=4,
        node1_id=2,
        node2_id=3,
        stiffness=400.0
    ))

    # Element 6: nodes 3 and 4 (nodes 4 and 5), k = 300 N/mm
    fem_system.elements.append(Element(
        id=5,
        node1_id=3,
        node2_id=4,
        stiffness=300.0
    ))

    fem_metadata = fem_system.calculate_displacements_with_gauss_elimination()

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
        x = e.stiffness * delta_u
        e.force = x

    # Output results
    print("Node Displacements:")
    for node in fem_system.nodes:
        if node.displacement == 0.0 or math.isnan(node.displacement):
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
