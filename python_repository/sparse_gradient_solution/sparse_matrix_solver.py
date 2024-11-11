class Node:
    def __init__(self, id):
        self.id = id
        self.displacement = 0.0  # Unknown unless prescribed
        self.force = 0.0  # Applied force
        self.is_displacement_prescribed = False


class Element:
    def __init__(self, id, node1_id, node2_id, stiffness):
        self.id = id
        self.node1_id = node1_id
        self.node2_id = node2_id
        self.stiffness = stiffness
        self.force = 0.0  # Force in the element


class FEMSystem:
    def __init__(self, n_nodes):
        self.nodes = [Node(i) for i in range(n_nodes)]
        self.elements = []

    def calculate_displacements(self):
        n_nodes = len(self.nodes)
        fem_metadata = FEMMetaData(n_nodes)

        # Assemble stiffness matrix in COO format
        for e in self.elements:
            i = e.node1_id
            j = e.node2_id
            k = e.stiffness

            # K[i][i] += k
            fem_metadata.stiffness_matrix.add_entry(i, i, k)
            # K[i][j] -= k
            fem_metadata.stiffness_matrix.add_entry(i, j, -k)
            # K[j][i] -= k
            fem_metadata.stiffness_matrix.add_entry(j, i, -k)
            # K[j][j] += k
            fem_metadata.stiffness_matrix.add_entry(j, j, k)

        # Compile forces vector, free and prescribed DOFs
        for i, node in enumerate(self.nodes):
            fem_metadata.forces_vector.append(node.force)

            if node.is_displacement_prescribed:
                fem_metadata.prescribed_dofs.append(i)
            else:
                fem_metadata.free_dofs.append(i)

        # Convert stiffness matrix to CSR format
        fem_metadata.stiffness_matrix.to_csr()

        # Solve using the sparse matrix method
        vec = fem_metadata.solve_linear_system_with_sparse_matrix_method()

        # Assign displacements to nodes
        for displacement_index, node_index in enumerate(fem_metadata.free_dofs):
            self.nodes[node_index].displacement = vec[displacement_index]

        return fem_metadata


class FEMMetaData:
    def __init__(self, nodes_count):
        self.stiffness_matrix = SparseMatrixCSR(nodes_count)
        self.forces_vector = []
        self.free_dofs = []
        self.prescribed_dofs = []

    def solve_linear_system_with_sparse_matrix_method(self):
        # Prepare reduced stiffness matrix and force vector
        n_free = len(self.free_dofs)

        # Adjust the force vector for prescribed displacements
        F_reduced = []
        for i in self.free_dofs:
            Fi = self.forces_vector[i]
            for j in self.prescribed_dofs:
                K_ij = self.stiffness_matrix.get_value(i, j)
                uj = 0.0  # Prescribed displacement is zero
                Fi -= K_ij * uj
            F_reduced.append(Fi)

        # Extract reduced stiffness matrix
        reduced_matrix = self.stiffness_matrix.extract_reduced_matrix(self.free_dofs)

        # Solve using Conjugate Gradient method
        tol = 1e-6
        max_iter = 1000
        x = conjugate_gradient(reduced_matrix, F_reduced, tol, max_iter)

        return x


class COOEntry:
    def __init__(self, row, col, value):
        self.row = row
        self.col = col
        self.value = value


class SparseMatrixCSR:
    def __init__(self, size):
        self.nrows = size
        self.ncols = size
        self.coo_entries = []
        self.values = []
        self.col_indices = []
        self.row_pointers = []

    def add_entry(self, row, col, value):
        self.coo_entries.append(COOEntry(row, col, value))

    def to_csr(self):
        # Combine duplicate entries
        entry_map = {}
        for entry in self.coo_entries:
            key = (entry.row, entry.col)
            if key in entry_map:
                entry_map[key] += entry.value
            else:
                entry_map[key] = entry.value

        # Sort entries
        entries = [COOEntry(row, col, value) for (row, col), value in entry_map.items()]
        entries.sort(key=lambda x: (x.row, x.col))

        # Build CSR format
        self.values = []
        self.col_indices = []
        self.row_pointers = [0] * (self.nrows + 1)

        current_row = 0
        idx = 0
        for entry in entries:
            while current_row < entry.row:
                self.row_pointers[current_row + 1] = idx
                current_row += 1
            self.values.append(entry.value)
            self.col_indices.append(entry.col)
            idx += 1

        # Complete row pointers
        for row in range(current_row, self.nrows):
            self.row_pointers[row + 1] = idx

    def get_value(self, row, col):
        start = self.row_pointers[row]
        end = self.row_pointers[row + 1]
        for idx in range(start, end):
            if self.col_indices[idx] == col:
                return self.values[idx]
        return 0.0

    def extract_reduced_matrix(self, free_dofs):
        n = len(free_dofs)
        reduced_matrix = SparseMatrixCSR(n)

        dof_map = {dof: idx for idx, dof in enumerate(free_dofs)}

        for row_global in free_dofs:
            row_local = dof_map[row_global]
            start = self.row_pointers[row_global]
            end = self.row_pointers[row_global + 1]
            for idx in range(start, end):
                col_global = self.col_indices[idx]
                if col_global in dof_map:
                    col_local = dof_map[col_global]
                    reduced_matrix.add_entry(row_local, col_local, self.values[idx])

        reduced_matrix.to_csr()
        return reduced_matrix


def conjugate_gradient(matrix, b, tol, max_iter):
    n = len(b)
    x = [0.0] * n
    r = b.copy()
    p = r.copy()
    rsold = dot_product(r, r)

    for _ in range(max_iter):
        Ap = csr_matvec_mul(matrix, p)
        pAp = dot_product(p, Ap)
        if pAp == 0.0:
            break
        alpha = rsold / pAp
        for i in range(n):
            x[i] += alpha * p[i]
            r[i] -= alpha * Ap[i]
        rsnew = dot_product(r, r)
        if rsnew ** 0.5 < tol:
            break
        beta = rsnew / rsold
        for i in range(n):
            p[i] = r[i] + beta * p[i]
        rsold = rsnew
    return x


def csr_matvec_mul(matrix, x):
    y = [0.0] * matrix.nrows
    for row in range(matrix.nrows):
        start = matrix.row_pointers[row]
        end = matrix.row_pointers[row + 1]
        for idx in range(start, end):
            col = matrix.col_indices[idx]
            y[row] += matrix.values[idx] * x[col]
    return y


def dot_product(a, b):
    return sum(ai * bi for ai, bi in zip(a, b))


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

    fem_metadata = fem_system.calculate_displacements()

    # Compute reactions at prescribed DOFs
    reactions = []
    for i in fem_metadata.prescribed_dofs:
        Ri = 0.0
        for j in range(n_nodes):
            Ri += fem_metadata.stiffness_matrix.get_value(i, j) * fem_system.nodes[j].displacement
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
            print("Node {}: Fixed (Displacement = {:.4f} mm)".format(node.id + 1, node.displacement))
        else:
            print("Node {}: u = {:.4f} mm".format(node.id + 1, node.displacement))

    print("\nElement Forces (Positive=Tension, Negative=Compression):")
    for e in fem_system.elements:
        print("Element {}: P = {:.2f} N".format(e.id + 1, e.force))

    print("\nReactions at Prescribed Nodes:")
    for node_id, reaction in reactions:
        print("Node {}: R = {:.2f} N".format(node_id + 1, reaction))


if __name__ == "__main__":
    main()
