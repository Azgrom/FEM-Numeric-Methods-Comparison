use std::collections::HashMap;

fn main() {
    let n_nodes = 5;
    let mut fem_system = FEMSystem::new(n_nodes);

    // Prescribed displacements at nodes 1 and 5 (indices 0 and 4)
    fem_system.nodes[0].is_displacement_prescribed = true; // Node 1 fixed
    fem_system.nodes[4].is_displacement_prescribed = true; // Node 5 fixed

    // Applied force at node 3 (index 2)
    fem_system.nodes[2].force = 1000.0; // F3 = 1000 N

    // Initialize elements
    // Element 1: nodes 0 and 1 (nodes 1 and 2), k = 500 N/mm
    fem_system.elements.push(Element {
        id: 0,
        node1_id: 0,
        node2_id: 1,
        stiffness: 500.0,
        force: 0.0,
    });

    // Element 2: nodes 1 and 3 (nodes 2 and 4), k = 400 N/mm
    fem_system.elements.push(Element {
        id: 1,
        node1_id: 1,
        node2_id: 3,
        stiffness: 400.0,
        force: 0.0,
    });

    // Element 3: nodes 2 and 1 (nodes 3 and 2), k = 600 N/mm
    fem_system.elements.push(Element {
        id: 2,
        node1_id: 2,
        node2_id: 1,
        stiffness: 600.0,
        force: 0.0,
    });

    // Element 4: nodes 0 and 2 (nodes 1 and 3), k = 200 N/mm
    fem_system.elements.push(Element {
        id: 3,
        node1_id: 0,
        node2_id: 2,
        stiffness: 200.0,
        force: 0.0,
    });

    // Element 5: nodes 2 and 3 (nodes 3 and 4), k = 400 N/mm
    fem_system.elements.push(Element {
        id: 4,
        node1_id: 2,
        node2_id: 3,
        stiffness: 400.0,
        force: 0.0,
    });

    // Element 6: nodes 3 and 4 (nodes 4 and 5), k = 300 N/mm
    fem_system.elements.push(Element {
        id: 5,
        node1_id: 3,
        node2_id: 4,
        stiffness: 300.0,
        force: 0.0,
    });

    let fem_metadata = fem_system.calculate_displacements();

    // Compute reactions at prescribed DOFs
    let mut reactions = Vec::new();
    for i in &fem_metadata.prescribed_dofs {
        let mut Ri = 0.0;
        for j in 0..n_nodes {
            Ri += fem_metadata.stiffness_matrix.get_value(*i, j) * fem_system.nodes[j].displacement;
        }
        Ri -= fem_metadata.forces_vector[*i];
        reactions.push((i, Ri));
    }

    // Compute element forces
    for e in fem_system.elements.iter_mut() {
        let i = e.node1_id;
        let j = e.node2_id;
        let ui = fem_system.nodes[i].displacement;
        let uj = fem_system.nodes[j].displacement;
        let delta_u = uj - ui;
        let x = e.stiffness * delta_u;
        e.force = x;
    }

    // Output results
    println!("Node Displacements:");
    for node in &fem_system.nodes {
        if node.is_displacement_prescribed {
            println!(
                "Node {}: Fixed (Displacement = {:.4} mm)",
                node.id + 1,
                node.displacement
            )
        } else {
            println!("Node {}: u = {:.4} mm", node.id + 1, node.displacement);
        }
    }

    println!("\nElement Forces (Positive=Tension, Negative=Compression):");
    for e in fem_system.elements {
        println!("Element {}: P = {:.2} N", e.id + 1, e.force);
    }

    println!("\nReactions at Prescribed Nodes:");
    for &(node_id, reaction) in &reactions {
        println!("Node {}: R = {:.2} N", node_id + 1, reaction);
    }
}

struct Node {
    id: usize,
    displacement: f64, // Unknown unless prescribed
    force: f64,        // Applied force
    is_displacement_prescribed: bool,
}

struct Element {
    id: usize,
    node1_id: usize, // Reference to first node
    node2_id: usize, // Reference to second node
    stiffness: f64,  // Stiffness k_e
    force: f64,      // Force in the element
}

struct FEMSystem {
    nodes: Vec<Node>,
    elements: Vec<Element>,
}

impl FEMSystem {
    // Method to initialize the system
    fn new(n_nodes: usize) -> Self {
        let mut nodes = Vec::new();
        for i in 0..n_nodes {
            nodes.push(Node {
                id: i,
                displacement: 0.0,
                force: 0.0,
                is_displacement_prescribed: false,
            });
        }
        Self {
            nodes,
            elements: Vec::new(),
        }
    }

    fn calculate_displacements(&mut self) -> FEMMetaData {
        let n_nodes = self.nodes.len();
        let mut fem_metadata = FEMMetaData::new(n_nodes);

        // Assemble stiffness matrix in COO format
        for e in &self.elements {
            let i = e.node1_id;
            let j = e.node2_id;
            let k = e.stiffness;

            // K[i][i] += k
            fem_metadata.stiffness_matrix.add_entry(i, i, k);
            // K[i][j] -= k
            fem_metadata.stiffness_matrix.add_entry(i, j, -k);
            // K[j][i] -= k
            fem_metadata.stiffness_matrix.add_entry(j, i, -k);
            // K[j][j] += k
            fem_metadata.stiffness_matrix.add_entry(j, j, k);
        }

        // Compile forces vector, free and prescribed DOFs
        for (i, node) in self.nodes.iter().enumerate() {
            fem_metadata.forces_vector.push(node.force);

            if node.is_displacement_prescribed {
                fem_metadata.prescribed_dofs.push(i);
            } else {
                fem_metadata.free_dofs.push(i)
            }
        }

        // Convert stiffness matrix to CSR format
        fem_metadata.stiffness_matrix.to_csr();

        // Solve using the sparse matrix method
        let vec = fem_metadata.solve_linear_system_with_sparse_matrix_method();

        // Assign displacements to nodes
        for (displacement_index, &node_index) in fem_metadata.free_dofs.iter().enumerate() {
            self.nodes[node_index].displacement = vec[displacement_index];
        }

        fem_metadata
    }
}

struct FEMMetaData {
    stiffness_matrix: SparseMatrixCSR,
    forces_vector: Vec<f64>,
    free_dofs: Vec<usize>,
    prescribed_dofs: Vec<usize>,
}

impl FEMMetaData {
    fn new(nodes_count: usize) -> Self {
        Self {
            stiffness_matrix: SparseMatrixCSR::new(nodes_count),
            forces_vector: Vec::with_capacity(nodes_count),
            free_dofs: Vec::with_capacity(nodes_count),
            prescribed_dofs: Vec::with_capacity(nodes_count),
        }
    }

    fn solve_linear_system_with_sparse_matrix_method(&mut self) -> Vec<f64> {
        // Prepare reduced stiffness matrix and force vector
        let n_free = self.free_dofs.len();

        // Adjust the force vector for prescribed displacements
        let mut F_reduced = Vec::with_capacity(n_free);
        for &i in &self.free_dofs {
            let mut Fi = self.forces_vector[i];
            for &j in &self.prescribed_dofs {
                let K_ij = self.stiffness_matrix.get_value(i, j);
                let uj = 0.0; // Prescribed displacement is zero
                Fi -= K_ij * uj;
            }
            F_reduced.push(Fi);
        }

        // Extract reduced stiffness matrix
        let reduced_matrix = self
            .stiffness_matrix
            .extract_reduced_matrix(&self.free_dofs);

        // Solve using Conjugate Gradient method
        let tol = 1e-6;
        let max_iter = 1000;
        let x = conjugate_gradient(&reduced_matrix, &F_reduced, tol, max_iter);

        x
    }
}

// Sparse Matrix in CSR format
struct SparseMatrixCSR {
    nrows: usize,
    ncols: usize,
    coo_entries: Vec<COOEntry>,
    values: Vec<f64>,
    col_indices: Vec<usize>,
    row_pointers: Vec<usize>,
}

impl SparseMatrixCSR {
    fn new(size: usize) -> Self {
        Self {
            nrows: size,
            ncols: size,
            coo_entries: Vec::new(),
            values: Vec::new(),
            col_indices: Vec::new(),
            row_pointers: Vec::new(),
        }
    }

    fn add_entry(&mut self, row: usize, col: usize, value: f64) {
        self.coo_entries.push(COOEntry { row, col, value });
    }

    fn to_csr(&mut self) {
        // Combine duplicate entries
        let mut entry_map = HashMap::new();
        for entry in &self.coo_entries {
            let key = (entry.row, entry.col);
            let val = entry_map.entry(key).or_insert(0.0);
            *val += entry.value;
        }

        // Sort entries
        let mut entries: Vec<COOEntry> = entry_map
            .into_iter()
            .map(|((row, col), value)| COOEntry { row, col, value })
            .collect();

        entries.sort_by(|a, b| {
            if a.row == b.row {
                a.col.cmp(&b.col)
            } else {
                a.row.cmp(&b.row)
            }
        });

        // Build CSR format
        self.values = Vec::with_capacity(entries.len());
        self.col_indices = Vec::with_capacity(entries.len());
        self.row_pointers = vec![0; self.nrows + 1];

        let mut current_row = 0;
        for entry in entries {
            while current_row < entry.row {
                self.row_pointers[current_row + 1] = self.values.len();
                current_row += 1;
            }
            self.values.push(entry.value);
            self.col_indices.push(entry.col);
        }

        // Complete row pointers
        for row in current_row..self.nrows {
            self.row_pointers[row + 1] = self.values.len();
        }
    }

    fn get_value(&self, row: usize, col: usize) -> f64 {
        let start = self.row_pointers[row];
        let end = self.row_pointers[row + 1];
        for idx in start..end {
            if self.col_indices[idx] == col {
                return self.values[idx];
            }
        }
        0.0
    }

    fn extract_reduced_matrix(&self, free_dofs: &Vec<usize>) -> SparseMatrixCSR {
        let n = free_dofs.len();
        let mut reduced_matrix = SparseMatrixCSR::new(n);

        let dof_map: HashMap<usize, usize> = free_dofs
            .iter()
            .enumerate()
            .map(|(idx, &dof)| (dof, idx))
            .collect();

        for &row_global in free_dofs {
            let row_local = dof_map[&row_global];
            let start = self.row_pointers[row_global];
            let end = self.row_pointers[row_global + 1];
            for idx in start..end {
                let col_global = self.col_indices[idx];
                if let Some(&col_local) = dof_map.get(&col_global) {
                    reduced_matrix.add_entry(row_local, col_local, self.values[idx]);
                }
            }
        }

        reduced_matrix.to_csr();
        reduced_matrix
    }
}

struct COOEntry {
    row: usize,
    col: usize,
    value: f64,
}

// Conjugate Gradient Solver
fn conjugate_gradient(
    matrix: &SparseMatrixCSR,
    b: &Vec<f64>,
    tol: f64,
    max_iter: usize,
) -> Vec<f64> {
    let n = b.len();
    let mut x = vec![0.0; n];
    let mut r = b.clone();
    let mut p = r.clone();
    let mut rsold = dot_product(&r, &r);

    for _ in 0..max_iter {
        let Ap = csr_matvec_mul(matrix, &p);
        let alpha = rsold / dot_product(&p, &Ap);
        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * Ap[i];
        }
        let rsnew = dot_product(&r, &r);
        if rsnew.sqrt() < tol {
            break;
        }
        for i in 0..n {
            p[i] = r[i] + (rsnew / rsold) * p[i];
        }
        rsold = rsnew;
    }
    x
}

fn csr_matvec_mul(matrix: &SparseMatrixCSR, x: &Vec<f64>) -> Vec<f64> {
    let mut y = vec![0.0; matrix.nrows];
    for row in 0..matrix.nrows {
        let start = matrix.row_pointers[row];
        let end = matrix.row_pointers[row + 1];
        for idx in start..end {
            let col = matrix.col_indices[idx];
            y[row] += matrix.values[idx] * x[col];
        }
    }
    y
}

fn dot_product(a: &Vec<f64>, b: &Vec<f64>) -> f64 {
    a.iter().zip(b.iter()).map(|(ai, bi)| ai * bi).sum()
}
