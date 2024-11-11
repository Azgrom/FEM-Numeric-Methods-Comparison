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

    let fem_metadata = fem_system.calculate_displacements_with_gauss_elimination();

    // Compute reactions at prescribed DOFs
    let mut reactions = Vec::new();
    for &i in &fem_metadata.prescribed_dofs {
        let mut Ri = 0.0;
        for j in 0..n_nodes {
            Ri += fem_metadata.stiffness_matrix[i][j] * fem_system.nodes[j].displacement;
        }
        Ri -= fem_metadata.forces_vector[i];
        reactions.push((i, Ri));
    }

    // Compute element forces
    for e in fem_system.elements.iter_mut() {
        let i = e.node1_id;
        let j = e.node2_id;
        let ui = fem_system.nodes[i].displacement;
        let uj = fem_system.nodes[j].displacement;
        let delta_u = uj - ui;
        let x = (e.stiffness * delta_u);
        e.force = x;
    }

    // Output results
    println!("Node Displacements:");
    for node in &fem_system.nodes {
        if node.displacement == 0f64 || node.displacement.is_nan() {
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
            nodes.push(
                (Node {
                    id: i,
                    displacement: 0.0,
                    force: 0.0,
                    is_displacement_prescribed: false,
                }),
            );
        }
        Self {
            nodes,
            elements: Vec::new(),
        }
    }

    fn calculate_displacements_with_gauss_elimination(&mut self) -> FEMMetaData {
        let n_nodes = self.nodes.len();
        let mut fem_metadata = FEMMetaData::new(n_nodes);

        for e in &self.elements {
            let i = e.node1_id;
            let j = e.node2_id;
            let k = e.stiffness;

            fem_metadata.stiffness_matrix[i][i] += k;
            fem_metadata.stiffness_matrix[i][j] -= k;
            fem_metadata.stiffness_matrix[j][i] -= k;
            fem_metadata.stiffness_matrix[j][j] += k;
        }

        // compiles forces vector, free and prescribed dofs
        for (i, node) in self.nodes.iter().enumerate() {
            fem_metadata.forces_vector.push(node.force);

            if node.is_displacement_prescribed {
                fem_metadata.prescribed_dofs.push(i);
            } else {
                fem_metadata.free_dofs.push(i)
            }
        }

        let vec = fem_metadata.solve_linear_system_with_gaussian_elimination_method();
        for (displacement_index, &node_index) in fem_metadata.free_dofs.iter().enumerate() {
            self.nodes[node_index].displacement = vec[displacement_index];
        }

        fem_metadata
    }

    fn calculate_displacements_with_conjugate_gradient(&mut self) -> FEMMetaData {
        let n_nodes = self.nodes.len();
        let mut fem_metadata = FEMMetaData::new(n_nodes);

        // Assemble global stiffness matrix
        for e in &self.elements {
            let i = e.node1_id;
            let j = e.node2_id;
            let k = e.stiffness;

            fem_metadata.stiffness_matrix[i][i] += k;
            fem_metadata.stiffness_matrix[i][j] -= k;
            fem_metadata.stiffness_matrix[j][i] -= k;
            fem_metadata.stiffness_matrix[j][j] += k;
        }

        // Compile forces vector, free and prescribed DOFs
        for (i, node) in self.nodes.iter().enumerate() {
            fem_metadata.forces_vector.push(node.force);

            if node.is_displacement_prescribed {
                fem_metadata.prescribed_dofs.push(i);
            } else {
                fem_metadata.free_dofs.push(i);
            }
        }

        // Solve the system using Conjugate Gradient method
        let vec = fem_metadata.solve_linear_system_with_gaussian_elimination_method();
        for (displacement_index, &node_index) in fem_metadata.free_dofs.iter().enumerate() {
            self.nodes[node_index].displacement = vec[displacement_index];
        }

        fem_metadata
    }
}

struct FEMMetaData {
    stiffness_matrix: Vec<Vec<f64>>,
    forces_vector: Vec<f64>,
    free_dofs: Vec<usize>,
    prescribed_dofs: Vec<usize>,
}

impl FEMMetaData {
    fn new(nodes_count: usize) -> Self {
        let base_usize_vec = Vec::<usize>::with_capacity(nodes_count);

        Self {
            stiffness_matrix: vec![vec![0f64; nodes_count]; nodes_count],
            forces_vector: Vec::with_capacity(nodes_count),
            free_dofs: base_usize_vec.clone(),
            prescribed_dofs: base_usize_vec,
        }
    }

    fn solve_linear_system_with_gaussian_elimination_method(&self) -> Vec<f64> {
        let free_dofs_len = self.free_dofs.len();
        let mut augmented_matrix = vec![vec![0f64; free_dofs_len + 1]; free_dofs_len];

        for (i, free_dof_i) in self.free_dofs.iter().enumerate() {
            augmented_matrix[i][free_dofs_len] = self.forces_vector[*free_dof_i];

            for (j, free_dof_j) in self.free_dofs.iter().enumerate() {
                augmented_matrix[i][j] = self.stiffness_matrix[*free_dof_i][*free_dof_j];
            }
        }

        for k in 0..free_dofs_len {
            let mut max = augmented_matrix[k][k].abs();
            let mut max_row = k;

            for i in k + 1..free_dofs_len {
                if augmented_matrix[i][k].abs() > max {
                    max = augmented_matrix[i][k].abs();
                    max_row = i;
                }
            }

            if max_row != k {
                augmented_matrix.swap(k, max_row);
            }

            for i in k + 1..free_dofs_len {
                let factor = augmented_matrix[i][k] / augmented_matrix[k][k];
                for j in k..=free_dofs_len {
                    augmented_matrix[i][j] -= factor * augmented_matrix[k][j];
                }
            }
        }

        let mut reactions = vec![0f64; free_dofs_len];
        for i in (0..free_dofs_len).rev() {
            reactions[i] = augmented_matrix[i][free_dofs_len];

            for j in i + 1..free_dofs_len {
                reactions[i] -= augmented_matrix[i][j] * reactions[j];
            }

            reactions[i] /= augmented_matrix[i][i];
        }

        reactions
    }
}

