using System;
using System.Collections.Generic;

class Node
{
    public int id;
    public double displacement; // Unknown unless prescribed
    public double force;        // Applied force
    public bool is_displacement_prescribed;

    public Node(int id)
    {
        this.id = id;
        this.displacement = 0.0;
        this.force = 0.0;
        this.is_displacement_prescribed = false;
    }
}

class Element
{
    public int id;
    public int node1_id; // Reference to first node
    public int node2_id; // Reference to second node
    public double stiffness;  // Stiffness k_e
    public double force;      // Force in the element

    public Element(int id, int node1_id, int node2_id, double stiffness)
    {
        this.id = id;
        this.node1_id = node1_id;
        this.node2_id = node2_id;
        this.stiffness = stiffness;
        this.force = 0.0;
    }
}

class FEMMetaData
{
    public double[][] stiffness_matrix;
    public List<double> forces_vector;
    public List<int> free_dofs;
    public List<int> prescribed_dofs;

    public FEMMetaData(int nodes_count)
    {
        stiffness_matrix = new double[nodes_count][];
        for (int i = 0; i < nodes_count; i++)
        {
            stiffness_matrix[i] = new double[nodes_count];
        }
        forces_vector = new List<double>(nodes_count);
        free_dofs = new List<int>();
        prescribed_dofs = new List<int>();
    }

    public List<double> solve_linear_system_with_gaussian_elimination_method()
    {
        int free_dofs_len = free_dofs.Count;
        double[][] augmented_matrix = new double[free_dofs_len][];
        for (int i = 0; i < free_dofs_len; i++)
        {
            augmented_matrix[i] = new double[free_dofs_len + 1];
        }

        for (int i = 0; i < free_dofs_len; i++)
        {
            int free_dof_i = free_dofs[i];
            augmented_matrix[i][free_dofs_len] = forces_vector[free_dof_i];

            for (int j = 0; j < free_dofs_len; j++)
            {
                int free_dof_j = free_dofs[j];
                augmented_matrix[i][j] = stiffness_matrix[free_dof_i][free_dof_j];
            }
        }

        // Gaussian elimination with partial pivoting
        for (int k = 0; k < free_dofs_len; k++)
        {
            // Find the k-th pivot
            int max_row = k;
            double max_value = Math.Abs(augmented_matrix[k][k]);
            for (int i = k + 1; i < free_dofs_len; i++)
            {
                if (Math.Abs(augmented_matrix[i][k]) > max_value)
                {
                    max_value = Math.Abs(augmented_matrix[i][k]);
                    max_row = i;
                }
            }

            if (max_value == 0)
            {
                throw new Exception("Matrix is singular!");
            }

            if (max_row != k)
            {
                // Swap rows k and max_row
                var temp = augmented_matrix[k];
                augmented_matrix[k] = augmented_matrix[max_row];
                augmented_matrix[max_row] = temp;
            }

            // Eliminate entries below pivot
            for (int i = k + 1; i < free_dofs_len; i++)
            {
                double factor = augmented_matrix[i][k] / augmented_matrix[k][k];
                for (int j = k; j <= free_dofs_len; j++)
                {
                    augmented_matrix[i][j] -= factor * augmented_matrix[k][j];
                }
            }
        }

        // Back substitution
        List<double> reactions = new List<double>(new double[free_dofs_len]);
        for (int i = free_dofs_len - 1; i >= 0; i--)
        {
            reactions[i] = augmented_matrix[i][free_dofs_len];
            for (int j = i + 1; j < free_dofs_len; j++)
            {
                reactions[i] -= augmented_matrix[i][j] * reactions[j];
            }
            reactions[i] /= augmented_matrix[i][i];
        }

        return reactions;
    }
}

class FEMSystem
{
    public List<Node> nodes;
    public List<Element> elements;

    public FEMSystem(int n_nodes)
    {
        nodes = new List<Node>();
        for (int i = 0; i < n_nodes; i++)
        {
            nodes.Add(new Node(i));
        }
        elements = new List<Element>();
    }

    public FEMMetaData calculate_displacements_with_gauss_elimination()
    {
        int n_nodes = nodes.Count;
        FEMMetaData fem_metadata = new FEMMetaData(n_nodes);

        foreach (var e in elements)
        {
            int i = e.node1_id;
            int j = e.node2_id;
            double k = e.stiffness;

            fem_metadata.stiffness_matrix[i][i] += k;
            fem_metadata.stiffness_matrix[i][j] -= k;
            fem_metadata.stiffness_matrix[j][i] -= k;
            fem_metadata.stiffness_matrix[j][j] += k;
        }

        // Compile forces vector, free and prescribed DOFs
        for (int i = 0; i < nodes.Count; i++)
        {
            Node node = nodes[i];
            fem_metadata.forces_vector.Add(node.force);

            if (node.is_displacement_prescribed)
            {
                fem_metadata.prescribed_dofs.Add(i);
            }
            else
            {
                fem_metadata.free_dofs.Add(i);
            }
        }

        List<double> vec = fem_metadata.solve_linear_system_with_gaussian_elimination_method();

        for (int displacement_index = 0; displacement_index < fem_metadata.free_dofs.Count; displacement_index++)
        {
            int node_index = fem_metadata.free_dofs[displacement_index];
            nodes[node_index].displacement = vec[displacement_index];
        }

        return fem_metadata;
    }
}

class Program
{
    static void Main(string[] args)
    {
        int n_nodes = 5;
        FEMSystem fem_system = new FEMSystem(n_nodes);

        // Prescribed displacements at nodes 1 and 5 (indices 0 and 4)
        fem_system.nodes[0].is_displacement_prescribed = true; // Node 1 fixed
        fem_system.nodes[4].is_displacement_prescribed = true; // Node 5 fixed

        // Applied force at node 3 (index 2)
        fem_system.nodes[2].force = 1000.0; // F3 = 1000 N

        // Initialize elements
        // Element 1: nodes 0 and 1 (nodes 1 and 2), k = 500 N/mm
        fem_system.elements.Add(new Element(
            id: 0,
            node1_id: 0,
            node2_id: 1,
            stiffness: 500.0
        ));

        // Element 2: nodes 1 and 3 (nodes 2 and 4), k = 400 N/mm
        fem_system.elements.Add(new Element(
            id: 1,
            node1_id: 1,
            node2_id: 3,
            stiffness: 400.0
        ));

        // Element 3: nodes 2 and 1 (nodes 3 and 2), k = 600 N/mm
        fem_system.elements.Add(new Element(
            id: 2,
            node1_id: 2,
            node2_id: 1,
            stiffness: 600.0
        ));

        // Element 4: nodes 0 and 2 (nodes 1 and 3), k = 200 N/mm
        fem_system.elements.Add(new Element(
            id: 3,
            node1_id: 0,
            node2_id: 2,
            stiffness: 200.0
        ));

        // Element 5: nodes 2 and 3 (nodes 3 and 4), k = 400 N/mm
        fem_system.elements.Add(new Element(
            id: 4,
            node1_id: 2,
            node2_id: 3,
            stiffness: 400.0
        ));

        // Element 6: nodes 3 and 4 (nodes 4 and 5), k = 300 N/mm
        fem_system.elements.Add(new Element(
            id: 5,
            node1_id: 3,
            node2_id: 4,
            stiffness: 300.0
        ));

        FEMMetaData fem_metadata = fem_system.calculate_displacements_with_gauss_elimination();

        // Compute reactions at prescribed DOFs
        List<(int, double)> reactions = new List<(int, double)>();
        foreach (int i in fem_metadata.prescribed_dofs)
        {
            double Ri = 0.0;
            for (int j = 0; j < n_nodes; j++)
            {
                Ri += fem_metadata.stiffness_matrix[i][j] * fem_system.nodes[j].displacement;
            }
            Ri -= fem_metadata.forces_vector[i];
            reactions.Add((i, Ri));
        }

        // Compute element forces
        foreach (var e in fem_system.elements)
        {
            int i = e.node1_id;
            int j = e.node2_id;
            double ui = fem_system.nodes[i].displacement;
            double uj = fem_system.nodes[j].displacement;
            double delta_u = uj - ui;
            double x = e.stiffness * delta_u;
            e.force = x;
        }

        // Output results
        Console.WriteLine("Node Displacements:");
        foreach (var node in fem_system.nodes)
        {
            if (node.displacement == 0.0 || double.IsNaN(node.displacement))
            {
                Console.WriteLine($"Node {node.id + 1}: Fixed (Displacement = {node.displacement:F4} mm)");
            }
            else
            {
                Console.WriteLine($"Node {node.id + 1}: u = {node.displacement:F4} mm");
            }
        }

        Console.WriteLine("\nElement Forces (Positive=Tension, Negative=Compression):");
        foreach (var e in fem_system.elements)
        {
            Console.WriteLine($"Element {e.id + 1}: P = {e.force:F2} N");
        }

        Console.WriteLine("\nReactions at Prescribed Nodes:");
        foreach (var reaction in reactions)
        {
            Console.WriteLine($"Node {reaction.Item1 + 1}: R = {reaction.Item2:F2} N");
        }
    }
}
