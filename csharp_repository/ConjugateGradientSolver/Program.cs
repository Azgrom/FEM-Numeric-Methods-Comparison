using System;
using System.Collections.Generic;

class Node
{
    public int Id { get; set; }
    public double Displacement { get; set; } = 0.0;  // Unknown unless prescribed
    public double Force { get; set; } = 0.0;         // Applied force
    public bool IsDisplacementPrescribed { get; set; } = false;

    public Node(int id)
    {
        Id = id;
    }
}

class Element
{
    public int Id { get; set; }
    public int Node1Id { get; set; }  // Reference to first node
    public int Node2Id { get; set; }  // Reference to second node
    public double Stiffness { get; set; }  // Stiffness k_e
    public double Force { get; set; } = 0.0;      // Force in the element

    public Element(int id, int node1Id, int node2Id, double stiffness)
    {
        Id = id;
        Node1Id = node1Id;
        Node2Id = node2Id;
        Stiffness = stiffness;
    }
}

class FEMSystem
{
    public List<Node> Nodes { get; set; }
    public List<Element> Elements { get; set; }

    public FEMSystem(int nNodes)
    {
        Nodes = new List<Node>();
        for (int i = 0; i < nNodes; i++)
        {
            Nodes.Add(new Node(i));
        }
        Elements = new List<Element>();
    }

    public FEMMetaData CalculateDisplacements()
    {
        int nNodes = Nodes.Count;
        FEMMetaData femMetadata = new FEMMetaData(nNodes);

        // Assemble global stiffness matrix
        foreach (var e in Elements)
        {
            int i = e.Node1Id;
            int j = e.Node2Id;
            double k = e.Stiffness;

            femMetadata.StiffnessMatrix[i][i] += k;
            femMetadata.StiffnessMatrix[i][j] -= k;
            femMetadata.StiffnessMatrix[j][i] -= k;
            femMetadata.StiffnessMatrix[j][j] += k;
        }

        // Compile forces vector, free and prescribed DOFs
        for (int i = 0; i < Nodes.Count; i++)
        {
            var node = Nodes[i];
            femMetadata.ForcesVector[i] = node.Force;

            if (node.IsDisplacementPrescribed)
            {
                femMetadata.PrescribedDOFs.Add(i);
            }
            else
            {
                femMetadata.FreeDOFs.Add(i);
            }
        }

        // Solve the system using Conjugate Gradient method
        var displacements = femMetadata.SolveLinearSystemWithConjugateGradientMethod();
        for (int displacementIndex = 0; displacementIndex < femMetadata.FreeDOFs.Count; displacementIndex++)
        {
            int nodeIndex = femMetadata.FreeDOFs[displacementIndex];
            Nodes[nodeIndex].Displacement = displacements[displacementIndex];
        }

        return femMetadata;
    }
}

class FEMMetaData
{
    public List<List<double>> StiffnessMatrix { get; set; }
    public List<double> ForcesVector { get; set; }
    public List<int> FreeDOFs { get; set; }
    public List<int> PrescribedDOFs { get; set; }

    public FEMMetaData(int nNodes)
    {
        StiffnessMatrix = new List<List<double>>(nNodes);
        for (int i = 0; i < nNodes; i++)
        {
            StiffnessMatrix.Add(new List<double>(new double[nNodes]));
        }
        ForcesVector = new List<double>(new double[nNodes]);
        FreeDOFs = new List<int>();
        PrescribedDOFs = new List<int>();
    }

    public List<double> SolveLinearSystemWithConjugateGradientMethod()
    {
        // Extract the reduced stiffness matrix and force vector
        int n = FreeDOFs.Count;
        List<List<double>> K_reduced = new List<List<double>>(n);
        List<double> F_reduced = new List<double>(n);

        for (int i = 0; i < n; i++)
        {
            int rowIndex = FreeDOFs[i];
            F_reduced.Add(ForcesVector[rowIndex]);

            List<double> row = new List<double>(n);
            for (int j = 0; j < n; j++)
            {
                int colIndex = FreeDOFs[j];
                row.Add(StiffnessMatrix[rowIndex][colIndex]);
            }
            K_reduced.Add(row);
        }

        // Conjugate Gradient method
        List<double> x = new List<double>(new double[n]);  // Initial guess

        List<double> r = VectorSubtract(F_reduced, MatrixVectorMultiply(K_reduced, x));
        List<double> p = new List<double>(r);
        double rs_old = VectorDot(r, r);

        double tol = 1e-6;
        int max_iter = 1000;

        for (int iter = 0; iter < max_iter; iter++)
        {
            List<double> Ap = MatrixVectorMultiply(K_reduced, p);
            double pAp = VectorDot(p, Ap);

            if (Math.Abs(pAp) < 1e-10)
            {
                break;  // Avoid division by zero
            }

            double alpha = rs_old / pAp;
            x = VectorAdd(x, VectorScalarMultiply(p, alpha));
            r = VectorSubtract(r, VectorScalarMultiply(Ap, alpha));

            double rs_new = VectorDot(r, r);
            if (Math.Sqrt(rs_new) < tol)
            {
                break;
            }

            double beta = rs_new / rs_old;
            p = VectorAdd(r, VectorScalarMultiply(p, beta));
            rs_old = rs_new;
        }

        return x;
    }

    // Helper methods for vector and matrix operations
    private double VectorDot(List<double> a, List<double> b)
    {
        double result = 0.0;
        for (int i = 0; i < a.Count; i++)
        {
            result += a[i] * b[i];
        }
        return result;
    }

    private List<double> VectorAdd(List<double> a, List<double> b)
    {
        List<double> result = new List<double>(a.Count);
        for (int i = 0; i < a.Count; i++)
        {
            result.Add(a[i] + b[i]);
        }
        return result;
    }

    private List<double> VectorSubtract(List<double> a, List<double> b)
    {
        List<double> result = new List<double>(a.Count);
        for (int i = 0; i < a.Count; i++)
        {
            result.Add(a[i] - b[i]);
        }
        return result;
    }

    private List<double> VectorScalarMultiply(List<double> a, double scalar)
    {
        List<double> result = new List<double>(a.Count);
        for (int i = 0; i < a.Count; i++)
        {
            result.Add(a[i] * scalar);
        }
        return result;
    }

    private List<double> MatrixVectorMultiply(List<List<double>> A, List<double> x)
    {
        List<double> result = new List<double>(A.Count);
        for (int i = 0; i < A.Count; i++)
        {
            double sum = 0.0;
            for (int j = 0; j < x.Count; j++)
            {
                sum += A[i][j] * x[j];
            }
            result.Add(sum);
        }
        return result;
    }
}

class Program
{
    static void Main(string[] args)
    {
        int nNodes = 5;
        FEMSystem femSystem = new FEMSystem(nNodes);

        // Prescribed displacements at nodes 1 and 5 (indices 0 and 4)
        femSystem.Nodes[0].IsDisplacementPrescribed = true;  // Node 1 fixed
        femSystem.Nodes[4].IsDisplacementPrescribed = true;  // Node 5 fixed

        // Applied force at node 3 (index 2)
        femSystem.Nodes[2].Force = 1000.0;  // F3 = 1000 N

        // Initialize elements
        femSystem.Elements.Add(new Element(0, 0, 1, 500.0));  // Element 1
        femSystem.Elements.Add(new Element(1, 1, 3, 400.0));  // Element 2
        femSystem.Elements.Add(new Element(2, 2, 1, 600.0));  // Element 3
        femSystem.Elements.Add(new Element(3, 0, 2, 200.0));  // Element 4
        femSystem.Elements.Add(new Element(4, 2, 3, 400.0));  // Element 5
        femSystem.Elements.Add(new Element(5, 3, 4, 300.0));  // Element 6

        FEMMetaData femMetadata = femSystem.CalculateDisplacements();

        int nNodesTotal = nNodes;
        // Compute reactions at prescribed DOFs
        List<Tuple<int, double>> reactions = new List<Tuple<int, double>>();
        foreach (int i in femMetadata.PrescribedDOFs)
        {
            double Ri = 0.0;
            for (int j = 0; j < nNodesTotal; j++)
            {
                Ri += femMetadata.StiffnessMatrix[i][j] * femSystem.Nodes[j].Displacement;
            }
            Ri -= femMetadata.ForcesVector[i];
            reactions.Add(new Tuple<int, double>(i, Ri));
        }

        // Compute element forces
        foreach (var e in femSystem.Elements)
        {
            int i = e.Node1Id;
            int j = e.Node2Id;
            double ui = femSystem.Nodes[i].Displacement;
            double uj = femSystem.Nodes[j].Displacement;
            double delta_u = uj - ui;
            e.Force = e.Stiffness * delta_u;
        }

        // Output results
        Console.WriteLine("Node Displacements:");
        foreach (var node in femSystem.Nodes)
        {
            if (node.IsDisplacementPrescribed)
            {
                Console.WriteLine($"Node {node.Id + 1}: Fixed (Displacement = {node.Displacement:F4} mm)");
            }
            else
            {
                Console.WriteLine($"Node {node.Id + 1}: u = {node.Displacement:F4} mm");
            }
        }

        Console.WriteLine("\nElement Forces (Positive=Tension, Negative=Compression):");
        foreach (var e in femSystem.Elements)
        {
            Console.WriteLine($"Element {e.Id + 1}: P = {e.Force:F2} N");
        }

        Console.WriteLine("\nReactions at Prescribed Nodes:");
        foreach (var reaction in reactions)
        {
            Console.WriteLine($"Node {reaction.Item1 + 1}: R = {reaction.Item2:F2} N");
        }
    }
}
