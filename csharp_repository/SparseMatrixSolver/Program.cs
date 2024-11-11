using System;
using System.Collections.Generic;

class Node
{
    public int Id;
    public double Displacement; // Unknown unless prescribed
    public double Force;        // Applied force
    public bool IsDisplacementPrescribed;

    public Node(int id)
    {
        Id = id;
        Displacement = 0.0;
        Force = 0.0;
        IsDisplacementPrescribed = false;
    }
}

class Element
{
    public int Id;
    public int Node1Id; // Reference to first node
    public int Node2Id; // Reference to second node
    public double Stiffness;  // Stiffness k_e
    public double Force;      // Force in the element

    public Element(int id, int node1Id, int node2Id, double stiffness)
    {
        Id = id;
        Node1Id = node1Id;
        Node2Id = node2Id;
        Stiffness = stiffness;
        Force = 0.0;
    }
}

class FEMSystem
{
    public List<Node> Nodes;
    public List<Element> Elements;

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

        // Assemble stiffness matrix in COO format
        foreach (var e in Elements)
        {
            int i = e.Node1Id;
            int j = e.Node2Id;
            double k = e.Stiffness;

            // K[i][i] += k
            femMetadata.StiffnessMatrix.AddEntry(i, i, k);
            // K[i][j] -= k
            femMetadata.StiffnessMatrix.AddEntry(i, j, -k);
            // K[j][i] -= k
            femMetadata.StiffnessMatrix.AddEntry(j, i, -k);
            // K[j][j] += k
            femMetadata.StiffnessMatrix.AddEntry(j, j, k);
        }

        // Compile forces vector, free and prescribed DOFs
        for (int i = 0; i < Nodes.Count; i++)
        {
            var node = Nodes[i];
            femMetadata.ForcesVector.Add(node.Force);

            if (node.IsDisplacementPrescribed)
            {
                femMetadata.PrescribedDofs.Add(i);
            }
            else
            {
                femMetadata.FreeDofs.Add(i);
            }
        }

        // Convert stiffness matrix to CSR format
        femMetadata.StiffnessMatrix.ToCSR();

        // Solve using the sparse matrix method
        var vec = femMetadata.SolveLinearSystemWithSparseMatrixMethod();

        // Assign displacements to nodes
        for (int displacementIndex = 0; displacementIndex < femMetadata.FreeDofs.Count; displacementIndex++)
        {
            int nodeIndex = femMetadata.FreeDofs[displacementIndex];
            Nodes[nodeIndex].Displacement = vec[displacementIndex];
        }

        return femMetadata;
    }
}

class FEMMetaData
{
    public SparseMatrixCSR StiffnessMatrix;
    public List<double> ForcesVector;
    public List<int> FreeDofs;
    public List<int> PrescribedDofs;

    public FEMMetaData(int nodesCount)
    {
        StiffnessMatrix = new SparseMatrixCSR(nodesCount);
        ForcesVector = new List<double>();
        FreeDofs = new List<int>();
        PrescribedDofs = new List<int>();
    }

    public List<double> SolveLinearSystemWithSparseMatrixMethod()
    {
        // Prepare reduced stiffness matrix and force vector
        int nFree = FreeDofs.Count;

        // Adjust the force vector for prescribed displacements
        var F_reduced = new List<double>(nFree);
        foreach (var i in FreeDofs)
        {
            double Fi = ForcesVector[i];
            foreach (var j in PrescribedDofs)
            {
                double K_ij = StiffnessMatrix.GetValue(i, j);
                double uj = 0.0; // Prescribed displacement is zero
                Fi -= K_ij * uj;
            }
            F_reduced.Add(Fi);
        }

        // Extract reduced stiffness matrix
        var reducedMatrix = StiffnessMatrix.ExtractReducedMatrix(FreeDofs);

        // Solve using Conjugate Gradient method
        double tol = 1e-6;
        int maxIter = 1000;
        var x = ConjugateGradient(reducedMatrix, F_reduced, tol, maxIter);

        return x;
    }

    // Conjugate Gradient Solver
    public static List<double> ConjugateGradient(SparseMatrixCSR matrix, List<double> b, double tol, int maxIter)
    {
        int n = b.Count;
        var x = new List<double>(new double[n]);
        var r = new List<double>(b);
        var p = new List<double>(r);
        double rsold = DotProduct(r, r);

        for (int iter = 0; iter < maxIter; iter++)
        {
            var Ap = CSRMatVecMul(matrix, p);
            double pAp = DotProduct(p, Ap);
            if (pAp == 0.0)
                break;
            double alpha = rsold / pAp;
            for (int i = 0; i < n; i++)
            {
                x[i] += alpha * p[i];
                r[i] -= alpha * Ap[i];
            }
            double rsnew = DotProduct(r, r);
            if (Math.Sqrt(rsnew) < tol)
                break;
            double beta = rsnew / rsold;
            for (int i = 0; i < n; i++)
            {
                p[i] = r[i] + beta * p[i];
            }
            rsold = rsnew;
        }
        return x;
    }

    public static List<double> CSRMatVecMul(SparseMatrixCSR matrix, List<double> x)
    {
        var y = new List<double>(new double[matrix.NRows]);
        for (int row = 0; row < matrix.NRows; row++)
        {
            int start = matrix.RowPointers[row];
            int end = matrix.RowPointers[row + 1];
            double sum = 0.0;
            for (int idx = start; idx < end; idx++)
            {
                int col = matrix.ColIndices[idx];
                sum += matrix.Values[idx] * x[col];
            }
            y[row] = sum;
        }
        return y;
    }

    public static double DotProduct(List<double> a, List<double> b)
    {
        double sum = 0.0;
        for (int i = 0; i < a.Count; i++)
        {
            sum += a[i] * b[i];
        }
        return sum;
    }
}

class SparseMatrixCSR
{
    public int NRows;
    public int NCols;
    public List<COOEntry> COOEntries;
    public List<double> Values;
    public List<int> ColIndices;
    public List<int> RowPointers;

    public SparseMatrixCSR(int size)
    {
        NRows = size;
        NCols = size;
        COOEntries = new List<COOEntry>();
        Values = new List<double>();
        ColIndices = new List<int>();
        RowPointers = new List<int>();
    }

    public void AddEntry(int row, int col, double value)
    {
        COOEntries.Add(new COOEntry(row, col, value));
    }

    public void ToCSR()
    {
        // Combine duplicate entries
        var entryMap = new Dictionary<(int, int), double>();
        foreach (var entry in COOEntries)
        {
            var key = (entry.Row, entry.Col);
            if (entryMap.ContainsKey(key))
            {
                entryMap[key] += entry.Value;
            }
            else
            {
                entryMap[key] = entry.Value;
            }
        }

        // Sort entries
        var entries = new List<COOEntry>();
        foreach (var kvp in entryMap)
        {
            entries.Add(new COOEntry(kvp.Key.Item1, kvp.Key.Item2, kvp.Value));
        }
        entries.Sort((a, b) => a.Row == b.Row ? a.Col.CompareTo(b.Col) : a.Row.CompareTo(b.Row));

        // Build CSR format
        Values = new List<double>();
        ColIndices = new List<int>();
        RowPointers = new List<int>(new int[NRows + 1]);

        int currentRow = 0;
        int idx = 0;
        foreach (var entry in entries)
        {
            while (currentRow < entry.Row)
            {
                RowPointers[currentRow + 1] = idx;
                currentRow++;
            }
            Values.Add(entry.Value);
            ColIndices.Add(entry.Col);
            idx++;
        }

        // Complete row pointers
        for (int row = currentRow; row < NRows; row++)
        {
            RowPointers[row + 1] = idx;
        }
    }

    public double GetValue(int row, int col)
    {
        int start = RowPointers[row];
        int end = RowPointers[row + 1];
        for (int idx = start; idx < end; idx++)
        {
            if (ColIndices[idx] == col)
            {
                return Values[idx];
            }
        }
        return 0.0;
    }

    public SparseMatrixCSR ExtractReducedMatrix(List<int> freeDofs)
    {
        int n = freeDofs.Count;
        var reducedMatrix = new SparseMatrixCSR(n);

        var dofMap = new Dictionary<int, int>();
        for (int idx = 0; idx < freeDofs.Count; idx++)
        {
            int dof = freeDofs[idx];
            dofMap[dof] = idx;
        }

        foreach (var rowGlobal in freeDofs)
        {
            int rowLocal = dofMap[rowGlobal];
            int start = RowPointers[rowGlobal];
            int end = RowPointers[rowGlobal + 1];
            for (int idx = start; idx < end; idx++)
            {
                int colGlobal = ColIndices[idx];
                if (dofMap.ContainsKey(colGlobal))
                {
                    int colLocal = dofMap[colGlobal];
                    reducedMatrix.AddEntry(rowLocal, colLocal, Values[idx]);
                }
            }
        }

        reducedMatrix.ToCSR();
        return reducedMatrix;
    }
}

class COOEntry
{
    public int Row;
    public int Col;
    public double Value;

    public COOEntry(int row, int col, double value)
    {
        Row = row;
        Col = col;
        Value = value;
    }
}

class Program
{
    static void Main(string[] args)
    {
        int nNodes = 5;
        var femSystem = new FEMSystem(nNodes);

        // Prescribed displacements at nodes 1 and 5 (indices 0 and 4)
        femSystem.Nodes[0].IsDisplacementPrescribed = true; // Node 1 fixed
        femSystem.Nodes[4].IsDisplacementPrescribed = true; // Node 5 fixed

        // Applied force at node 3 (index 2)
        femSystem.Nodes[2].Force = 1000.0; // F3 = 1000 N

        // Initialize elements
        // Element 1: nodes 0 and 1 (nodes 1 and 2), k = 500 N/mm
        femSystem.Elements.Add(new Element(
            id: 0,
            node1Id: 0,
            node2Id: 1,
            stiffness: 500.0
        ));

        // Element 2: nodes 1 and 3 (nodes 2 and 4), k = 400 N/mm
        femSystem.Elements.Add(new Element(
            id: 1,
            node1Id: 1,
            node2Id: 3,
            stiffness: 400.0
        ));

        // Element 3: nodes 2 and 1 (nodes 3 and 2), k = 600 N/mm
        femSystem.Elements.Add(new Element(
            id: 2,
            node1Id: 2,
            node2Id: 1,
            stiffness: 600.0
        ));

        // Element 4: nodes 0 and 2 (nodes 1 and 3), k = 200 N/mm
        femSystem.Elements.Add(new Element(
            id: 3,
            node1Id: 0,
            node2Id: 2,
            stiffness: 200.0
        ));

        // Element 5: nodes 2 and 3 (nodes 3 and 4), k = 400 N/mm
        femSystem.Elements.Add(new Element(
            id: 4,
            node1Id: 2,
            node2Id: 3,
            stiffness: 400.0
        ));

        // Element 6: nodes 3 and 4 (nodes 4 and 5), k = 300 N/mm
        femSystem.Elements.Add(new Element(
            id: 5,
            node1Id: 3,
            node2Id: 4,
            stiffness: 300.0
        ));

        var femMetadata = femSystem.CalculateDisplacements();

        // Compute reactions at prescribed DOFs
        var reactions = new List<(int, double)>();
        foreach (var i in femMetadata.PrescribedDofs)
        {
            double Ri = 0.0;
            for (int j = 0; j < femSystem.Nodes.Count; j++)
            {
                Ri += femMetadata.StiffnessMatrix.GetValue(i, j) * femSystem.Nodes[j].Displacement;
            }
            Ri -= femSystem.Nodes[i].Force;
            reactions.Add((i, Ri));
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
                Console.WriteLine("Node {0}: Fixed (Displacement = {1:F4} mm)", node.Id + 1, node.Displacement);
            }
            else
            {
                Console.WriteLine("Node {0}: u = {1:F4} mm", node.Id + 1, node.Displacement);
            }
        }

        Console.WriteLine("\nElement Forces (Positive=Tension, Negative=Compression):");
        foreach (var e in femSystem.Elements)
        {
            Console.WriteLine("Element {0}: P = {1:F2} N", e.Id + 1, e.Force);
        }

        Console.WriteLine("\nReactions at Prescribed Nodes:");
        foreach (var (nodeId, reaction) in reactions)
        {
            Console.WriteLine("Node {0}: R = {1:F2} N", nodeId + 1, reaction);
        }
    }
}
