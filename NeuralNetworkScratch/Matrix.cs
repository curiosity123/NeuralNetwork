using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkScratch
{

    public static class Matrix
    {
        public static double[,] Transpose(double[,] M)
        {
            double[,] T = new double[M.GetLength(1), M.GetLength(0)];

            for (int i = 0; i < M.GetLength(0); i++)
                for (int j = 0; j < M.GetLength(1); j++)
                    T[j, i] = M[i, j];

            return T;
        }
        public static double[,] Func(double[,] MatrixA, double[,] MatrixB, Func<double, double, double> Function)
        {
            int rA = MatrixA.GetLength(0);
            int cA = MatrixA.GetLength(1);
            int rB = MatrixB.GetLength(0);
            int cB = MatrixB.GetLength(1);
            double[,] resultMatrix = new double[rA, cB];

            for (int i = 0; i < rA; i++)
                for (int j = 0; j < cA; j++)
                    resultMatrix[i, j] = Function(MatrixA[i, j], (MatrixB[i, j]));
            return resultMatrix;
        }
        public static double[,] Func(double[,] A, Func<double, double> Function)
        {
            int rA = A.GetLength(0);
            int cA = A.GetLength(1);

            double[,] Result = new double[rA, cA];

            for (int i = 0; i < rA; i++)
                for (int j = 0; j < cA; j++)
                    Result[i, j] = Function(A[i, j]);

            return Result;
        }
        public static double[,] Mul(double[,] A, double[,] B)
        {
            int rowA = A.GetLength(0);
            int columnA = A.GetLength(1);
            int rowB = B.GetLength(0);
            int columnB = B.GetLength(1);
            double temp = 0;
            double[,] ResultMatrix = new double[rowA, columnB];

            for (int i = 0; i < rowA; i++)
            {
                for (int j = 0; j < columnB; j++)
                {
                    temp = 0;
                    for (int k = 0; k < columnA; k++)
                        temp += A[i, k] * B[k, j];

                    ResultMatrix[i, j] = temp;
                }
            }
            return ResultMatrix;
        }
        public static double[,] Rand(double[,] matrix, Random r)
        {
            for (int i = 0; i < matrix.GetLength(0); i++)
                for (int j = 0; j < matrix.GetLength(1); j++)
                    matrix[i, j] = (( r.NextDouble()) );

            return matrix;
        }
        public static double[,] AddFeatureBias(double[,] A, double value)
        {
            double[,] B = new double[A.GetLength(0), A.GetLength(1) + 1];

            for (int i = 0; i < A.GetLength(0); i++)
            {
                for (int j = 0; j < A.GetLength(1); j++)
                    B[i, j] = A[i, j];
                B[i, B.GetLength(1) - 1] = value;
            }

            return B;
        }
        public static double[,] RemoveFeatureBias(double[,] A)
        {
            double[,] B = new double[A.GetLength(0), A.GetLength(1) - 1];

            for (int i = 0; i < A.GetLength(0); i++)
            {
                for (int j = 0; j < A.GetLength(1) - 1; j++)
                    B[i, j] = A[i, j];

            }

            return B;
        }
        public static double[,] SetFeatureBias(double[,] A, double value)
        {
            for (int i = 0; i < A.GetLength(0); i++)
                A[i, A.GetLength(1) - 1] = value;
            return A;
        }
        public static double[,] AddWeightBias(double[,] A, double value)
        {
            double[,] B = new double[A.GetLength(0) + 1, A.GetLength(1)];

            for (int i = 0; i < A.GetLength(0); i++)
                for (int j = 0; j < A.GetLength(1); j++)
                    B[i, j] = A[i, j];

            for (int i = 0; i < B.GetLength(1); i++)
                B[B.GetLength(0) - 1, i] = value;

            return B;
        }
        public static double TanhPrime(double x)
        {
            return (1 - Math.Pow(Math.Tanh(x), 2));// (x * (1 - x));
        }
        public static void Print(double[,] z)
        {
            for (int i = 0; i < z.GetLength(0); i++)
            {
                for (int j = 0; j < z.GetLength(1); j++)
                    Console.Write(Math.Round(z[i, j], 4).ToString() + "\t");
                Console.Write("\n");
            }
        }
    }
}
