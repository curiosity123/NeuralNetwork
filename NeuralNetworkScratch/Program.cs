using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace NeuralNetworkScratch
{
    class Program
    {
        static void Main(string[] args)
        {
            //row,column




            //[examples - rows, features -columns
            double[,] X = new double[,] {
            {1.4,  -1, 0.4 },
            {0.4,  -1, 0.1 },
            {5.4,  -1, 4 },
            {1.5,  -1, 1 },
            {1.8,   1, 1 } };

            double[,] Y = new double[,]{
            {0.45},
            {0.8},
            {0.2},
            {0.5},
            { 0.55} };


            //wiersz to jeden pomiar, kolumna to jeden ficzer
            Layer[] layers = new Layer[]
            {
                new Layer(LayerType.Input,  X, ActivationFunction.Tanh),
                new Layer(LayerType.Hidden, 3, ActivationFunction.Tanh),
                new Layer(LayerType.Hidden, 2, ActivationFunction.Tanh),
                new Layer(LayerType.Output, Y, ActivationFunction.Tanh)
            };

            NEngine nn = new NEngine(layers, Y);
            Matrix.Print(nn.ForwardPropagation());
            Console.WriteLine("\n\n");
            nn.BackwardPropagation();
            Matrix.Print(nn.ForwardPropagation());
            Console.ReadKey();
        }




    }

    internal enum LayerType
    {
        Input,
        Hidden,
        Output
    }

    internal enum ActivationFunction
    {
        Tanh,
        Sigmoid,
        RELU
    }

    internal class Layer
    {
        public LayerType type;
        public double[,] matrix;
        public int neurons;
        public ActivationFunction func;

        public Layer(LayerType input, double[,] x, ActivationFunction tanh)
        {
            this.type = input;
            this.matrix = x;
            this.func = tanh;
        }
        public Layer(LayerType input, int neuronsCount, ActivationFunction tanh)
        {
            this.type = input;
            this.neurons = neuronsCount;
            this.func = tanh;
        }
    }

    internal class NEngine
    {

        private double[,] X;
        private double[,] Y;
        private double[][,] W;
        private double[][,] Gradient;
        private double[][,] Sigma;
        private double[][,] A;
        private double[][,] Z;

        private Layer[] layers;

        public NEngine(Layer[] layers, double[,] Y)
        {
            this.Y = Y;
            this.layers = layers;

            Initialize();
            ForwardPropagation();
        }



        private void Initialize()
        {
            X = layers[0].matrix;
            X = Matrix.AddFeatureBias(X, 1);
            W = new double[layers.Length - 1][,];
            Gradient = new double[layers.Length - 1][,];
            Sigma = new double[layers.Length - 1][,];
            A = new double[layers.Length - 1][,];
            Z = new double[layers.Length - 1][,];

            for (int i = 1; i < layers.Length; i++)
            {
                if (i == 1)
                {
                    A[i - 1] = new double[X.GetLength(0), layers[i].neurons];
                    W[i - 1] = new double[X.GetLength(1)-1, layers[i].neurons];
                }
                else if (i == layers.Length - 1)
                {
                    A[i - 1] = new double[layers[i - 1].neurons, layers[i].matrix.GetLength(1)];
                    W[i - 1] = new double[layers[i - 1].neurons, layers[i].matrix.GetLength(1)];
                }
                else
                {
                    A[i - 1] = new double[layers[i - 1].neurons, layers[i].neurons];
                    W[i - 1] = new double[layers[i - 1].neurons, layers[i].neurons];
                }

                W[i - 1] = Matrix.Rand(W[i - 1], new Random());

                //if (i == 1)
                //    W[i - 1] = new double[3, 3] {
                //    {0.01,  0.05,  0.07},
                //    {0.20,  0.041, 0.11},
                //    {0.04,  0.56, 0.13}};


                //if (i == 2)
                //    W[i - 1] = new double[3, 2] {
                //    {0.04,  0.78},
                //    {0.40,  0.45},
                //    {0.65,  0.23}};


                //if (i == 3)
                //    W[i - 1] = new double[2, 1] {
                //    {0.04},
                //    {0.41}};

                W[i - 1] = Matrix.AddWeightBias(W[i - 1], 0.1);
            }
        }
        public double[,] ForwardPropagation()
        {

            for (int i = 0; i < layers.Length - 1; i++)
            {
                if (i == 0)
                {
                    Z[i] = Matrix.Mul(X, W[i]);
                    A[i] = Matrix.Func(Z[i], (x) => Math.Tanh(x));
                }
                else
                {
                    Z[i] = Matrix.Mul(Matrix.AddFeatureBias(A[i - 1], 1), W[i]);
                    A[i] = Matrix.Func(Z[i], (x) => Math.Tanh(x));
                }
            }

            return A[layers.Length - 2];
        }
        public void BackwardPropagation()
        {
            for (int i = 0; i < 100000; i++)
            {

                GradientFunction();
                UpdateWeight();
                ForwardPropagation();

                if (i % 1000 == 0)
                {
                   // Console.Clear();
                   // Matrix.Print(A[A.Length-1]);
                   // Console.Write("\n");
                   // Matrix.Print(Y);
                   // Thread.Sleep(5);
                }
               
                
            }
        }




        public void GradientFunction()
        {
            for (int i = layers.Length - 1; i > 0; i--)
            {
                if (i == 1) // input layer 
                {
                    Sigma[0] = Matrix.Func(Matrix.Mul(Sigma[1], Matrix.Transpose(W[1])), Matrix.Func(Matrix.AddFeatureBias(Z[0], 1), (x) => Matrix.TanhPrime(x)), (x, y) => x * y);
                    Gradient[0] = Matrix.RemoveFeatureBias(Matrix.Mul(Matrix.Transpose(Matrix.AddFeatureBias(X, 1)), Sigma[0]));
                    
                }
                else if (i == layers.Length - 1) // output layer
                {
                    var cost = Matrix.Func(Y, A[A.Length-1], (x, y) => x - y);
                    Sigma[Sigma.Length-1] = Matrix.Func(cost, Z[Z.Length-1], (x, y) => -x * Matrix.TanhPrime(y));
                    Gradient[Gradient.Length-1] = Matrix.Mul(Matrix.Transpose(Matrix.AddFeatureBias(A[A.Length-2], 1)), Sigma[Sigma.Length-1]);
                }
                else  // hidden layer
                {
                    Sigma[i-1] = Matrix.Func(Matrix.Mul(Sigma[i], Matrix.Transpose(W[i])), Matrix.Func(Matrix.AddFeatureBias(Z[i-1], 1), (x) => Matrix.TanhPrime(x)), (x, y) => x * y);
                    Gradient[i-1] = Matrix.RemoveFeatureBias(Matrix.Mul(Matrix.Transpose(Matrix.AddFeatureBias(A[i-2], 1)), Sigma[i-1]));
                    Sigma[i - 1] = Matrix.RemoveFeatureBias(Sigma[i - 1]);
                }
            }




        }

        public void UpdateWeight()
        {
            for(int i=0;i<W.Length;i++)
                W[i] = Matrix.Func(W[i], Matrix.Func( Gradient[i], (x) => (0.1*((double)1/X.GetLength(0))* x)),(x,y)=> x-y);  
        }


    }

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
                    matrix[i, j] = (2 * r.NextDouble()) - 1;

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

    public static class Helper
    {
        public static double Normalize(double value, double minValue, double maxValue)
        {
            return (value - minValue) / (maxValue - minValue);
        }
    }
}
