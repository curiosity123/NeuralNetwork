using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkScratch
{
    class Program
    {
        static void Main(string[] args)
        {
            //row,column





            double[,] X = new double[5, 3] {
            {1.4,  -1, 0.4 },
            {0.4,  -1, 0.1 },
            {5.4,  -1, 4 },
            {1.5,  -1, 1 },
            {1.8,   1, 1 } };

            double[,] Y = new double[4, 1]{
            {0.45},
            {0.52},
            {0.25},
            {0.31}};




            //wiersz to jeden pomiar, kolumna to jeden ficzer
            Layer[] layers = new Layer[]
            {
                new Layer(LayerType.Input,  X, ActivationFunction.Tanh),
                new Layer(LayerType.Hidden, 3, ActivationFunction.Tanh),
                new Layer(LayerType.Hidden, 2, ActivationFunction.Tanh),
                new Layer(LayerType.Output, Y, ActivationFunction.Tanh)
            };

            NEngine nn = new NEngine(layers);
            nn.Initialize();
            Helper.Print(nn.ForwardPropagation());
            Helper.Print(nn.ForwardPropagation());
            Helper.Print(nn.ForwardPropagation());






            double[,] W1 = new double[3, 3] {
            {0.01,  0.05,  0.07},
            {0.20,  0.041, 0.11},
            {0.04,  0.56, 0.13}};


            double[,] Z1 = Matrix.Mul(X, W1);
            double[,] A1 = Matrix.Func(Z1, (x) => Math.Tanh(x));


            double[,] W2 = new double[3, 2] {
            {0.04,  0.78},
            {0.40,  0.45},
            {0.65,  0.23}};


            double[,] Z2 = Matrix.Mul(A1, W2);
            double[,] A2 = Matrix.Func(Z2, (x) => Math.Tanh(x));


            double[,] W3 = new double[2, 1] {
            {0.04},
            {0.41}};

            double[,] Z3 = Matrix.Mul(A2, W3);
            double[,] A3 = Matrix.Func(Z3, (x) => Math.Tanh(x));



            Helper.Print(Matrix.Func(Matrix.Mul(Matrix.AddFeatureBias(X, 1), Matrix.AddWeightBias(W1, 0.1)), (x) => Math.Tanh(x)));

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
        private double[][,] W;
        private double[][,] A;

        private Layer[] layers;

        public NEngine(Layer[] layers)
        {
            this.layers = layers;
        }



        public void Initialize()
        {
            X = layers[0].matrix;
            X = Matrix.AddFeatureBias(X, 1);
            W = new double[layers.Length - 1][,];
            A = new double[layers.Length - 1][,];

            for (int i = 1; i < layers.Length; i++)
            {
                if (i == 1)
                {
                    A[i - 1] = new double[X.GetLength(0), layers[i].neurons];
                    W[i - 1] = new double[X.GetLength(0), layers[i].neurons];
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

                //W[i - 1] = Matrix.Rand(W[i - 1], new Random());

                if (i == 1)
                    W[i - 1] = new double[3, 3] {
                    {0.01,  0.05,  0.07},
                    {0.20,  0.041, 0.11},
                    {0.04,  0.56, 0.13}};


                if (i == 2)
                    W[i - 1] = new double[3, 2] {
                    {0.04,  0.78},
                    {0.40,  0.45},
                    {0.65,  0.23}};


                if (i == 3)
                    W[i - 1] = new double[2, 1] {
                    {0.04},
                    {0.41}};

                W[i - 1] = Matrix.AddWeightBias(W[i - 1], 0.1);
            }
        }
        public double[,] ForwardPropagation()
        {

            for (int i = 0; i < layers.Length-1; i++)
            {
                if (i == 0)
                    A[i] = Matrix.Func(Matrix.Mul(X, W[i]), (x) => Math.Tanh(x));
                else
                    A[i] = Matrix.Func(Matrix.Mul(Matrix.AddFeatureBias(A[i-1],1), W[i]), (x) => Math.Tanh(x));
            }

            return A[layers.Length - 2];
        }
        public void BackwardPropagation() { }



    }

    public static class Matrix
    {
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
        public static double[,] Func(double[,] A, Func<double, double> Function, bool Swap = false)
        {
            int rA = A.GetLength(0);
            int cA = A.GetLength(1);

            double[,] ResultMatrix = new double[cA, rA];
            double[,] ResultMatrixSwap = new double[rA, cA];

            for (int i = 0; i < rA; i++)
                for (int j = 0; j < cA; j++)
                    if (Function == null)
                    {
                        ResultMatrix[j, i] = A[i, j];
                        ResultMatrixSwap[i, j] = A[i, j];
                    }
                    else
                    {
                        ResultMatrix[j, i] = Function(A[i, j]);
                        ResultMatrixSwap[i, j] = Function(A[i, j]);
                    }
            if (!Swap)
                return ResultMatrixSwap;
            else
                return ResultMatrix;
        }
        public static double[,] Mul(double[,] A, double[,] B)
        {
            int rA = A.GetLength(0);
            int cA = A.GetLength(1);
            int rB = B.GetLength(0);
            int cB = B.GetLength(1);
            double temp = 0;
            double[,] ResultMatrix = new double[rA, cB];

            for (int i = 0; i < rA; i++)
            {
                for (int j = 0; j < cB; j++)
                {
                    temp = 0;
                    for (int k = 0; k < cA; k++)
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
    }

    public static class Helper
    {
        public static double TanhDeriv(double x)
        {
            return (x * (1 - x));
        }


        // SIGMOID a nie TANH
        //public static double Tanh(double x)
        //{
        //    return 1 / (1 + Math.Exp(-x));
        //}

        public static double Normalize(double value, double minValue, double maxValue)
        {
            return (value - minValue) / (maxValue - minValue);
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
