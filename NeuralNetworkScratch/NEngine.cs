using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace NeuralNetworkScratch
{
    public class NEngine
    {

        private double[,] X;
        private double[,] Y;
        private double[][,] W;
        private double[][,] Gradient;
        private double[][,] Sigma;
        private double[][,] A;
        private double[][,] Z;
        private Func<double, double>[] activationFunc;
        private Func<double, double>[] activationFuncPrime;

        private Layer[] layers;
        private readonly double _learningRate = 0.1;
        private readonly double _regularizationRate = 0.1;

        public NEngine(Layer[] layers, double[,] Y, double LearningRate, double ReguralizationRate)
        {
            this.Y = Y;
            this.layers = layers;
            _learningRate = LearningRate;
            _regularizationRate = ReguralizationRate;

            Initialize();
            ForwardPropagation();
        }


        private void Initialize()
        {
            X = layers[0].matrix;
            X = Matrix.Unsort(ref X,ref Y, new Random());
            X = Matrix.Unsort(ref X, ref Y, new Random());
            X = Matrix.AddFeatureBias(X, 1);
            W = new double[layers.Length - 1][,];
            Gradient = new double[layers.Length - 1][,];
            Sigma = new double[layers.Length - 1][,];
            A = new double[layers.Length - 1][,];
            Z = new double[layers.Length - 1][,];

            activationFunc = new Func<double, double>[layers.Length - 1];
            activationFuncPrime = new Func<double, double>[layers.Length - 1];

            for (int i = 1; i < layers.Length; i++)
            {
                if (i == 1)
                {
                    A[i - 1] = new double[X.GetLength(0), layers[i].neurons];
                    W[i - 1] = new double[X.GetLength(1) - 1, layers[i].neurons];
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

                ActivationFunctionFactory.SetFunctions(ref activationFunc[i - 1], ref activationFuncPrime[i - 1], layers[i].Type);

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

        public double[,] CheckAnswer(double[,] TX)
        {
            double[][,] At = new double[layers.Length - 1][,];
            double[][,] Zt = new double[layers.Length - 1][,];
            double[,] TestedX = Matrix.AddFeatureBias(TX, 1);

            for (int i = 1; i < layers.Length; i++)
            {
                if (i == 1)
                {
                    At[i - 1] = new double[TestedX.GetLength(0), layers[i].neurons];
                }
                else if (i == layers.Length - 1)
                {
                    At[i - 1] = new double[layers[i - 1].neurons, layers[i].matrix.GetLength(1)];
                }
                else
                {
                    At[i - 1] = new double[layers[i - 1].neurons, layers[i].neurons];
                }
            }


            for (int i = 0; i < layers.Length - 1; i++)
            {
                if (i == 0)
                {
                    Zt[i] = Matrix.Mul(TestedX, W[i]);
                    At[i] = Matrix.Func(Zt[i], activationFunc[i]);
                }
                else
                {
                    Zt[i] = Matrix.Mul(Matrix.AddFeatureBias(At[i - 1], 1), W[i]);
                    At[i] = Matrix.Func(Zt[i], activationFunc[i]);
                }
            }

            return At[layers.Length - 2];
        }

        public double[,] ForwardPropagation()
        {

            for (int i = 0; i < layers.Length - 1; i++)
            {
                if (i == 0)
                {
                    Z[i] = Matrix.Mul(X, W[i]);
                    A[i] = Matrix.Func(Z[i], activationFunc[i]);
                }
                else
                {
                    Z[i] = Matrix.Mul(Matrix.AddFeatureBias(A[i - 1], 1), W[i]);
                    A[i] = Matrix.Func(Z[i], activationFunc[i]);
                }
            }

            return A[layers.Length - 2];
        }

        public void BackwardPropagation(int Iteration = 1000)
        {
            for (int i = 0; i < Iteration; i++)
            {
                GradientFunction();
                UpdateWeight();
                ForwardPropagation();
            }
        }

        public void GradientFunction()
        {
            for (int i = layers.Length - 1; i > 0; i--)
            {
                if (i == 1) // input layer 
                {
                    Sigma[0] = Matrix.Func(Matrix.Mul(Sigma[1], Matrix.Transpose(W[1])), Matrix.Func(Matrix.AddFeatureBias(Z[0], 1), activationFuncPrime[0]), (x, y) => x * y);
                    Sigma[0] = Matrix.RemoveFeatureBias(Sigma[0]);
                    Gradient[0] = Matrix.Mul(Matrix.Transpose(X), Sigma[0]);
                    //    Gradient[0] = Matrix.Func(Gradient[0], Matrix.Func(W[0], (x) => x * _regularizationRate), (x, y) => x - y);

                }
                else if (i == layers.Length - 1) // output layer
                {
                    var cost = Matrix.Func(Y, A[A.Length - 1], (x, y) => x - y);
                    Sigma[Sigma.Length - 1] = Matrix.Func(cost, Z[Z.Length - 1], (x, y) => x * -activationFuncPrime[i - 1](y));
                    Gradient[Gradient.Length - 1] = Matrix.Mul(Matrix.Transpose(Matrix.AddFeatureBias(A[A.Length - 2], 1)), Sigma[Sigma.Length - 1]);
                    //    Gradient[Gradient.Length - 1] = Matrix.Func(Gradient[Gradient.Length - 1], Matrix.Func(W[i - 1], (x) => x * _regularizationRate), (x, y) => x - y);
                }
                else  // hidden layer
                {
                    Sigma[i - 1] = Matrix.Func(Matrix.Mul(Sigma[i], Matrix.Transpose(W[i])), Matrix.Func(Matrix.AddFeatureBias(Z[i - 1], 1), activationFuncPrime[i - 1]), (x, y) => x * y);
                    Gradient[i - 1] = Matrix.RemoveFeatureBias(Matrix.Mul(Matrix.Transpose(Matrix.AddFeatureBias(A[i - 2], 1)), Sigma[i - 1]));
                    Sigma[i - 1] = Matrix.RemoveFeatureBias(Sigma[i - 1]);
                    //   Gradient[Gradient.Length - 1] = Matrix.Func(Gradient[Gradient.Length - 1], Matrix.Func(W[i - 1], (x) => x * _regularizationRate), (x, y) => x - y);
                }

            }




        }

        public void UpdateWeight()
        {
            for (int i = 0; i < W.Length; i++)
            {
                W[i] = Matrix.Func(W[i], Matrix.Func(Gradient[i], (x) => (_learningRate / (double)X.GetLength(0)) * x), (x, y) => x - y);
            }
        }

        public string GetCurrentLoss(double[,] DataSet, double[,] ExpectedResult)
        {
            double[,] loss = Matrix.Func(CheckAnswer(DataSet), ExpectedResult, (x,y)=> Math.Abs(x-y));


        }


    }
}
