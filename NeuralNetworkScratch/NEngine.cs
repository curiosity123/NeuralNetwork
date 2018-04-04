using System;
using System.Collections.Generic;
using System.Diagnostics;
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
        private double[,] AllX;
        private double[,] AllY;
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
        private readonly int _batchSize;

        public NEngine(Layer[] layers, double[,] Y, double LearningRate, double ReguralizationRate, int BatchSize = 1200)
        {
            AllY = Y;
            AllX = layers[0].matrix;
            this.layers = layers;
            _learningRate = LearningRate;
            _regularizationRate = ReguralizationRate;
            _batchSize = BatchSize;
            Initialize();
            ForwardPropagation();
        }

        private void LoadNextBatchSet(int offset)
        {
            if (offset < layers[0].matrix.GetLength(0))
            {
                X = layers[0].matrix.GetBatch(_batchSize, offset);
                X = X.AddFeatureBias(1);
                Y = AllY.GetBatch(_batchSize, offset);
            }
        }


        private void Initialize()
        {
            X = layers[0].matrix.GetBatch(_batchSize, 0);
            X = X.AddFeatureBias(1);
            Y = AllY.GetBatch(_batchSize, 0);
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

                W[i - 1] = W[i - 1].Rand().AddWeightBias(0.1);
            }

        }

        public double[,] CheckAnswer(double[,] TX)
        {

            double[][,] At = new double[layers.Length - 1][,];
            double[][,] Zt = new double[layers.Length - 1][,];
            double[,] TestedX = TX.AddFeatureBias(1);

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
                    Zt[i] = TestedX.Mul(W[i]);
                    At[i] = Zt[i].Func(activationFunc[i]);
                }
                else if (i == layers.Length - 1)
                {
                    Zt[i] = At[i - 1].AddFeatureBias(1).Mul(W[i]);
                    At[i] = Zt[i].Func((x) => x);
                }
                else
                {
                    Zt[i] = At[i - 1].AddFeatureBias(1).Mul(W[i]);
                    At[i] = Zt[i].Func(activationFunc[i]);
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
                    Z[i] = X.Mul(W[i]);
                    A[i] = Z[i].Func(activationFunc[i]);
                }
                else if (i == layers.Length - 1)
                {
                    Z[i] = A[i - 1].AddFeatureBias(1).Mul(W[i]);
                    A[i] = Z[i].Func((x) => x);
                }
                else
                {
                    Z[i] = A[i - 1].AddFeatureBias(1).Mul(W[i]);
                    A[i] = Z[i].Func(activationFunc[i]);
                }
            }

            return A[layers.Length - 2];
        }

        public void BackwardPropagation(int Iteration = 1000)
        {
            for (int i = 0; i < Iteration; i++)
            {
                int batchCount = AllX.GetLength(0) / _batchSize;
                if (((double)AllX.GetLength(0) /(double) _batchSize) > (double)batchCount)
                    batchCount++;

                int offset = 0;
                for (int b = 0; b < batchCount; b++)
                {
                    ForwardPropagation();
                    GradientFunction();
                    UpdateWeight();


                    offset += _batchSize;
                   
                    LoadNextBatchSet(offset);
                }
            }
        }

        private void GradientFunction()
        {
            for (int i = layers.Length - 1; i > 0; i--)
            {
                if (i == 1) // input layer 
                {
                    Sigma[0] = Sigma[1].Mul(W[1].Transpose()).Func(Z[0].AddFeatureBias(1).Func(activationFuncPrime[0]), (x, y) => x * y);
                    Sigma[0] = Sigma[0].RemoveFeatureBias();
                    Gradient[0] = X.Transpose().Mul(Sigma[0]);
                    //    Gradient[0] = Matrix.Func(Gradient[0], Matrix.Func(W[0], (x) => x * _regularizationRate), (x, y) => x - y);

                }
                else if (i == layers.Length - 1) // output layer
                {
                    var cost = Y.Func(A[A.Length - 1], (x, y) => x - y);
                    Sigma[Sigma.Length - 1] = cost.Func(Z[Z.Length - 1], (x, y) => x * -activationFuncPrime[i - 1](y));
                    Gradient[Gradient.Length - 1] = A[A.Length - 2].AddFeatureBias(1).Transpose().Mul(Sigma[Sigma.Length - 1]);
                    //    Gradient[Gradient.Length - 1] = Matrix.Func(Gradient[Gradient.Length - 1], Matrix.Func(W[i - 1], (x) => x * _regularizationRate), (x, y) => x - y);
                }
                else  // hidden layer
                {
                    Sigma[i - 1] = Sigma[i].Mul(W[i].Transpose()).Func(Z[i - 1].AddFeatureBias(1).Func(activationFuncPrime[i - 1]), (x, y) => x * y);
                    Gradient[i - 1] = A[i - 2].AddFeatureBias(1).Transpose().Mul(Sigma[i - 1]).RemoveFeatureBias();
                    Sigma[i - 1] = Sigma[i - 1].RemoveFeatureBias();
                    //   Gradient[Gradient.Length - 1] = Matrix.Func(Gradient[Gradient.Length - 1], Matrix.Func(W[i - 1], (x) => x * _regularizationRate), (x, y) => x - y);
                }

            }




        }

        private void UpdateWeight()
        {
            Parallel.For(0, W.Length, i =>
            {
                W[i] = W[i].Func(Gradient[i].Func((x) => (_learningRate / (double)X.GetLength(0)) * x), (x, y) => x - y);
            });
        }

        public string GetMAELoss()
        {
            double[,] loss = CheckAnswer(X.RemoveFeatureBias()).Func(Y, (x, y) => Math.Abs(x - y));

            double result = loss.Sum();///Matrix.Sum(Y);
            return Math.Round(result, 4).ToString();
        }

        public string GetRMSELoss(double[,] data, double[,] newY)
        {
            double[,] loss = CheckAnswer(data).Func(newY, (x, y) => Math.Pow(x - y, 2));

            double result = Math.Sqrt(loss.Sum() / loss.GetLength(0));///Matrix.Sum(Y);
            return Math.Round(result, 4).ToString();
        }


    }
}
