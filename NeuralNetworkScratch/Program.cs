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

            NEngine nn = new NEngine(layers, Y,0.1,1);
            nn.ForwardPropagation().Print();
            Console.WriteLine("\n\n");
            nn.BackwardPropagation();
            nn.ForwardPropagation().Print();
            Console.ReadKey();
        }




    }


   


    //public static class Helper
    //{
    //    public static double Normalize(double value, double minValue, double maxValue)
    //    {
    //        return (value - minValue) / (maxValue - minValue);
    //    }
    //}
}
