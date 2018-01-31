using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkScratch
{
    static class ActivationFunctionFactory
    {
        public static void SetFunctions(ref Func<double, double> activation, ref Func<double, double> prime, ActivationFunction type)
        {
            switch (type)
            {
                case ActivationFunction.Tanh:
                    activation = Math.Tanh;
                    prime = (x) => (1 - Math.Pow(Math.Tanh(x), 2));
                    break;
                case ActivationFunction.RELU:
                    activation = (x) => (x < 0) ? 0 : x;
                    prime = (x) => (x < 0) ? 0 : x;
                    break;
                default:
                    activation = (x) => 0;
                    prime = (x) => 0;
                    break;
            }


        }
    }
}
