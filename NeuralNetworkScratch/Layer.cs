using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkScratch
{
    public class Layer
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
    public enum LayerType
    {
        Input,
        Hidden,
        Output
    }

    public enum ActivationFunction
    {
        Tanh,
        Sigmoid,
        RELU
    }


}
