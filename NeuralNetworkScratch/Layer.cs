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
        public ActivationFunction Type;

        public Layer(LayerType input, double[,] x, ActivationFunction tanh)
        {
            this.type = input;
            this.matrix = x;
            this.Type = tanh;
        }
        public Layer(LayerType input, int neuronsCount, ActivationFunction tanh)
        {
            this.type = input;
            this.neurons = neuronsCount;
            this.Type = tanh;
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
