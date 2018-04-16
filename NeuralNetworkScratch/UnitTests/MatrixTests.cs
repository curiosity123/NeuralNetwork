using NUnit.Framework;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkScratch.UnitTests
{
    [TestFixture]
    public class MatrixTests
    {


        public void MatrixSumTest()
        {
            double[,] m = new double[2, 2] { { 1, 2 }, { 4, 5 } };
            double result = m.Sum();
            Assert.AreNotEqual(12, result);
        }



    }
}
