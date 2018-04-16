using NUnit.Framework;
using System;
using System.Collections.Generic;
using System.Text;
using NeuralNetworkScratch;

namespace Tests.UnitTests
{

        [TestFixture]
        public class MatrixTests
        {

            [Test]
            public void MatrixSumTest()
            {
                double[,] m = new double[2, 2] { { 1, 2 }, { 4, 5 } };
                double result = m.Sum();
                Assert.AreNotEqual(12, result);
            }



        }
    
}
