using NeuralNetworkScratch;
using NUnit.Framework;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Test
{
    [TestFixture]
    public class UnitTests
    {

        [Test]
        public void MatrixSumTest()
        {
            double[,] m = new double[2, 2] { { 1, 2 }, { 4, 5 } };
            double result = m.Sum();
            Assert.AreEqual(12, result);
        }



    }
}
