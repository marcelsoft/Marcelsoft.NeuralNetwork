using MathNet.Numerics.LinearAlgebra;

namespace Marcelsoft.NeuralNetwork.Training
{
    public abstract class TrainerBase
    {
        protected double CalculateLoss(
            NeuralNetwork network,
            List<Matrix<double>> inputs,
            List<Matrix<double>> expectedOutputs)
        {
            var totalLoss = 0.0;
            for (var i = 0; i < inputs.Count; i++)
            {
                var x = inputs[i];
                var y = network.FastForward(x);
                var loss = expectedOutputs[i] - y;
                totalLoss += loss.L1Norm();
            }

            return totalLoss / inputs.Count;
        }
    }
}
