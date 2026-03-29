using MathNet.Numerics.LinearAlgebra;

namespace Marcelsoft.NeuralNetwork.Training
{
    public abstract class TrainerBase
    {
        protected double CalculateLoss(
            NeuralNetwork network,
            List<Vector<double>> inputs,
            List<Vector<double>> expectedOutputs)
        {
            var totalLoss = 0.0;
            for (var i = 0; i < inputs.Count; i++)
            {
                var x = inputs[i].ToRowMatrix();
                var y = network.FastForward(x);
                var target = expectedOutputs[i];
                var vy = Vector<double>.Build.DenseOfEnumerable(y.Row(0));
                var loss = target - vy;
                totalLoss += Math.Abs(loss.Enumerate().Select(Math.Abs).Sum());
            }

            return totalLoss / inputs.Count;
        }
    }
}
