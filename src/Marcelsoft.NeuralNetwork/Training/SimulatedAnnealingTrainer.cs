using MathNet.Numerics.LinearAlgebra;

namespace Marcelsoft.NeuralNetwork.Training
{
    public class SimulatedAnnealingTrainer : TrainerBase
    {
        private readonly List<Vector<double>> _inputs;
        private readonly List<Vector<double>> _expectedOutputs;
        private readonly Random _random;

        private NeuralNetwork _network;

        public SimulatedAnnealingTrainer(NeuralNetwork network, List<Vector<double>> inputs, List<Vector<double>> expectedOutputs)
        {
            _network = network;
            _inputs = inputs;
            _expectedOutputs = expectedOutputs;
            _random = new Random();
        }

        public void Train(double temp, double distance, double learningRate)
        {
            var step = 0;

            var bestNetwork = (NeuralNetwork)_network.Clone();
            var bestNetworkLoss = CalculateLoss(_network);

            Console.WriteLine($"Initial loss: {bestNetworkLoss}");

            // Train NN using Simulated Annealing
            while (temp > 0.0)
            {
                step++;
                var currentLoss = CalculateLoss(_network);
                var neighbour = GetNeighbour(_network, distance);
                var neighbourLoss = CalculateLoss(neighbour);

                if (neighbourLoss < currentLoss)
                {
                    _network = neighbour.Clone();
                }

                if (neighbourLoss > currentLoss)
                {
                    if (_random.NextDouble() < temp)
                    {
                        _network = neighbour.Clone();
                    }
                }

                if (neighbourLoss < bestNetworkLoss)
                {
                    bestNetwork = neighbour.Clone();
                    bestNetworkLoss = neighbourLoss;
                }

                if (step % 1000 == 0)
                {
                    Console.WriteLine($"Step {step}, temp: {temp}, currentLoss: {currentLoss}, bestNetworkLoss: {bestNetworkLoss}");
                }

                temp -= learningRate;
                distance -= learningRate;
            }

            Console.WriteLine($"Best network found with loss: {bestNetworkLoss}");
            _network = bestNetwork;
        }

        private double CalculateLoss(NeuralNetwork network)
        {
            var totalLoss = 0.0;
            for (var i = 0; i < _inputs.Count; i++)
            {
                var x = _inputs[i].ToRowMatrix();
                var y = network.FastForward(x);
                var target = _expectedOutputs[i];
                var vy = Vector<double>.Build.DenseOfEnumerable(y.Row(0));
                var loss = target - vy;
                totalLoss += Math.Abs(loss.Enumerate().Select(Math.Abs).Sum());
            }

            return totalLoss / _inputs.Count;
        }

        private NeuralNetwork GetNeighbour(NeuralNetwork current, double distance)
        {
            var neighbour = current.Clone();
            foreach (var layer in neighbour.Layers)
            {
                layer.Weights = layer.Weights.Map(v =>
                {
                    var x = distance * (_random.NextDouble() - 0.5);
                    var tmp = v + x;
                    if (tmp > 1.0) tmp = 1.0;
                    if (tmp < -1.0) tmp = -1.0;
                    return tmp;
                });

                layer.Biases = layer.Biases.Map(v =>
                {
                    var x = distance * (_random.NextDouble() - 0.5);
                    var tmp = v + x;
                    if (tmp > 1.0) tmp = 1.0;
                    if (tmp < -1.0) tmp = -1.0;
                    return tmp;
                });
            }

            return neighbour;
        }
    }
}
