using MathNet.Numerics.LinearAlgebra;

namespace Marcelsoft.NeuralNetwork.Training
{
    public class Trainer : TrainerBase
    {
        private readonly List<Vector<double>> _inputs;
        private readonly List<Vector<double>> _expectedOutputs;

        private NeuralNetwork _network;

        public Trainer(NeuralNetwork network, List<Vector<double>> inputs, List<Vector<double>> expectedOutputs)
        {
            _network = network;
            _inputs = inputs;
            _expectedOutputs = expectedOutputs;
        }

        public void Train(int numEpochs, double learningRate)
        {
            var learningCurve = learningRate / numEpochs;
            var progresStep = numEpochs / 10;

            var bestNetwork = _network.Clone();
            var bestNetworkLoss = CalculateLoss(bestNetwork, _inputs, _expectedOutputs);

            Console.WriteLine($"Initial loss: {bestNetworkLoss}");

            // Train NN using backpropagation
            for (var epoch = 0; epoch < numEpochs; epoch++)
            {
                // Loop over input sequences
                for (var i = 0; i < _inputs.Count; i++)
                {
                    // Forward pass
                    var input = _inputs[i].ToRowMatrix();      
                    //output is not used directly, but it is needed to calculate the output error and the hidden layer errors         
                    var output = _network.Forward(input);
                    var target = _expectedOutputs[i].ToRowMatrix();

                    var outputLayer = _network.Layers.Last();
                    var outputError = (outputLayer.Output - target).PointwiseMultiply(outputLayer.OutputDerivative);
                    var outputLayerWeightsCorrection = learningRate * (outputLayer.Input.Transpose() * outputError);
                    var outputLayerBiasCorrection = learningRate * outputError;

                    outputLayer.UpdateWeights(outputLayerWeightsCorrection);
                    outputLayer.UpdateBiases(outputLayerBiasCorrection);

                    var hiddenError = outputError;
                    var previousLayer = outputLayer;
                    for (var h = _network.Layers.Count - 2; h >= 0; h--)
                    {
                        var hiddenLayer = _network.Layers[h];
                        hiddenError = (hiddenError * previousLayer.Weights.Transpose()).PointwiseMultiply(hiddenLayer.OutputDerivative);
                        var hiddenLayerWeightsCorrection = learningRate * (hiddenLayer.Input.Transpose() * hiddenError);
                        var hiddenLayerBiasCorrection = learningRate * hiddenError;

                        hiddenLayer.UpdateWeights(hiddenLayerWeightsCorrection);
                        hiddenLayer.UpdateBiases(hiddenLayerBiasCorrection);

                        previousLayer = hiddenLayer;
                    }
                }

                var totalLoss = CalculateLoss(_network, _inputs, _expectedOutputs);
                if (totalLoss < bestNetworkLoss)
                {
                    bestNetworkLoss = totalLoss;
                    bestNetwork = (NeuralNetwork)_network.Clone();
                }

                // Print loss for this epoch
                if (epoch % progresStep == 0)
                {
                    Console.WriteLine($"Epoch {epoch}, loss: {totalLoss}");
                }

                learningRate -= learningCurve;
            }

            Console.WriteLine($"Best network found with loss: {bestNetworkLoss}");
            _network = bestNetwork;
        }
    }
}
