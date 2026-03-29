using MathNet.Numerics.LinearAlgebra;

namespace Marcelsoft.NeuralNetwork
{
    public class NeuralNetwork
    {
        private readonly List<NetworkLayer> _layers;

        public NeuralNetwork()
        {
            _layers = new List<NetworkLayer>();
        }

        public List<NetworkLayer> Layers => _layers;

        public void CreateLayer(int inputSize, int outputSize)
        {
            _layers.Add(new NetworkLayer(inputSize, outputSize));
        }

        public Matrix<double> Forward(Matrix<double> input)
        {
            var tmp = input;
            foreach (var layer in _layers)
            {
                tmp = layer.Forward(tmp);
            }
            return tmp;
        }

        public Matrix<double> FastForward(Matrix<double> input)
        {
            var tmp = input;
            foreach (var layer in _layers)
            {
                tmp = layer.FastForward(tmp);
            }
            return tmp;
        }

        public NeuralNetwork Clone()
        {
            var clone = new NeuralNetwork();
            foreach (var layer in _layers)
            {
                clone.Layers.Add(layer.Clone());
            }

            return clone;
        }
    }
}
