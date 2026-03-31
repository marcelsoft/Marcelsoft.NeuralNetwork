using MathNet.Numerics.LinearAlgebra;

namespace Marcelsoft.NeuralNetwork
{
    public class NetworkLayer
    {
        public int InputSize { get; private set; }
        public int OutputSize { get; private set; }

        public Matrix<double> Weights { get; set; }
        public Matrix<double> Biases { get; set; }
        public Matrix<double> Input { get; private set; }
        public Matrix<double> Output { get; private set; }
        public Matrix<double> OutputDerivative { get; private set; }

        public NetworkLayer(int inputSize, int outputSize)
        {
            InputSize = inputSize;
            OutputSize = outputSize;

            Weights = Matrix<double>.Build.Random(inputSize, outputSize);
            Biases = Vector<double>.Build.Random(outputSize).ToRowMatrix();
        }

        public NetworkLayer(Matrix<double> weights, Matrix<double> biases)
        {
            Weights = weights;
            Biases = biases;

            InputSize = weights.RowCount;
            OutputSize = weights.ColumnCount;
        }

        public Matrix<double> Forward(Matrix<double> input)
        {
            Input = input;
            var o = Utils.Sigmoid(input * Weights + Biases);
            Output = o;
            // Sigmoid derivative: d/dz σ(z) = σ(z) * (1 - σ(z))
            // Since o is already σ(z), compute element-wise to avoid a temporary matrix
            OutputDerivative = o.Map(v => v * (1 - v));
           
            return o;
        }

        public Matrix<double> FastForward(Matrix<double> input)
        {
            return Utils.Sigmoid(input * Weights + Biases);
        }

        public void UpdateWeights(Matrix<double> weightsCorrection)
        {
            Weights -= weightsCorrection;
        }

        public void UpdateBiases(Matrix<double> biasesCorrection)
        {
            Biases -= biasesCorrection;
        }

        public NetworkLayer Clone()
        {
            return new NetworkLayer(InputSize, OutputSize)
            {
                Biases = Biases.Clone(),
                Weights = Weights.Clone()
            };
        }
    }
}
