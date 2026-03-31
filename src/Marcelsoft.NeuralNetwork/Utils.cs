using MathNet.Numerics.LinearAlgebra;

namespace Marcelsoft.NeuralNetwork
{
    public class Utils
    {
        public static Matrix<double> Softmax(Matrix<double> x)
        {
            var expX = x.Map(Math.Exp);
            var sumExpX = expX.RowSums().Sum();
            return expX / sumExpX;
        }

        public static Matrix<double> Sigmoid(Matrix<double> x)
        {
            return x.Map(Sigmoid);
        }

        public static Matrix<double> SigmoidDeriv(Matrix<double> x)
        {
            return x.Map(SigmoidDeriv);
        }

        public static Vector<double> Softmax(Vector<double> x)
        {
            var expX = x.Map(Math.Exp);
            var sumExpX = expX.Sum();
            return expX / sumExpX;
        }

        public static Vector<double> Sigmoid(Vector<double> x)
        {
            return x.Map(Sigmoid);
        }

        public static Vector<double> SigmoidDeriv(Vector<double> x)
        {
            return x.Map(SigmoidDeriv);
        }

        public static double SigmoidDeriv(double d)
        {
            var sig = Sigmoid(d);
            return sig * (1 - sig);
        }

        public static double Sigmoid(double d)
        {
            return 1 / (1 + Math.Exp(-d));
        }

        public static double ReLu(double d)
        {
            return Math.Max(0, d);
        }

        public static double ReLuDerivative(double d)
        {
            if (d < 0)
            {
                return 0;
            }
            else
            {
                return 1;
            }
        }

        public static Matrix<double> ReLu(Matrix<double> x)
        {
            return x.Map(ReLu);
        }

        public static Matrix<double> ReLuDerivative(Matrix<double> x)
        {
            return x.Map(ReLuDerivative);
        }
    }
}
