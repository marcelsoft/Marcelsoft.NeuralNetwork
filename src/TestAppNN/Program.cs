using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.Data.Text;
using System;
using System.Collections.Generic;
using Marcelsoft.NeuralNetwork;
using Marcelsoft.NeuralNetwork.Training;
using MathNet.Numerics;
using MathNet.Numerics.Providers.CUDA;

namespace TestAppNN
{
    internal class Program
    {
        public static void Main(string[] args)
        {            
            var tmp = CudaControl.TryUseNativeCUDA();            
            var t = Control.TryUseNativeCUDA();
            var r = Control.TryUseNativeMKL();
            var c = Control.TryUseNativeOpenBLAS();
            Console.WriteLine($"Check what providers are available:");
            Console.WriteLine($"CUDA1: {tmp}");
            Console.WriteLine($"CUDA2: {t}");
            Console.WriteLine($"MKL: {r}");
            Console.WriteLine($"OpenBLAS: {c}");

            Control.UseMultiThreading();
            Control.UseBestProviders();

            Console.WriteLine($"{Control.Describe()}");

            Console.WriteLine("Create NeuralNetowork structure.");
            var rnn = new NeuralNetwork();
            rnn.CreateLayer(4, 400);
            rnn.CreateLayer(400, 30);
            rnn.CreateLayer(30, 4);

            Console.WriteLine("Check NN before learning:");
            CheckNN(rnn);

            Console.WriteLine("Learning using backpropagation:");
            TestLearning(rnn);

            Console.WriteLine("Check NN after learning:");
            CheckNN(rnn);

            Console.WriteLine("Save NN to file:");
            Storage.Save(rnn, "tmp", "testnn");

            Console.WriteLine("Load NN from file:");
            var loadedRnn = Storage.Load("tmp", "testnn");

            Console.WriteLine("Ckeck loaded NN:");
            CheckNN(loadedRnn);
        }

        private static void TestLearning(NeuralNetwork rnn)
        {
            // just some pattern to learn, in real case it should be some time series data
            var numbers = new[] { 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1 };
            var patern = new List<double>();

            for (int i = 0; i < 40; i++)
            {
                patern.Add(numbers[i % numbers.Length]);
            }

            // learning parameters: learning rate and number of epochs
            var learningRate = 0.05;
            var numEpochs = 5000;

            var inputs = new List<Vector<double>>();
            var outputs = new List<Vector<double>>();

            // Loop over input sequences
            for (var i = 0; i < patern.Count - 8; i++)
            {
                inputs.Add(Vector<double>.Build.DenseOfArray(patern.GetRange(i, 4).ToArray()));
                outputs.Add(Vector<double>.Build.DenseOfArray(patern.GetRange(i + 4, 4).ToArray()));
            }

            // Trainer class will handle the training loop and backpropagation
            var trainer = new Trainer(rnn, inputs, outputs);
            trainer.Train(numEpochs, learningRate);          
        }

        private static void CheckNN(NeuralNetwork rnn)
        {
            // Test RNN on new input sequence
            var input = Vector<double>.Build.DenseOfArray(new[] { 0.2, 0.3, 0.4, 0.5 });
            var expected = Vector<double>.Build.DenseOfArray(new[] { 0.6, 0.7, 0.8, 0.9 });
            var output = rnn.Forward(input.ToRowMatrix());
            Console.WriteLine($"Input: {string.Join(" ", input)}");
            Console.WriteLine($"Expected output: {string.Join(" ", expected)}");
            Console.WriteLine($"Real output: {string.Join(" ", output)}");  
        }        

        private static void QuotesLearning()
        {
            var steps = GetStepsNumber(940, Binance.Net.Enums.KlineInterval.OneHour);
            var dataCache = new DataCache();
            var quotes = dataCache.GetQuotes(steps, "BTCUSDT", Binance.Net.Enums.KlineInterval.OneHour);
            var data = DataNormalizator.GetMovements(quotes);

            var rnn = new NeuralNetwork();
            rnn.CreateLayer(10, 400);
            rnn.CreateLayer(400, 4);

            var learningRate = 0.1;
            var numEpochs = 5000;
            var inputs = new List<Vector<double>>();
            var outputs = new List<Vector<double>>();

            // Loop over input sequences
            for (var i = 0; i < data.Count - 28; i++)
            {
                var normalizedInput = DataNormalizator.Normalize(data.GetRange(i, 10));
                var normalizedOutput = DataNormalizator.Normalize(data.GetRange(i + 10, 4));

                // Forward pass
                inputs.Add(Vector<double>.Build.DenseOfArray(normalizedInput.ToArray()));
                outputs.Add(Vector<double>.Build.DenseOfArray(normalizedOutput.ToArray()));
            }

            var trainer = new Trainer(rnn, inputs, outputs);
            trainer.Train(numEpochs, learningRate);

            // Test RNN on new input sequence
            var random = new Random();
            var testindex = (int)random.NextInt64(100);
            var input = inputs[testindex];
            var output = rnn.Forward(input.ToRowMatrix());
            var expected = outputs[testindex];
            Console.WriteLine($"Input: {input}");
            Console.WriteLine($"Output: {output}");
            Console.WriteLine($"Expected: {expected}");
        }

        private static int GetStepsNumber(int daysCount, Binance.Net.Enums.KlineInterval interval)
        {
            int stepsInDay = 0;
            switch (interval)
            {
                case Binance.Net.Enums.KlineInterval.OneDay: stepsInDay = 1; break;
                case Binance.Net.Enums.KlineInterval.FourHour: stepsInDay = 6; break;
                case Binance.Net.Enums.KlineInterval.OneHour: stepsInDay = 24; break;
                case Binance.Net.Enums.KlineInterval.FifteenMinutes: stepsInDay = 96; break;
                case Binance.Net.Enums.KlineInterval.FiveMinutes: stepsInDay = 288; break;
                case Binance.Net.Enums.KlineInterval.OneMinute: stepsInDay = 1440; break;
            }

            return daysCount * stepsInDay;
        }

        private static void TestSaving()
        {
        }
    }
}