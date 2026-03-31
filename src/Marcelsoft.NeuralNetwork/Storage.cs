using MathNet.Numerics.Data.Text;
using System.IO.Compression;

namespace Marcelsoft.NeuralNetwork
{
    public static class Storage
    {
        public static void Save(NeuralNetwork network, string path, string name)
        {
            try
            {
                var workingDir = Path.Combine(path, name);
                Directory.CreateDirectory(workingDir);

                // Save metadata
                using (var sw = new StreamWriter(Path.Combine(workingDir, "network.def")))
                {
                    sw.WriteLine($"Type: NN");
                    sw.WriteLine($"Layers: {network.Layers.Count}");
                }

                // Save layer data
                for (int i = 0; i < network.Layers.Count; i++)
                {
                    MatrixMarketWriter.WriteMatrix(Path.Combine(workingDir, $"layer{i}.weights.mtx"),
                        network.Layers[i].Weights);
                    MatrixMarketWriter.WriteMatrix(Path.Combine(workingDir, $"layer{i}.biases.mtx"),
                        network.Layers[i].Biases);
                }

                // Create zip archive
                var zipPath = Path.Combine(path, $"{name}.zip");
                if (File.Exists(zipPath))
                    File.Delete(zipPath);

                ZipFile.CreateFromDirectory(workingDir, zipPath);

                // Clean up temporary directory
                Directory.Delete(workingDir, true);
            }
            catch (Exception ex)
            {
                throw new Exception($"Failed to save neural network to {name}.zip: {ex.Message}");
            }
        }

        public static NeuralNetwork Load(string path, string name)
        {
            try
            {
                var zipPath = Path.Combine(path, $"{name}.zip");
                var workingDir = Path.Combine(path, name);

                // Extract zip archive
                if (Directory.Exists(workingDir))
                    Directory.Delete(workingDir, true);

                ZipFile.ExtractToDirectory(zipPath, workingDir);

                var network = new NeuralNetwork();

                using (var sr = new StreamReader(Path.Combine(workingDir, "network.def")))
                {
                    var typeRow = sr.ReadLine();
                    var layersRow = sr.ReadLine();

                    var type = typeRow.Split(' ')[1];
                    var layersCountString = layersRow.Split(' ')[1];
                    var layerscount = int.Parse(layersCountString);

                    for (int i = 0; i < layerscount; i++)
                    {
                        var weights =
                            MatrixMarketReader.ReadMatrix<double>(Path.Combine(workingDir, $"layer{i}.weights.mtx"));
                        var biases =
                            MatrixMarketReader.ReadMatrix<double>(Path.Combine(workingDir, $"layer{i}.biases.mtx"));

                        network.Layers.Add(new NetworkLayer(weights, biases));
                    }
                }

                // Clean up temporary directory
                Directory.Delete(workingDir, true);

                return network;
            }
            catch (Exception ex)
            {
                throw new Exception($"Failed to load neural network from {name}.zip: {ex.Message}");
            }
        }
    }
}
