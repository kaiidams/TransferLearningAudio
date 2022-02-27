using Microsoft.ML;
using TransferLearningAudio.DataModels;
using static TransferLearningAudio.Model.ConsoleHelpers;

namespace TransferLearningAudio.Model
{
    public class ModelBuilder
    {
        private static string LabelAsKey = nameof(LabelAsKey);
        private static string PredictedLabelValue = nameof(PredictedLabelValue);

        private MLContext mlContext;
        private string inputONNXModelFilePath;
        private string outputMlNetModelFilePath;

        public ModelBuilder(string inputModelLocation, string outputModelLocation)
        {
            this.inputONNXModelFilePath = inputModelLocation;
            this.outputMlNetModelFilePath = outputModelLocation;
            mlContext = new MLContext(seed: 1);
        }

        public void BuildAndTrain(string audiosetFolder)
        {
            try
            {

                ConsoleWriteHeader("Read model");
                Console.WriteLine($"Model location: {inputONNXModelFilePath}");

                // 2. Load the initial full image-set into an IDataView and shuffle so it'll be better balanced
                IDataView shuffledFullAudioDataset = LoadAudioClipsFromDirectory(audiosetFolder);

                // 4. Split the data 80:20 into train and test sets, train and evaluate.
                var trainTestData = mlContext.Data.TrainTestSplit(shuffledFullAudioDataset, testFraction: 0.2);
                IDataView trainDataView = RepeatWithOffset(trainTestData.TrainSet, audiosetFolder);
                IDataView testDataView = SampleWithOffset(trainTestData.TestSet, audiosetFolder);

                Action <AudioFilePathData, WaveformData> DecodeWavAction = (s, f) =>
                {
                    f.Waveform = DecodeAudio(s.AudioPath, s.Offset);
                };

                var dataProcessPipeline = mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: LabelAsKey, inputColumnName: "Label")

                    //
                    .Append(mlContext.Transforms.CustomMapping(DecodeWavAction, "DecodeWav"))

                    // Retrieves the 'Prediction' from TensorFlow and copies to a column
                    .Append(mlContext.Transforms.CopyColumns(Config.EncoderInput, "Waveform"))

                    // Passes the data to TensorFlow for scoring
                    .Append(mlContext.Transforms.ApplyOnnxModel(
                        modelFile: inputONNXModelFilePath,
                        shapeDictionary: new Dictionary<string, int[]>()
                        {
                            [Config.EncoderInput] = new int[] { Config.PatchWindowLength },
                            [Config.EncoderOutput] = new int[] { 1, Config.EmbeddingSize }
                        }))

                    .Append(mlContext.Transforms.CopyColumns("Features", Config.EncoderOutput));

                // 3. Set the training algorithm and convert back the key to the categorical values                            
                var trainer = mlContext.MulticlassClassification.Trainers.LbfgsMaximumEntropy(labelColumnName: LabelAsKey, featureColumnName: "Features");
                var trainingPipeline = dataProcessPipeline.Append(trainer)
                    .Append(mlContext.Transforms.Conversion.MapKeyToValue(PredictedLabelValue, "PredictedLabel"));

                // 4. Train the model
                // Measuring training time
                var watch = System.Diagnostics.Stopwatch.StartNew();

                ConsoleWriteHeader("Training the ML.NET classification model");
                ITransformer model = trainingPipeline.Fit(trainDataView);

                watch.Stop();
                long elapsedMs = watch.ElapsedMilliseconds;
                Console.WriteLine("Training with transfer learning took: " + (elapsedMs / 1000).ToString() + " seconds");

                // 5. Make bulk predictions and calculate quality metrics
                ConsoleWriteHeader("Create Predictions and Evaluate the model quality");
                IDataView predictionsDataView = model.Transform(testDataView);

                // 5.1 Show the predictions
                ConsoleWriteHeader("*** Showing all the predictions ***");
                List<AudioPredictionEx> predictions = mlContext.Data.CreateEnumerable<AudioPredictionEx>(predictionsDataView, false, true).ToList();
                predictions.ForEach(pred => ConsoleWriteImagePrediction(pred.AudioPath, pred.Label, pred.PredictedLabelValue, pred.Score.Max()));
#if false
                var predictionEngine = mlContext.Model.CreatePredictionEngine<AudioData, CommandPrediction>(model);
                foreach (var data in audioSet)
                {
                    var result = predictionEngine.Predict(data);
                    Console.WriteLine("{0} {1}", data.Label, result.Prediction);
                }
#endif
            }
            catch (Exception ex)
            {
                Console.WriteLine(ex.ToString());
            }
        }

        IDataView LoadAudioClipsFromDirectory(string folder)
        {
            string metadataPath = Path.Combine(folder, "meta", "esc50.csv");
            IDataView trainingDataView = mlContext.Data.LoadFromTextFile<ESC50Data>(
                path: metadataPath, hasHeader: true, separatorChar: ',');
            return mlContext.Data.FilterByCustomPredicate<ESC50Data>(
                trainingDataView, filterPredicate: data => !(data.category == "dog" || data.category == "cat"));
        }

        IDataView RepeatWithOffset(IDataView dataView, string folder)
        {
            var result = new List<AudioData>();
            var data = mlContext.Data.CreateEnumerable<ESC50Data>(dataView, false, true);
            int repeat = (int)(((Config.SampleSeconds - Config.PatchWindowSeconds) / Config.PatchHopSeconds) + 1);
            for (int i = 0; i < repeat; i++)
            {
                foreach (var row in data)
                {
                    result.Add(new AudioData
                    {
                        AudioPath = Path.Combine(folder, "audio", row.filename),
                        Offset = Config.PatchHopSeconds * i,
                        Label = row.category
                    });
                }
            }
            return mlContext.Data.LoadFromEnumerable(result);
        }

        IDataView SampleWithOffset(IDataView dataView, string folder)
        {
            var result = new List<AudioData>();
            var data = mlContext.Data.CreateEnumerable<ESC50Data>(dataView, false, true);
            int repeat = (int)(((Config.SampleSeconds - Config.PatchWindowSeconds) / Config.PatchHopSeconds) + 1);
            int i = 0;
            foreach (var row in data)
            {
                result.Add(new AudioData
                {
                    AudioPath = Path.Combine(folder, "audio", row.filename),
                    Offset = Config.PatchHopSeconds * (i % repeat),
                    Label = row.category
                });
                i++;
            }
            return mlContext.Data.LoadFromEnumerable(result);
        }

        float[] DecodeAudio(string audioPath, double offset)
        {
            short[] waveform = WaveFile.ReadWAV(audioPath, Config.SampleRate);
            int patchLength = (int)(Config.SampleRate * Config.PatchWindowSeconds);
            int patchOffset = (int)(Config.SampleRate * offset);
            float[] normalized = Normalize(waveform);
            return normalized.AsSpan(patchOffset, patchLength).ToArray();
        }

        private static float[] Normalize(short[] waveform)
        {
            int maxValue = 1;
            for (int i = 0; i < waveform.Length; i++)
            {
                int value = Math.Abs((int)waveform[i]);
                if (value > maxValue) maxValue = value;
            }
            double scale = 1.0f / maxValue;
            float[] result = new float[waveform.Length];
            for (int i = 0; i < result.Length; i++)
            {
                result[i] = (float)(waveform[i] * scale);
            }
            return result;
        }
    }
}