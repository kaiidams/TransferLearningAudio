using TransferLearningAudio.Model;

namespace TransferLearningAudio.Train
{
    public class Program
    {
        public static void Main(string[] args)
        {
            const string assetsRelativePath = @"../../../../assets";
            var assetsPath = Path.GetFullPath(assetsRelativePath);

            string modelLocation = Path.Combine(assetsPath, "yamnet.onnx");

            var audioClassifierZip = Path.Combine(assetsPath, "outputs", "audioClassifier.zip");

            //string _dataPath = Path.Combine(assetsRelativePath, "mini_speech_commands");
            string fullAudiosetFolderPath = DownloadImageSet(assetsRelativePath);
            Console.WriteLine($"Audio folder: {fullAudiosetFolderPath}");

            var modelBuilder = new ModelBuilder(modelLocation, audioClassifierZip);
            modelBuilder.BuildAndTrain(fullAudiosetFolderPath);
        }

        public static string DownloadImageSet(string imagesDownloadFolder)
        {
            string fullAudiosetFolderPath = Path.Combine(imagesDownloadFolder, "ESC-50-master");
            if (!Directory.Exists(fullAudiosetFolderPath))
            {
                throw new InvalidDataException("Download from https://github.com/karoldvl/ESC-50/archive/master.zip");
            }
            return fullAudiosetFolderPath;
        }
    }
}