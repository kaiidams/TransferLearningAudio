using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TransferLearningAudio.DataModels
{
    static class Config
    {
        public const int SampleRate = 16000;
        public const string EncoderOutput = "global_average_pooling2d";
        public const string EncoderInput = "input_1";
        public const int EmbeddingSize = 1024;
        public const double SampleSeconds = 5.0;
        public const double PatchWindowSeconds = 0.96;
        public const double PatchHopSeconds = 0.48;
        public const int PatchWindowLength = (int)(SampleRate * PatchWindowSeconds);
    }
}
