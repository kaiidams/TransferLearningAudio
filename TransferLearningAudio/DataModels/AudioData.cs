using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TransferLearningAudio.DataModels
{
    public class ESC50Data
    {
        [LoadColumn(0)]
        public string filename;
        [LoadColumn(1)]
        public int fold;
        [LoadColumn(2)]
        public int target;
        [LoadColumn(3)]
        public string category;
        [LoadColumn(4)]
        public bool esc10;
        [LoadColumn(5)]
        public string src_file;
        [LoadColumn(6)]
        public string take;
    }

    public class AudioData
    {
        public string AudioPath { get; set; } = string.Empty;
        public double Offset { get; set; }
        public string Label { get; set; } = string.Empty;
    }

    public class AudioFilePathData
    {
        public string AudioPath { get; set; } = string.Empty;
        public double Offset { get; set; }
    }

    public class WaveformData
    {
        [VectorType(Config.PatchWindowLength)]
        public float[] Waveform { get; set; } = Array.Empty<float>();
    }

    internal class AudioPredictionEx
    {
        public string AudioPath { get; set; } = string.Empty;
        public string Label { get; set; } = string.Empty;
        public float[] Score { get; set; } = Array.Empty<float>();
        public string PredictedLabelValue { get; set; } = string.Empty;
    }
}
