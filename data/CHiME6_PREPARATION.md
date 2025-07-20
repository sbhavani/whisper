# CHiME-6 Dataset Preparation Guide

This guide explains how to prepare the CHiME-6 dataset from CHiME-5 data following the Kaldi s5_track1 recipe stage 0 process.

## Overview

The CHiME-6 dataset is created from CHiME-5 data by applying array synchronization fixes to address two main issues:
1. **Audio frame-dropping** (affects Kinect devices only)
2. **Clock-drift** (affects all devices)

The preparation process follows the stage 0 of the [Kaldi CHiME-6 s5_track1 recipe](https://github.com/kaldi-asr/kaldi/tree/master/egs/chime6/s5_track1) and extracts binaural recordings (`*_P??.wav`) with corresponding transcripts.

## Prerequisites

### Required Software
- **Python 3.6+**
- **SoX** (version 14.4.2 recommended)
  - macOS: `brew install sox`
  - Ubuntu/Debian: `sudo apt-get install sox`
  - CentOS/RHEL: `sudo yum install sox`

### Required Data
- **CHiME-5 dataset**: Download from [CHiME-5 Challenge](https://spandh.dcs.shef.ac.uk//chime_challenge/CHiME5/download.html)
  - You need to register and agree to the license terms
  - The dataset is approximately 23GB

## Usage

### Basic Usage

```bash
# Navigate to the data directory
cd /path/to/whisper/data

# Run the preparation script
python prepare_chime6_dataset.py \
    --chime5_path /path/to/CHiME5 \
    --output_dir /path/to/output
```

### Example

```bash
# Example with specific paths
python prepare_chime6_dataset.py \
    --chime5_path ~/datasets/CHiME5 \
    --output_dir ~/datasets/CHiME6_prepared \
    --verbose
```

### Command Line Arguments

- `--chime5_path`: Path to the CHiME-5 dataset directory (required)
- `--output_dir`: Output directory for the prepared CHiME-6 dataset (required)
- `--verbose`: Enable verbose logging (optional)

## What the Script Does

### Stage 0: Array Synchronization

The script implements the key components of Kaldi's stage 0 process:

1. **Downloads synchronization tools**:
   - CHiME-6 synchronization software from GitHub
   - Pre-computed audio edits JSON file (`chime6_audio_edits.json`)

2. **Validates CHiME-5 data structure**:
   - Checks for required directories (`audio/`, `transcriptions/`)
   - Verifies data integrity

3. **Applies synchronization fixes**:
   - Frame-drop compensation using pre-computed locations
   - Clock-drift correction using cross-correlation analysis
   - Uses SoX for audio processing

4. **Extracts binaural recordings**:
   - Copies `*_P??.wav` files (binaural recordings)
   - Maintains directory structure for train/dev/eval splits
   - Preserves corresponding transcription files

### Output Structure

The script creates the following directory structure:

```
CHiME6/
├── audio/
│   ├── train/
│   │   ├── S01/
│   │   │   ├── S01_U01_P05.wav
│   │   │   ├── S01_U01_P09.wav
│   │   │   └── ...
│   │   └── ...
│   ├── dev/
│   └── eval/
├── transcriptions/
│   ├── train/
│   ├── dev/
│   └── eval/
├── binaural_manifest.json
└── dataset_info.json
```

## Understanding the Data

### Binaural Recordings

- **Pattern**: `*_P??.wav` where `??` represents participant IDs
- **Format**: 16kHz, 2-channel (binaural)
- **Source**: Recordings from speakers with the lowest ID numbers (used as reference)
- **Purpose**: These provide the most reliable audio for speech recognition

### Synchronization Process

The synchronization addresses two key issues:

1. **Frame-dropping**: 
   - Affects Kinect devices
   - Compensated by inserting zeros where samples were dropped
   - Locations pre-computed and stored in `chime6_audio_edits.json`

2. **Clock-drift**:
   - Affects all recording devices
   - Computed by cross-correlation with reference binaural recordings
   - Corrected using SoX speed adjustment commands
   - Typically very subtle adjustments (<100ms over 2.5 hours)

### Dataset Splits

- **Train**: Training sessions for model development
- **Dev**: Development/validation sessions for hyperparameter tuning
- **Eval**: Evaluation sessions for final performance assessment

## Integration with Whisper

After preparation, you can use the CHiME-6 dataset with Whisper:

```python
import whisper
import json
from pathlib import Path

# Load the dataset manifest
with open('CHiME6/binaural_manifest.json', 'r') as f:
    manifest = json.load(f)

# Load Whisper model
model = whisper.load_model("base")

# Process audio files
for split in ['dev', 'eval']:  # Use dev/eval for evaluation
    for audio_file in manifest[split]:
        audio_path = Path('CHiME6/audio') / split / audio_file
        result = model.transcribe(str(audio_path))
        print(f"File: {audio_file}")
        print(f"Transcription: {result['text']}")
```

## Troubleshooting

### Common Issues

1. **SoX not found**:
   ```bash
   # Install SoX
   brew install sox  # macOS
   sudo apt-get install sox  # Ubuntu
   ```

2. **CHiME-5 data structure issues**:
   - Ensure you have the complete CHiME-5 dataset
   - Check that `audio/` and `transcriptions/` directories exist
   - Verify you have train/dev/eval splits

3. **Permission errors**:
   - Ensure write permissions to output directory
   - Check available disk space (CHiME-6 will be similar size to CHiME-5)

4. **Network issues**:
   - The script downloads synchronization tools from GitHub
   - Ensure internet connectivity
   - Check firewall settings if downloads fail

### Validation

After preparation, validate the output:

```bash
# Check directory structure
ls -la CHiME6/

# Count binaural files
find CHiME6/audio -name "*_P??.wav" | wc -l

# Check manifest
cat CHiME6/binaural_manifest.json

# Verify audio file properties
soxi CHiME6/audio/dev/*/S*_P*.wav | head -20
```

## References

1. **CHiME-6 Challenge**: [Official Website](https://chimechallenge.github.io/chime6/)
2. **Kaldi Recipe**: [s5_track1](https://github.com/kaldi-asr/kaldi/tree/master/egs/chime6/s5_track1)
3. **Synchronization Tool**: [CHiME6-synchronisation](https://github.com/chimechallenge/CHiME6-synchronisation)
4. **CHiME-5 Dataset**: [Download Page](https://spandh.dcs.shef.ac.uk//chime_challenge/CHiME5/download.html)

## Citation

If you use this prepared dataset, please cite the original CHiME-6 challenge:

```bibtex
@inproceedings{watanabe2020chime6,
  title={CHiME-6 Challenge: Tackling Multispeaker Speech Recognition for Unsegmented Recordings},
  author={Watanabe, Shinji and Mandel, Michael and Barker, Jon and Vincent, Emmanuel and Arora, Ashish and Chang, Xuankai and Fonseca, Piotr and Hannun, Awni and Kanda, Naoyuki and Kilgour, Kevin and others},
  booktitle={The 6th International Workshop on Speech Processing in Everyday Environments (CHiME 2020)},
  year={2020}
}
```

## License

This preparation script is provided under the same license as the Whisper project. The CHiME-5/6 datasets have their own licensing terms that must be respected.