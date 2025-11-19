# LungSounds-Model-Prediction

This repository contains tools for lung sound classification and visualization.

## Original Purpose
Classifies lung sounds (Normal / Wheeze / Crackle). Logs events with simulated timestamps. Visualizes trends, spikes, and alerts in a dashboard. Demonstrates how real-time monitoring could work in a hospital setting.

## New Addition: Lung Audio Visualization Tool

Python scripts to load and visualize lung sound audio data from WAV files using pandas and numpy.

### Files

- `lung_audio_visualizer.py` - Complete visualization tool with advanced features
- `simple_audio_example.py` - Simple example script for basic visualization
- `requirements.txt` - Required Python packages

### Setup

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

### Data Structure

The scripts expect lung audio files in this format:
- **Location**: `C:\Users\Arnav\Downloads\LungData_audios\Audio Files\`
- **Format**: WAV files
- **Naming Convention**: `BP###_Condition,SoundType,Position,Age,Gender.wav`

Example: `BP100_N,N,P R M,70,F.wav`
- BP100: Patient ID
- N: Condition (Normal)
- N: Sound type
- P R M: Position (Posterior Right Middle)
- 70: Age
- F: Gender (Female)

### Usage

#### Simple Example
Run the basic visualization script:
```bash
python simple_audio_example.py
```

This will:
- Load all WAV files from the lung data folder
- Create a pandas DataFrame with audio information
- Display summary statistics
- Visualize the first audio file as a line graph
- Show multiple lung sounds for comparison

#### Advanced Tool
Run the complete visualization tool:
```bash
python lung_audio_visualizer.py
```

### Features

#### Data Analysis
- **Audio Information**: Duration, sample rate, amplitude statistics
- **Medical Data**: Condition, age, gender, position parsing
- **Pandas Integration**: DataFrame with all audio metadata
- **Statistical Summary**: Condition distribution, age analysis

#### Visualizations
- **Line Graphs**: Time-domain audio waveforms (similar to Audio_Recorder.py)
- **Spectrograms**: Frequency analysis over time
- **Condition Comparison**: Side-by-side visualization of different lung conditions
- **Statistical Charts**: Distribution plots, box plots, bar charts

#### Supported Conditions
- Normal (N)
- Asthma
- COPD (Chronic Obstructive Pulmonary Disease)
- Heart Failure
- Pneumonia
- Lung Fibrosis
- Bronchitis (BRON)
- And more...

### Data Insights

The lung audio dataset contains:
- 336+ audio files
- Average duration: ~17 seconds
- Sample rate: 4000 Hz
- 14 different medical conditions
- Age range from 12 to 90 years
- Both male and female patients

Perfect for machine learning model training and audio analysis research!
