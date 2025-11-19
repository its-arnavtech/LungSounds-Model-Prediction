import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from pathlib import Path
import seaborn as sns

class LungAudioVisualizer:
    def __init__(self, audio_folder_path):
        """
        Initialize the LungAudioVisualizer with the path to the audio files.
        
        Args:
            audio_folder_path (str): Path to the folder containing lung audio files
        """
        self.audio_folder_path = Path(audio_folder_path)
        self.audio_data = []
        self.audio_info_df = None
        
    def parse_filename(self, filename):
        """
        Parse the filename to extract medical information.
        Example: BP100_N,N,P R M,70,F.wav
        Returns: dict with condition, sound_type, position, age, gender
        """
        try:
            # Remove .wav extension and split by underscores
            base_name = filename.replace('.wav', '')
            parts = base_name.split('_', 1)
            
            if len(parts) < 2:
                return None
                
            bp_id = parts[0]
            info_part = parts[1]
            
            # Split by commas
            info_parts = info_part.split(',')
            
            if len(info_parts) >= 5:
                condition = info_parts[0]
                sound_type = info_parts[1]
                position = info_parts[2]
                age = info_parts[3]
                gender = info_parts[4]
                
                return {
                    'bp_id': bp_id,
                    'condition': condition,
                    'sound_type': sound_type,
                    'position': position,
                    'age': age,
                    'gender': gender,
                    'filename': filename
                }
        except Exception as e:
            print(f"Error parsing filename {filename}: {e}")
        
        return None
    
    def load_audio_files(self):
        """
        Load all WAV files from the audio folder and extract metadata.
        """
        print("Loading audio files...")
        audio_files = list(self.audio_folder_path.glob("**/*.wav"))
        
        if not audio_files:
            print(f"No WAV files found in {self.audio_folder_path}")
            return
            
        audio_info_list = []
        
        for audio_file in audio_files:
            try:
                # Load audio data
                data, sample_rate = sf.read(str(audio_file))
                
                # Parse filename for metadata
                file_info = self.parse_filename(audio_file.name)
                
                if file_info:
                    file_info.update({
                        'file_path': str(audio_file),
                        'sample_rate': sample_rate,
                        'duration': len(data) / sample_rate,
                        'samples': len(data),
                        'max_amplitude': np.max(np.abs(data)),
                        'mean_amplitude': np.mean(np.abs(data)),
                        'std_amplitude': np.std(data)
                    })
                    
                    audio_info_list.append(file_info)
                    self.audio_data.append({
                        'info': file_info,
                        'data': data,
                        'sample_rate': sample_rate
                    })
                    
            except Exception as e:
                print(f"Error loading {audio_file}: {e}")
        
        # Create DataFrame with audio information
        self.audio_info_df = pd.DataFrame(audio_info_list)
        print(f"Loaded {len(self.audio_data)} audio files successfully.")
        
        return self.audio_info_df
    
    def visualize_single_audio(self, audio_index=0, max_duration=10):
        """
        Visualize a single audio file as a line graph.
        
        Args:
            audio_index (int): Index of audio file to visualize
            max_duration (float): Maximum duration to plot (in seconds)
        """
        if not self.audio_data:
            print("No audio data loaded. Please run load_audio_files() first.")
            return
            
        if audio_index >= len(self.audio_data):
            print(f"Audio index {audio_index} out of range. Available: 0-{len(self.audio_data)-1}")
            return
            
        audio_item = self.audio_data[audio_index]
        data = audio_item['data']
        sample_rate = audio_item['sample_rate']
        info = audio_item['info']
        
        # Limit duration if requested
        max_samples = int(max_duration * sample_rate)
        if len(data) > max_samples:
            data = data[:max_samples]
            
        # Generate time array
        duration = len(data) / sample_rate
        time = np.linspace(0, duration, len(data))
        
        # Create the plot
        plt.figure(figsize=(15, 8))
        
        # Main waveform plot
        plt.subplot(2, 1, 1)
        plt.plot(time, data, linewidth=0.8, alpha=0.8, color='blue')
        plt.title(f"Lung Sound Waveform - {info['condition']} (File: {info['filename']})", 
                 fontsize=14, fontweight='bold')
        plt.xlabel("Time (seconds)")
        plt.ylabel("Amplitude")
        plt.grid(True, alpha=0.3)
        
        # Add metadata text
        metadata_text = f"Condition: {info['condition']} | Position: {info['position']} | "
        metadata_text += f"Age: {info['age']} | Gender: {info['gender']} | Duration: {duration:.2f}s"
        plt.figtext(0.5, 0.02, metadata_text, ha='center', fontsize=10, 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        
        # Spectrogram subplot
        plt.subplot(2, 1, 2)
        plt.specgram(data, Fs=sample_rate, cmap='viridis', alpha=0.8)
        plt.title("Spectrogram")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Frequency (Hz)")
        plt.colorbar(label='Power (dB)')
        
        plt.tight_layout()
        plt.show()
        
        # Print audio statistics
        print(f"\nAudio Statistics for {info['filename']}:")
        print(f"Sample Rate: {sample_rate} Hz")
        print(f"Duration: {duration:.2f} seconds")
        print(f"Max Amplitude: {info['max_amplitude']:.4f}")
        print(f"Mean Amplitude: {info['mean_amplitude']:.4f}")
        print(f"Standard Deviation: {info['std_amplitude']:.4f}")
    
    def visualize_multiple_conditions(self, conditions=None, max_files_per_condition=3):
        """
        Visualize multiple audio files grouped by medical condition.
        
        Args:
            conditions (list): List of conditions to visualize. If None, uses top conditions.
            max_files_per_condition (int): Maximum files to show per condition
        """
        if self.audio_info_df is None or self.audio_info_df.empty:
            print("No audio data loaded. Please run load_audio_files() first.")
            return
            
        if conditions is None:
            # Get most common conditions
            conditions = self.audio_info_df['condition'].value_counts().head(4).index.tolist()
        
        fig, axes = plt.subplots(len(conditions), 1, figsize=(15, 4*len(conditions)))
        if len(conditions) == 1:
            axes = [axes]
            
        for i, condition in enumerate(conditions):
            condition_files = self.audio_info_df[self.audio_info_df['condition'] == condition]
            
            # Select up to max_files_per_condition files
            selected_files = condition_files.head(max_files_per_condition)
            
            ax = axes[i]
            colors = plt.cm.tab10(np.linspace(0, 1, len(selected_files)))
            
            for j, (_, file_info) in enumerate(selected_files.iterrows()):
                # Find corresponding audio data
                audio_item = next((item for item in self.audio_data 
                                 if item['info']['filename'] == file_info['filename']), None)
                
                if audio_item:
                    data = audio_item['data']
                    sample_rate = audio_item['sample_rate']
                    
                    # Limit to first 5 seconds for comparison
                    max_samples = int(5 * sample_rate)
                    if len(data) > max_samples:
                        data = data[:max_samples]
                    
                    time = np.linspace(0, len(data)/sample_rate, len(data))
                    
                    # Normalize and offset for better visualization
                    normalized_data = data / np.max(np.abs(data)) + j * 2.5
                    
                    ax.plot(time, normalized_data, color=colors[j], 
                           label=f"{file_info['bp_id']} (Age: {file_info['age']}, {file_info['gender']})",
                           alpha=0.8, linewidth=0.8)
            
            ax.set_title(f"Condition: {condition}", fontsize=12, fontweight='bold')
            ax.set_xlabel("Time (seconds)")
            ax.set_ylabel("Normalized Amplitude (offset)")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def create_summary_statistics(self):
        """
        Create summary statistics and visualizations of the audio dataset.
        """
        if self.audio_info_df is None or self.audio_info_df.empty:
            print("No audio data loaded. Please run load_audio_files() first.")
            return
            
        # Convert age to numeric
        self.audio_info_df['age_numeric'] = pd.to_numeric(self.audio_info_df['age'], errors='coerce')
        
        print("Dataset Summary:")
        print("="*50)
        print(f"Total audio files: {len(self.audio_info_df)}")
        print(f"Unique conditions: {self.audio_info_df['condition'].nunique()}")
        print(f"Age range: {self.audio_info_df['age_numeric'].min():.0f} - {self.audio_info_df['age_numeric'].max():.0f} years")
        print(f"Gender distribution: {self.audio_info_df['gender'].value_counts().to_dict()}")
        
        # Create visualization dashboard
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Condition distribution
        condition_counts = self.audio_info_df['condition'].value_counts()
        axes[0, 0].bar(range(len(condition_counts)), condition_counts.values, color='skyblue')
        axes[0, 0].set_xticks(range(len(condition_counts)))
        axes[0, 0].set_xticklabels(condition_counts.index, rotation=45, ha='right')
        axes[0, 0].set_title("Distribution of Medical Conditions")
        axes[0, 0].set_ylabel("Number of Files")
        
        # Age distribution by condition
        sns.boxplot(data=self.audio_info_df, x='condition', y='age_numeric', ax=axes[0, 1])
        axes[0, 1].set_title("Age Distribution by Condition")
        axes[0, 1].set_xticklabels(axes[0, 1].get_xticklabels(), rotation=45, ha='right')
        
        # Duration distribution
        axes[1, 0].hist(self.audio_info_df['duration'], bins=20, color='lightgreen', alpha=0.7)
        axes[1, 0].set_title("Audio Duration Distribution")
        axes[1, 0].set_xlabel("Duration (seconds)")
        axes[1, 0].set_ylabel("Frequency")
        
        # Amplitude statistics by condition
        condition_amplitude = self.audio_info_df.groupby('condition')['max_amplitude'].mean().sort_values(ascending=False)
        axes[1, 1].bar(range(len(condition_amplitude)), condition_amplitude.values, color='lightcoral')
        axes[1, 1].set_xticks(range(len(condition_amplitude)))
        axes[1, 1].set_xticklabels(condition_amplitude.index, rotation=45, ha='right')
        axes[1, 1].set_title("Average Max Amplitude by Condition")
        axes[1, 1].set_ylabel("Max Amplitude")
        
        plt.tight_layout()
        plt.show()
        
        return self.audio_info_df.describe()

def main():
    """
    Main function to demonstrate the lung audio visualization functionality.
    """
    # Path to the lung audio data
    audio_folder = r"C:\Users\Arnav\Downloads\LungData_audios\Audio Files"
    
    print("Lung Sound Audio Visualization Tool")
    print("="*40)
    
    # Initialize visualizer
    visualizer = LungAudioVisualizer(audio_folder)
    
    # Load audio files
    audio_info_df = visualizer.load_audio_files()
    
    if audio_info_df is not None and not audio_info_df.empty:
        print("\nDataset loaded successfully!")
        
        # Show summary statistics
        print("\nGenerating summary statistics...")
        stats = visualizer.create_summary_statistics()
        
        # Visualize first audio file
        print("\nVisualizing first audio file...")
        visualizer.visualize_single_audio(audio_index=0)
        
        # Visualize multiple conditions
        print("\nVisualizing multiple conditions...")
        visualizer.visualize_multiple_conditions()
        
        print("\nVisualization complete!")
        print(f"Audio info DataFrame shape: {audio_info_df.shape}")
        print("\nFirst few rows of the dataset:")
        print(audio_info_df[['filename', 'condition', 'age', 'gender', 'duration']].head())
        
    else:
        print("Failed to load audio data. Please check the folder path.")

if __name__ == "__main__":
    main()