import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import os
from pathlib import Path

def load_and_visualize_lung_audio(audio_file_path):
    """
    Load a single lung audio file and visualize it similar to Audio_Recorder.py
    
    Args:
        audio_file_path (str): Path to the WAV file
    """
    try:
        # Load audio data using soundfile (similar to sounddevice)
        data, sample_rate = sf.read(audio_file_path)
        
        # Calculate duration
        duration = len(data) / sample_rate
        
        # Generate time array (similar to Audio_Recorder.py)
        time = np.linspace(0, duration, len(data))
        
        # Create the plot (similar to Audio_Recorder.py style)
        plt.figure(figsize=(12, 6))
        plt.plot(time, data, color='blue', linewidth=0.8)
        
        # Extract filename for title
        filename = Path(audio_file_path).name
        plt.title(f"Lung Sound Wave - {filename}")
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude")
        plt.grid()
        
        # Print audio info
        print(f"File: {filename}")
        print(f"Duration: {duration:.2f} seconds")
        print(f"Sample Rate: {sample_rate} Hz")
        print(f"Max Amplitude: {np.max(np.abs(data)):.4f}")
        
        plt.tight_layout()
        plt.show()
        
        return data, sample_rate, time
        
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return None, None, None

def create_lung_audio_dataframe(audio_folder_path):
    """
    Create a pandas DataFrame with information about all lung audio files
    
    Args:
        audio_folder_path (str): Path to folder containing WAV files
    
    Returns:
        pandas.DataFrame: DataFrame with audio file information
    """
    audio_folder = Path(audio_folder_path)
    audio_files = list(audio_folder.glob("*.wav"))
    
    if not audio_files:
        print(f"No WAV files found in {audio_folder_path}")
        return pd.DataFrame()
    
    data_list = []
    
    for audio_file in audio_files:
        try:
            # Load basic audio info
            data, sample_rate = sf.read(str(audio_file))
            duration = len(data) / sample_rate
            
            # Parse filename for medical info (basic parsing)
            filename = audio_file.name
            
            # Extract condition from filename (first part after underscore)
            if '_' in filename:
                condition_part = filename.split('_')[1].split(',')[0]
            else:
                condition_part = "Unknown"
            
            # Create data dictionary
            file_info = {
                'filename': filename,
                'condition': condition_part,
                'duration_seconds': duration,
                'sample_rate': sample_rate,
                'total_samples': len(data),
                'max_amplitude': np.max(np.abs(data)),
                'mean_amplitude': np.mean(np.abs(data)),
                'std_amplitude': np.std(data),
                'file_path': str(audio_file)
            }
            
            data_list.append(file_info)
            
        except Exception as e:
            print(f"Error processing {audio_file}: {e}")
    
    return pd.DataFrame(data_list)

def visualize_multiple_lung_sounds(df, num_files=5):
    """
    Visualize multiple lung sounds from the DataFrame
    
    Args:
        df (pandas.DataFrame): DataFrame with audio file information
        num_files (int): Number of files to visualize
    """
    if df.empty:
        print("DataFrame is empty!")
        return
    
    # Select first few files
    files_to_plot = df.head(num_files)
    
    fig, axes = plt.subplots(num_files, 1, figsize=(15, 3*num_files))
    if num_files == 1:
        axes = [axes]
    
    for i, (_, row) in enumerate(files_to_plot.iterrows()):
        try:
            # Load audio data
            data, sample_rate = sf.read(row['file_path'])
            duration = len(data) / sample_rate
            time = np.linspace(0, duration, len(data))
            
            # Plot
            axes[i].plot(time, data, color=f'C{i}', linewidth=0.8)
            axes[i].set_title(f"{row['filename']} - Condition: {row['condition']}")
            axes[i].set_xlabel("Time [s]")
            axes[i].set_ylabel("Amplitude")
            axes[i].grid(True, alpha=0.3)
            
        except Exception as e:
            print(f"Error plotting {row['filename']}: {e}")
    
    plt.tight_layout()
    plt.show()

def main():
    """
    Main function - demonstrates lung audio visualization
    """
    print("Simple Lung Audio Visualization")
    print("=" * 35)
    
    # Path to lung audio data
    audio_folder = r"C:\Users\Arnav\Downloads\LungData_audios\Audio Files"
    
    # Check if folder exists
    if not os.path.exists(audio_folder):
        print(f"Audio folder not found: {audio_folder}")
        print("Please check the path and try again.")
        return
    
    # Create DataFrame with audio information
    print("Loading lung audio files...")
    df = create_lung_audio_dataframe(audio_folder)
    
    if df.empty:
        print("No audio files found!")
        return
    
    print(f"Loaded {len(df)} audio files")
    
    # Display basic statistics using pandas
    print("\nDataset Overview:")
    print("=" * 20)
    print(f"Total files: {len(df)}")
    print(f"Average duration: {df['duration_seconds'].mean():.2f} seconds")
    print(f"Conditions found: {df['condition'].nunique()}")
    print(f"Sample rates: {df['sample_rate'].unique()}")
    
    # Show condition distribution
    print("\nCondition Distribution:")
    print(df['condition'].value_counts().head(10))
    
    # Show first few rows
    print("\nFirst 5 files:")
    print(df[['filename', 'condition', 'duration_seconds', 'max_amplitude']].head())
    
    # Visualize first audio file in detail
    if len(df) > 0:
        first_file_path = df.iloc[0]['file_path']
        print(f"\nVisualizing first audio file: {df.iloc[0]['filename']}")
        load_and_visualize_lung_audio(first_file_path)
    
    # Visualize multiple files
    print("\nVisualizing multiple lung sounds...")
    visualize_multiple_lung_sounds(df, num_files=min(3, len(df)))

if __name__ == "__main__":
    main()