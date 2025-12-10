"""
Script to extract metadata from srinathmkce/wiki-ai-audio dataset.
Creates a CSV file with id, title, and duration (in hours:minutes:seconds format).
"""

from datasets import load_dataset
import pandas as pd

# Check for required audio libraries (needed by datasets library for audio decoding)
import librosa  # noqa: F401
import soundfile  # noqa: F401


def format_duration(seconds):
    """
    Convert seconds to hours:minutes:seconds format.
    
    Args:
        seconds: Duration in seconds (float)
    
    Returns:
        str: Formatted duration as "HH:MM:SS"
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def calculate_audio_duration(audio_data):
    """
    Calculate audio duration from audio data dictionary.
    
    Args:
        audio_data: AudioDecoder object or dictionary with 'array' and 'sampling_rate' keys
    
    Returns:
        float: Duration in seconds
    """
    if audio_data is None:
        return 0.0
    
    try:
        # Access audio data directly (AudioDecoder objects support dictionary-style access)
        audio_array = audio_data["array"]
        sampling_rate = audio_data["sampling_rate"]
        
        if audio_array is None or sampling_rate is None:
            return 0.0
        
        # Calculate duration: length of array divided by sampling rate
        duration_seconds = len(audio_array) / sampling_rate
        
        return duration_seconds
    except (KeyError, TypeError, AttributeError) as e:
        print(f"Error accessing audio data: {e}")
        return 0.0


def create_metadata_csv(output_path="metadata.csv"):
    """
    Load dataset, extract metadata, and save to CSV.
    
    Args:
        output_path: Path where the CSV file will be saved (default: "metadata.csv")
    
    Returns:
        pd.DataFrame: The created dataframe
    """
    print("=" * 60)
    print("Creating Metadata CSV from Dataset")
    print("=" * 60)
    
    # Load dataset
    print("\nLoading dataset from Hugging Face...")
    dataset = load_dataset("srinathmkce/wiki-ai-audio", split="train")
    
    print(f"Dataset loaded. Total samples: {len(dataset)}")
    print("\nExtracting metadata from each audio sample...")
    
    # Initialize lists to store data
    ids = []
    titles = []
    durations_formatted = []
    
    # Process each sample one by one
    for i, sample in enumerate(dataset):
        try:
            # Extract id and title
            sample_id = sample.get("id", f"sample_{i}")
            title = sample.get("title", "")
            
            # Calculate duration from audio
            # Access audio directly from sample (it's an AudioDecoder object)
            audio_data = sample["audio"]
            duration_sec = calculate_audio_duration(audio_data)
            duration_formatted = format_duration(duration_sec)
            
            # Store data
            ids.append(sample_id)
            titles.append(title)
            durations_formatted.append(duration_formatted)
            
            # Progress indicator
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(dataset)} samples...")
        except Exception as e:
            print(f"Warning: Error processing sample {i}: {e}")
            import traceback
            traceback.print_exc()
            # Add placeholder data for failed samples
            ids.append(sample.get("id", f"sample_{i}"))
            titles.append(sample.get("title", ""))
            durations_formatted.append("00:00:00")
    
    # Create dataframe
    print("\nCreating dataframe...")
    df = pd.DataFrame({
        "id": ids,
        "title": titles,
        "duration": durations_formatted
    })
    
    # Save to CSV
    print(f"\nSaving metadata to {output_path}...")
    df.to_csv(output_path, index=False)
    print("Metadata saved successfully!")
    
    # Print summary
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Total samples: {len(df)}")
    print("  First few rows:")
    print(df.head().to_string(index=False))
    print("=" * 60)
    
    return df


def main():
    """
    Main function to create metadata CSV.
    """
    try:
        create_metadata_csv("metadata.csv")
        return 0
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
