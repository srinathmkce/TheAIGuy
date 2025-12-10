"""
Script to merge audio from srinathmkce/wiki-ai-audio dataset with background noise.
The noise is added as a subtle background sound that doesn't overpower the main audio.
"""

import os
import random
from datasets import load_dataset
from pydub import AudioSegment
import numpy as np
from tqdm import tqdm

# Check for required audio libraries (needed by datasets library for audio decoding)
import librosa  # noqa: F401
import soundfile  # noqa: F401


def audio_data_to_segment(audio_data):
    """
    Convert audio data from dataset to pydub AudioSegment.

    Args:
        audio_data: AudioDecoder object with 'array' and 'sampling_rate' keys

    Returns:
        AudioSegment: The converted audio as a pydub AudioSegment
    """
    # Convert numpy audio array to pydub AudioSegment
    # Handle both mono and stereo audio
    audio_array = audio_data["array"]
    sampling_rate = audio_data["sampling_rate"]

    # Convert to int16 if needed
    if audio_array.dtype != np.int16:
        # Normalize to int16 range
        if audio_array.dtype == np.float32 or audio_array.dtype == np.float64:
            audio_array = (audio_array * 32767).astype(np.int16)
        else:
            audio_array = audio_array.astype(np.int16)

    # Determine number of channels
    if len(audio_array.shape) == 1:
        channels = 1
    else:
        channels = audio_array.shape[1]

    # Create AudioSegment
    audio_segment = AudioSegment(
        audio_array.tobytes(),
        frame_rate=sampling_rate,
        sample_width=audio_array.dtype.itemsize,
        channels=channels,
    )

    return audio_segment


def load_noise_audio(noise_path="noise1/noise1.wav", verbose=False):
    """
    Load background noise audio file.

    Args:
        noise_path: Path to the noise audio file
        verbose: Whether to print progress messages (default: False)

    Returns:
        AudioSegment: The loaded noise audio
    """
    if not os.path.exists(noise_path):
        raise FileNotFoundError(f"Noise file not found: {noise_path}")

    if verbose:
        print(f"Loading noise audio from {noise_path}...")
    noise_audio = AudioSegment.from_file(noise_path, format="wav")

    if verbose:
        print(
            f"Noise audio loaded: {len(noise_audio) / 1000:.2f} seconds, "
            f"{noise_audio.frame_rate} Hz, {noise_audio.channels} channel(s)"
        )

    return noise_audio


def match_audio_properties(audio_segment, noise_audio, verbose=False):
    """
    Match noise audio properties (sample rate, channels) to main audio.

    Args:
        audio_segment: Main audio segment
        noise_audio: Noise audio segment
        verbose: Whether to print progress messages (default: False)

    Returns:
        AudioSegment: Noise audio with matched properties
    """
    # Match sample rate
    if noise_audio.frame_rate != audio_segment.frame_rate:
        if verbose:
            print(
                f"Converting noise sample rate from {noise_audio.frame_rate} Hz "
                f"to {audio_segment.frame_rate} Hz..."
            )
        noise_audio = noise_audio.set_frame_rate(audio_segment.frame_rate)

    # Match channels
    if noise_audio.channels != audio_segment.channels:
        if verbose:
            print(
                f"Converting noise channels from {noise_audio.channels} "
                f"to {audio_segment.channels}..."
            )
        if audio_segment.channels == 1:
            noise_audio = noise_audio.set_channels(1)
        else:
            noise_audio = noise_audio.set_channels(2)

    return noise_audio


def match_audio_length(noise_audio, target_length_ms, verbose=False):
    """
    Match noise audio length to target length by repeating or truncating.

    Args:
        noise_audio: Noise audio segment
        target_length_ms: Target length in milliseconds
        verbose: Whether to print progress messages (default: False)

    Returns:
        AudioSegment: Noise audio with matched length
    """
    noise_length_ms = len(noise_audio)

    if noise_length_ms < target_length_ms:
        # Repeat noise to match length
        repeats = (target_length_ms // noise_length_ms) + 1
        noise_extended = noise_audio * repeats
        noise_matched = noise_extended[:target_length_ms]
        if verbose:
            print(
                f"Extended noise from {noise_length_ms / 1000:.2f}s to "
                f"{target_length_ms / 1000:.2f}s by repeating"
            )
    else:
        # Truncate if noise is longer
        noise_matched = noise_audio[:target_length_ms]
        if verbose:
            print(
                f"Truncated noise from {noise_length_ms / 1000:.2f}s to "
                f"{target_length_ms / 1000:.2f}s"
            )

    return noise_matched


def merge_audio_with_noise(audio_segment, noise_audio, noise_db_reduction=15, verbose=False):
    """
    Merge main audio with background noise.

    Args:
        audio_segment: Main audio segment
        noise_audio: Background noise audio segment
        noise_db_reduction: How much to reduce noise volume in dB (default: 15)
                          Lower values = louder/more audible noise. Typical range: 10-20 dB
        verbose: Whether to print progress messages (default: False)

    Returns:
        AudioSegment: Merged audio with noise in background
    """
    # Match audio properties
    noise_audio = match_audio_properties(audio_segment, noise_audio, verbose=verbose)

    # Match audio length
    target_length = len(audio_segment)
    noise_matched = match_audio_length(noise_audio, target_length, verbose=verbose)

    # Reduce noise volume to make it subtle background noise
    # Typical background noise should be 20-30 dB quieter than main audio
    noise_quiet = noise_matched - noise_db_reduction
    if verbose:
        print(f"Reduced noise volume by {noise_db_reduction} dB for background effect")

    # Overlay noise on the audio segment
    merged_audio = audio_segment.overlay(noise_quiet)

    return merged_audio


def save_audio(audio_segment, output_path, verbose=False):
    """
    Save audio segment to file.

    Args:
        audio_segment: Audio segment to save
        output_path: Output file path
        verbose: Whether to print progress messages (default: False)
    """
    # Ensure output directory exists
    os.makedirs(
        os.path.dirname(output_path) if os.path.dirname(output_path) else ".",
        exist_ok=True,
    )

    if verbose:
        print(f"Saving audio to {output_path}...")
    audio_segment.export(output_path, format="wav")
    if verbose:
        print("Audio saved successfully!")


def get_random_noise_file():
    """
    Get a random noise file path from noise1.wav to noise10.wav.

    Returns:
        str: Path to a random noise file
    """
    noise_number = random.randint(1, 10)
    noise_path = f"noise1/noise{noise_number}.wav"
    return noise_path


def process_single_audio(sample, sample_index, output_dir, noise_db_reduction=15):
    """
    Process a single audio sample: merge with random noise and save both original and merged.

    Args:
        sample: Dataset sample containing audio, id, and title
        sample_index: Index of the sample in the dataset
        output_dir: Directory where output files will be saved
        noise_db_reduction: How much to reduce noise volume in dB (default: 15)
                           Lower values = louder/more audible noise

    Returns:
        tuple: (success: bool, error_message: str or None)
    """
    try:
        # Get sample ID for filename
        sample_id = sample.get("id", f"sample_{sample_index}")
        
        # Convert audio data to AudioSegment
        audio_data = sample["audio"]
        audio_segment = audio_data_to_segment(audio_data)

        # Get random noise file
        noise_path = get_random_noise_file()
        noise_audio = load_noise_audio(noise_path, verbose=False)

        # Merge audio with noise
        merged_audio = merge_audio_with_noise(
            audio_segment, noise_audio, noise_db_reduction=noise_db_reduction, verbose=False
        )

        # Generate output filenames
        original_output_path = os.path.join(output_dir, f"{sample_id}_original.wav")
        merged_output_path = os.path.join(output_dir, f"{sample_id}_merged.wav")

        # Save both original and merged audio
        save_audio(audio_segment, original_output_path, verbose=False)
        save_audio(merged_audio, merged_output_path, verbose=False)

        return True, None
    except Exception as e:
        return False, str(e)


def main(
    output_dir="temp/merged_audios",
    noise_db_reduction=15,
    process_all=True,
    sample_index=None,
):
    """
    Main function to merge audio from dataset with background noise.

    Args:
        output_dir: Directory where output files will be saved (default: "temp/merged_audios")
        noise_db_reduction: How much to reduce noise volume in dB (default: 15)
                           Lower values = louder/more audible noise. Typical range: 10-20 dB
        process_all: If True, process all samples; if False, process only sample_index (default: True)
        sample_index: Index of a single sample to process (only used if process_all=False)

    Returns:
        int: Exit code (0 for success, 1 for error)
    """
    print("=" * 60)
    print("Audio Merging Script - Dataset + Background Noise")
    print("=" * 60)

    # Load dataset
    print("\nLoading dataset from Hugging Face...")
    dataset = load_dataset("srinathmkce/wiki-ai-audio", split="train")
    print(f"Dataset loaded. Total samples: {len(dataset)}")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Determine which samples to process
    if process_all:
        samples_to_process = range(len(dataset))
        print(f"\nProcessing all {len(dataset)} audio samples...")
    else:
        if sample_index is None:
            sample_index = 0
        samples_to_process = [sample_index]
        print(f"\nProcessing single audio sample at index {sample_index}...")

    # Process samples with progress bar
    successful = 0
    failed = 0
    failed_samples = []

    for i in tqdm(samples_to_process, desc="Processing audios"):
        sample = dataset[i]
        success, error = process_single_audio(
            sample, i, output_dir, noise_db_reduction=noise_db_reduction
        )
        
        if success:
            successful += 1
        else:
            failed += 1
            failed_samples.append((i, error))
            tqdm.write(f"Error processing sample {i}: {error}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUCCESS! Audio merging completed.")
    print("=" * 60)
    print("\nSummary:")
    print(f"  Total samples processed: {len(samples_to_process)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Output directory: {output_dir}")
    print(f"  Noise volume reduction: {noise_db_reduction} dB")
    
    if failed_samples:
        print("\nFailed samples:")
        for idx, error in failed_samples:
            print(f"  Sample {idx}: {error}")

    return 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Merge audio from srinathmkce/wiki-ai-audio dataset with background noise"
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=None,
        help="Index of a single audio sample to process (if not provided, processes all samples)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="temp/merged_audios",
        help="Directory where output files will be saved (default: temp/merged_audios)",
    )
    parser.add_argument(
        "--noise-db",
        type=float,
        default=10.0,
        help="Noise volume reduction in dB. Lower = louder/more audible noise (default: 15.0, range: 10-20 recommended)",
    )

    args = parser.parse_args()

    # Determine if processing all or single sample
    process_all = args.sample_index is None

    exit(
        main(
            output_dir=args.output_dir,
            noise_db_reduction=args.noise_db,
            process_all=process_all,
            sample_index=args.sample_index,
        )
    )
