import time
from transformers import AutoModelForRNNT, AutoProcessor
from transformers.audio_utils import load_audio

# Load the model and processor
model_id = "nvidia/nemotron-3.5-asr-streaming-0.6b"
audio_path = r"D:\youtube\TheAIGuy\Speech\nemetron\shrinath-test-audio.mp3"
print("Loading model and processor...")
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForRNNT.from_pretrained(model_id, device_map="auto")

# Load the audio file
print(f"Loading audio from {audio_path}...")
print(f"Sampling rate: {processor.feature_extractor.sampling_rate} Hz")
audio = load_audio(
    audio_path,
    sampling_rate=processor.feature_extractor.sampling_rate,
)

# Run inference
print(f"Audio shape: {audio.shape}")
print("Running offline inference in device: ", model.device)
# Condition on a known language ...
start_time = time.time()
inputs = processor(audio, sampling_rate=processor.feature_extractor.sampling_rate, language="en-US")
inputs.to(model.device, dtype=model.dtype)
output = model.generate(**inputs, return_dict_in_generate=True)
end_time = time.time()
print(processor.decode(output.sequences, skip_special_tokens=True))
print(f"Inference time: {end_time - start_time:.2f} seconds")
