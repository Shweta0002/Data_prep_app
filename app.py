import streamlit as st
import numpy as np
import os
import io
from librosa import load
from soundfile import write
import soundfile as sf

# Function to process and add noise to clean audio
def process_audio(clean_audio_file, noise_audio_file, snr):
    # Load clean audio and noise
    clean_audio, sr_clean = load(io.BytesIO(clean_audio_file.read()))
    noise, sr_noise = load(io.BytesIO(noise_audio_file.read()))

    # Ensure both audio files have the same number of channels
    if clean_audio.ndim > 1 and noise.ndim == 1:
        noise = np.tile(noise[:, np.newaxis], (1, clean_audio.shape[1]))
    elif clean_audio.ndim == 1 and noise.ndim > 1:
        clean_audio = np.tile(clean_audio[:, np.newaxis], (1, noise.shape[1]))

    # Calculate power of clean audio
    clean_power = 1/len(clean_audio)*np.sum(clean_audio**2) 
    # st.write(f"Clean audio power: {clean_power}")

 
    # Select a segment of noise that matches the length of the clean audio
    start = np.random.randint(0, len(noise) - len(clean_audio))
    n = noise[start:start + len(clean_audio)]

    # Calculate noise power and adjust it to achieve the desired SNR
    noise_power = 1/len(n)*np.sum(n**2)
    # st.write(f"Noise segment power: {noise_power}")
    noise_power_target = clean_power * np.power(10, -snr / 10)
    # st.write(f"Target noise power: {noise_power_target}")
    k = noise_power_target / noise_power
    # st.write(f"Scaling factor: {k}")
    n = n * np.sqrt(k)
    x = clean_audio + n

    # Add the scaled noise to the clean audio
    noisy_audio = clean_audio + x
        
    
    fileaname = os.path.basename("random.wav")    
    output_dir = r'D:\LungSound Dataset\HF_Lung_V1-master\Prepared_Data\generated_noisy'
  
    write(os.path.join(output_dir, fileaname), x, sr_clean)

    # Convert the processed audio to a BytesIO object for downloading
    output_bytes = io.BytesIO()
    write(output_bytes, x, sr_clean, format='WAV')
    output_bytes.seek(0)

    return output_bytes, sr_clean

# Streamlit UI components
st.title("Audio Data Preparation App")

# Upload fields for audio files
original_audio_file = st.file_uploader("Upload Original Clean Audio", type=["wav"])
noise_audio_file = st.file_uploader("Upload Noise Audio", type=["wav"])
reference_audio_file = st.file_uploader("Upload Reference Noisy Audio (for listening)", type=["wav"])
snr_value = st.number_input("Enter SNR value (dB)", value=10)

if st.button("Process"):
    if original_audio_file and noise_audio_file:
        # Process audio by adding noise
        noisy_audio_bytes, sr = process_audio(original_audio_file, noise_audio_file, snr_value)

        if noisy_audio_bytes:
            # Display the processed audio
            st.audio(noisy_audio_bytes, format='audio/wav')
            st.download_button("Download Noisy Audio", noisy_audio_bytes, file_name="noisy_audio.wav")

    else:
        st.error("Please upload both the original and noise audio files.")

# Display the uploaded original and noisy audios for reference
# if original_audio_file:
#     st.subheader("Original Clean Audio")
#     st.audio(original_audio_file, format="audio/wav")

# if noise_audio_file:
#     st.subheader("Noise Audio")
#     st.audio(noise_audio_file, format="audio/wav")

# if reference_audio_file:
#     st.subheader("Reference Noisy Audio (for listening)")
#     st.audio(reference_audio_file, format="audio/wav")
audio_col1, audio_col2 = st.columns(2)

with audio_col1:
    if original_audio_file:
        st.subheader("Original Clean Audio")
        st.audio(original_audio_file, format="audio/wav")

with audio_col2:
    if noise_audio_file:
        st.subheader("Noise Audio")
        st.audio(noise_audio_file, format="audio/wav")

# Display reference audio in a single row
if reference_audio_file:
    st.subheader("Reference Noisy Audio (for listening)")
    st.audio(reference_audio_file, format="audio/wav")
