import time
import torch
from transformers import Qwen2_5OmniModel, Qwen2_5OmniProcessor
from inference import inference, clear_gpu_cache
import librosa
import soundfile as sf
import argparse
import numpy as np
import logging
import warnings
import os
import gc
import tempfile
import shutil
from tqdm import tqdm

# Suppress unnecessary warnings
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("root").setLevel(logging.ERROR)

model_path = "Qwen/Qwen2.5-Omni-7B"
cache_dir = "./models"
audio_path = "./wS8dibvhlv4.mp3"
prompt = 'Transcribe the Russian audio into text with correct punctuation. The audio is from a single speaker. Write in natural, readable Russian.'
sys_prompt = 'You are a highly accurate speech recognition model specialized in transcribing single-speaker Russian audio. Your transcription must include correct punctuation and be easy to read, preserving the natural flow of conversation.'

def clear_gpu_memory():
    """Clear GPU memory cache to reduce memory usage."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, 'ipc_collect'):
            torch.cuda.ipc_collect()

def get_gpu_memory_info():
    """Get GPU memory usage information."""
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        total_memory = torch.cuda.get_device_properties(device).total_memory
        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        free_memory = total_memory - reserved
        
        total_gb = total_memory / (1024**3)
        allocated_gb = allocated / (1024**3)
        reserved_gb = reserved / (1024**3)
        free_gb = free_memory / (1024**3)
        
        return {
            "total_gb": round(total_gb, 2),
            "allocated_gb": round(allocated_gb, 2),
            "reserved_gb": round(reserved_gb, 2),
            "free_gb": round(free_gb, 2)
        }
    return None

def prepare_audio_chunks(audio_data, sr, chunk_duration=30, overlap_seconds=2, temp_dir=None):
    """Split audio into chunks with overlap and save to temp directory."""
    # Calculate chunk size in samples
    chunk_samples = int(chunk_duration * sr)
    overlap_samples = int(overlap_seconds * sr)
    step_samples = chunk_samples - overlap_samples
    
    # Calculate total number of chunks
    total_samples = len(audio_data)
    num_chunks = max(1, int(np.ceil((total_samples - overlap_samples) / step_samples)))
    
    chunk_paths = []
    for i in range(num_chunks):
        start_sample = i * step_samples
        end_sample = min(start_sample + chunk_samples, total_samples)
        
        # Extract chunk
        chunk_data = audio_data[start_sample:end_sample]
        
        # Create temp file
        chunk_file = os.path.join(temp_dir, f"chunk_{i+1}.wav")
        sf.write(chunk_file, chunk_data, sr, 'PCM_16')
        chunk_paths.append(chunk_file)
    
    return chunk_paths, num_chunks

def process_audio_in_chunks(audio_path, chunk_duration=30, model=None, processor=None, overlap_seconds=2, batch_size=8):
    """Process a long audio file in chunks and return the combined transcription."""
    # Load the audio file
    print(f"Loading audio file: {audio_path}")
    audio_data, sr = librosa.load(audio_path, sr=None, mono=True)
    total_duration_seconds = len(audio_data) / sr
    print(f"Total audio duration: {total_duration_seconds:.2f} seconds ({total_duration_seconds/60:.2f} minutes)")
    
    # Create temporary directory for audio chunks
    temp_dir = tempfile.mkdtemp(prefix="audio_chunks_")
    
    # Prepare audio chunks
    chunk_paths, num_chunks = prepare_audio_chunks(
        audio_data, sr, 
        chunk_duration=chunk_duration, 
        overlap_seconds=overlap_seconds,
        temp_dir=temp_dir
    )
    
    print(f"Processing audio in {num_chunks} chunks of {chunk_duration} seconds each (with {overlap_seconds}s overlap)")
    print(f"Using batch size of {batch_size}")
    
    # Process chunks in batches
    results = [None] * num_chunks
    
    # Process in batches
    for batch_start in range(0, num_chunks, batch_size):
        batch_end = min(batch_start + batch_size, num_chunks)
        batch_chunks = list(range(batch_start, batch_end))
        
        print(f"\nProcessing batch {batch_start//batch_size + 1}/{(num_chunks+batch_size-1)//batch_size} (chunks {batch_start+1}-{batch_end})")
        
        # Clear memory between batches
        clear_gpu_memory()
        memory_info = get_gpu_memory_info()
        if memory_info:
            print(f"GPU Memory before batch: Free: {memory_info['free_gb']}GB, Allocated: {memory_info['allocated_gb']}GB")
        
        # Process this batch sequentially
        for i in tqdm(batch_chunks, desc=f"Batch {batch_start//batch_size + 1}"):
            chunk_path = chunk_paths[i]
            chunk_prompt = f'Transcribe this segment of Russian audio (part {i+1}/{num_chunks}) into text with correct punctuation.'
            
            try:
                chunk_start_time = time.time()
                
                # Simple inference call without extra wrappers that might cause errors
                response = inference(
                    chunk_path, 
                    prompt=chunk_prompt, 
                    sys_prompt=sys_prompt, 
                    model=model, 
                    processor=processor
                )
                
                chunk_elapsed_time = time.time() - chunk_start_time
                
                # Extract the transcription from response
                transcription = response[0].split("assistant\n")[-1].strip() if response else ""
                
                results[i] = transcription
                
                # Print progress info
                print(f"\nChunk {i+1}/{num_chunks} processed in {chunk_elapsed_time:.2f} seconds")
                
                # Print preview of transcription
                if transcription:
                    preview = transcription[:100] + "..." if len(transcription) > 100 else transcription
                    print(f"Chunk {i+1}: {preview}")
                
                # Synchronize CUDA operations
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                
            except Exception as e:
                print(f"Error processing chunk {i+1}: {str(e)}")
                results[i] = f"[Error transcribing chunk {i+1}]"
        
        # Clean up between batches
        clear_gpu_memory()
        memory_info = get_gpu_memory_info()
        if memory_info:
            print(f"GPU Memory after batch: Free: {memory_info['free_gb']}GB, Allocated: {memory_info['allocated_gb']}GB")
    
    # Combine all transcriptions in the correct order
    complete_transcription = " ".join(filter(None, results))
    
    # Clean up temp directory
    shutil.rmtree(temp_dir, ignore_errors=True)
        
    return complete_transcription

def main():
    parser = argparse.ArgumentParser(description='Process audio files with Qwen2.5-Omni')
    parser.add_argument('--flash-attn', action='store_true', help='Use flash attention')
    parser.add_argument('--audio-file', type=str, default=audio_path, help='Path to the audio file')
    parser.add_argument('--chunk-duration', type=int, default=30, help='Duration of each chunk in seconds')
    parser.add_argument('--overlap', type=int, default=2, help='Overlap between chunks in seconds')
    parser.add_argument('--torch-dtype', type=str, default='float16', choices=['float16', 'bfloat16', 'float32'], 
                      help='Torch data type for model')
    parser.add_argument('--batch-size', type=int, default=8, help='Number of chunks to process in a batch')
    parser.add_argument('--optimize-memory', action='store_true', help='Enable aggressive memory optimization')
    args = parser.parse_args()

    start_time = time.time()

    # Set torch data type
    if args.torch_dtype == 'float16':
        torch_dtype = torch.float16
    elif args.torch_dtype == 'float32':
        torch_dtype = torch.float32
    else:
        torch_dtype = torch.bfloat16
    
    # Clear GPU memory before loading model
    clear_gpu_memory()
    
    print(f"Loading model with {args.torch_dtype} precision...")
    model = Qwen2_5OmniModel.from_pretrained(
        model_path,
        torch_dtype=torch_dtype, 
        device_map="auto",
        attn_implementation="flash_attention_2" if args.flash_attn else "sdpa",
        cache_dir=cache_dir,
        enable_audio_output=False,
        low_cpu_mem_usage=True,
    )
    
    processor = Qwen2_5OmniProcessor.from_pretrained(model_path, cache_dir=cache_dir)

    # Display initial memory status
    memory_info = get_gpu_memory_info()
    if memory_info:
        print(f"Initial GPU Memory: Total: {memory_info['total_gb']}GB, Free: {memory_info['free_gb']}GB, Allocated: {memory_info['allocated_gb']}GB")

    # Process the audio file in chunks
    complete_transcription = process_audio_in_chunks(
        args.audio_file, 
        chunk_duration=args.chunk_duration,
        overlap_seconds=args.overlap,
        model=model, 
        processor=processor,
        batch_size=args.batch_size
    )
    
    elapsed_time = time.time() - start_time
    print("\n==== FULL TRANSCRIPTION ====")
    print(complete_transcription)
    print(f"\nTotal transcription time: {elapsed_time:.2f} seconds")

    # Save the transcription to a file
    output_file = os.path.splitext(args.audio_file)[0] + "_transcription.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(complete_transcription)
    print(f"Transcription saved to: {output_file}")

    # Clean up resources
    del model, processor
    clear_gpu_memory()

if __name__ == "__main__":
    main()
