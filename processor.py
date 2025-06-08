import re
import pandas as pd
import cv2
import pytesseract
import os
from datetime import datetime
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
from functools import partial

def first_frame(video_path):
    capture = cv2.VideoCapture(video_path)
    ret, frame = capture.read()
    if ret:
        os.makedirs("static/frames", exist_ok=True)
        frame_output_path = os.path.join("static", "frames", "firstframe.png")
        cv2.imwrite(frame_output_path, frame)
        capture.release()  # Added release
        return frame_output_path
    capture.release()
    return None

def extract_number_optimized(text):
    numbers = re.findall(r'\d+\.?\d*', text)
    if numbers:
        try:
            value = float(numbers[0])  # Take first number found
            if 30 <= value <= 45:
                return round(value, 1)
            else:
                return 0
        except ValueError:
            return 0
    return 0

def process_roi_batch(frame, regions, tesseract_config):
    results = []
    
    for region in regions:
        try:
            x1, y1 = int(region["x"]), int(region["y"])
            w, h = int(region["width"]), int(region["height"])
            x2, y2 = x1 + w, y1 + h
            
            # Extract and preprocess ROI
            roi = frame[y1:y2, x1:x2]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # Optimized thresholding - try adaptive if simple fails
            _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
            
            # OCR with optimized config
            text = pytesseract.image_to_string(thresh, config=tesseract_config)
            value = extract_number_optimized(text)
            results.append(value)
            
        except Exception as e:
            print(f"Error processing region: {e}")
            results.append(0)
    
    return results

def process_video(video_path, region_coordinates, headings_inputs):
    
    filename = f"VideoData_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_path = os.path.join("results", filename + ".xlsx")
    graph_path = os.path.join("results", filename + ".png")
    
    # Ensure results directory exists
    os.makedirs("results", exist_ok=True)
    
    # Initialize video capture with optimizations
    capture = cv2.VideoCapture(video_path)
    
    # Set buffer size to reduce memory usage
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    fps = capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Adaptive frame skipping based on video length
    if total_frames > 1000:
        skip_frames = int(fps)  # Skip 1 second worth of frames for long videos
    else:
        skip_frames = max(1, int(fps // 2))  # Original logic for shorter videos
    
    print(f"Processing video: {total_frames} frames at {fps} FPS, skipping every {skip_frames} frames")
    
    # Pre-compile tesseract config
    tesseract_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789.'
    
    # Pre-allocate lists for better memory management
    num_regions = len(region_coordinates)
    region_texts = [[] for _ in range(num_regions)]
    
    frame_count = 0
    processed_frames = 0
    
    # Batch processing approach
    batch_size = 50  # Process frames in batches
    frame_batch = []
    
    while True:
        ret, frame = capture.read()
        if not ret:
            break
            
        if frame_count % skip_frames == 0:
            frame_batch.append(frame.copy())
            
            # Process batch when it's full or at end of video
            if len(frame_batch) >= batch_size:
                process_frame_batch(frame_batch, region_coordinates, region_texts, tesseract_config)
                frame_batch = []
                processed_frames += batch_size
                print(f"Processed {processed_frames} frames...")
        
        frame_count += 1
    
    # Process remaining frames in batch
    if frame_batch:
        process_frame_batch(frame_batch, region_coordinates, region_texts, tesseract_config)
    
    capture.release()
    cv2.destroyAllWindows()
    
    # Create DataFrame with proper error handling
    headings = [h.strip() for h in headings_inputs.split(",")]
    
    # Ensure we have enough headings
    while len(headings) < num_regions:
        headings.append(f"Region_{len(headings)+1}")
    
    data = {headings[i]: region_texts[i] for i in range(min(len(headings), num_regions))}
    df = pd.DataFrame(data)
    
    # Vectorized data cleaning - much faster than iterating
    for column in df.columns:
        column_data = df[column]
        non_zero_mask = column_data != 0
        
        if non_zero_mask.any():
            avg_value = column_data[non_zero_mask].mean()
            df[column] = column_data.where(non_zero_mask, avg_value)
    
    # Save to Excel
    df.to_excel(output_path, index=False)
    
    # Optimized plotting
    create_optimized_plot(df, graph_path)
    
    print(f"Processing complete: {output_path}, {graph_path}")
    return output_path, graph_path

def process_frame_batch(frame_batch, region_coordinates, region_texts, tesseract_config):
    for frame in frame_batch:
        results = process_roi_batch(frame, region_coordinates, tesseract_config)
        for i, value in enumerate(results):
            region_texts[i].append(value)

def create_optimized_plot(df, graph_path):
    plt.figure(figsize=(12, 8))
    
    # Use more efficient plotting
    for column in df.columns:
        plt.plot(df.index, df[column], label=column, linewidth=1.5)
    
    plt.title('Temperature Readings Over Time', fontsize=14)
    plt.xlabel('Time (frame intervals)', fontsize=12)
    plt.ylabel('Temperature (°C)', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Save with optimization
    plt.savefig(graph_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()

# Parallel frame processing version for maximum performance
def process_video_parallel(video_path, region_coordinates, headings_inputs, max_workers=None):
    
    if max_workers is None:
        max_workers = min(mp.cpu_count(), 8)  # Limit to prevent overwhelming system
    
    filename = f"VideoData_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_path = os.path.join("results", filename + ".xlsx")
    graph_path = os.path.join("results", filename + ".png")
    
    os.makedirs("results", exist_ok=True)
    
    # Step 1: Extract all frames that need processing (fast, sequential)
    print("Extracting frames for processing...")
    frames_data = extract_frames_with_indices(video_path)
    
    if not frames_data:
        print("No frames to process")
        return None, None
    
    print(f"Extracted {len(frames_data)} frames for parallel processing using {max_workers} workers")
    
    # Step 2: Process all frames in parallel (this is where the magic happens!)
    tesseract_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789.'
    
    # Create a partial function with fixed parameters
    process_single_frame = partial(
        process_frame_with_regions, 
        regions=region_coordinates, 
        tesseract_config=tesseract_config
    )
    
    # Parallel processing using ThreadPoolExecutor
    print("Starting parallel frame processing...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Map each frame to a worker thread
        results = list(executor.map(process_single_frame, frames_data))
    
    print("Parallel processing complete, organizing results...")
    
    # Step 3: Organize results by region (maintain frame order)
    num_regions = len(region_coordinates)
    region_texts = [[] for _ in range(num_regions)]
    
    for frame_idx, frame_results in results:
        for region_idx, value in enumerate(frame_results):
            if region_idx < num_regions:
                region_texts[region_idx].append(value)
    
    # Step 4: Create DataFrame and save
    headings = [h.strip() for h in headings_inputs.split(",")]
    while len(headings) < num_regions:
        headings.append(f"Region_{len(headings)+1}")
    
    data = {headings[i]: region_texts[i] for i in range(min(len(headings), num_regions))}
    df = pd.DataFrame(data)
    
    # Vectorized data cleaning
    for column in df.columns:
        column_data = df[column]
        non_zero_mask = column_data != 0
        if non_zero_mask.any():
            avg_value = column_data[non_zero_mask].mean()
            df[column] = column_data.where(non_zero_mask, avg_value)
    
    df.to_excel(output_path, index=False)
    create_optimized_plot(df, graph_path)
    
    print(f"Parallel processing complete: {output_path}, {graph_path}")
    return output_path, graph_path

def process_frame_with_regions(frame_data, regions, tesseract_config):
    frame_idx, frame = frame_data
    results = []
    
    for region in regions:
        try:
            x1, y1 = int(region["x"]), int(region["y"])
            w, h = int(region["width"]), int(region["height"])
            x2, y2 = x1 + w, y1 + h
            
            # Extract and preprocess ROI
            roi = frame[y1:y2, x1:x2]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # Optimized thresholding
            _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
            
            # OCR with optimized config
            text = pytesseract.image_to_string(thresh, config=tesseract_config)
            value = extract_number_optimized(text)
            results.append(value)
            
        except Exception as e:
            print(f"Error processing region in frame {frame_idx}: {e}")
            results.append(0)
    
    return frame_idx, results

# Even faster: Multiprocessing version (uses separate processes instead of threads)
def process_video_multiprocess(video_path, region_coordinates, headings_inputs, max_workers=None):
    
    if max_workers is None:
        max_workers = min(mp.cpu_count() - 1, 6)  # Leave one core free
    
    filename = f"VideoData_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_path = os.path.join("results", filename + ".xlsx")
    graph_path = os.path.join("results", filename + ".png")
    
    os.makedirs("results", exist_ok=True)
    
    print("Extracting frames for multiprocessing...")
    frames_data = extract_frames_with_indices(video_path)
    
    if not frames_data:
        print("No frames to process")
        return None, None
    
    print(f"Processing {len(frames_data)} frames using {max_workers} processes")
    
    # Multiprocessing - each process gets its own Python interpreter
    tesseract_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789.'
    
    process_func = partial(
        process_frame_multiprocess_worker,
        regions=region_coordinates,
        tesseract_config=tesseract_config
    )
    
    # Use multiprocessing Pool for CPU-intensive work
    with mp.Pool(processes=max_workers) as pool:
        results = pool.map(process_func, frames_data)
    
    # Organize results
    num_regions = len(region_coordinates)
    region_texts = [[] for _ in range(num_regions)]
    
    # Sort results by frame index to maintain order
    results.sort(key=lambda x: x[0])
    
    for frame_idx, frame_results in results:
        for region_idx, value in enumerate(frame_results):
            if region_idx < num_regions:
                region_texts[region_idx].append(value)
    
    # Create DataFrame and save
    headings = [h.strip() for h in headings_inputs.split(",")]
    while len(headings) < num_regions:
        headings.append(f"Region_{len(headings)+1}")
    
    data = {headings[i]: region_texts[i] for i in range(min(len(headings), num_regions))}
    df = pd.DataFrame(data)
    
    # Vectorized data cleaning
    for column in df.columns:
        column_data = df[column]
        non_zero_mask = column_data != 0
        if non_zero_mask.any():
            avg_value = column_data[non_zero_mask].mean()
            df[column] = column_data.where(non_zero_mask, avg_value)
    
    df.to_excel(output_path, index=False)
    create_optimized_plot(df, graph_path)
    
    print(f"Multiprocessing complete: {output_path}, {graph_path}")
    return output_path, graph_path

def process_frame_multiprocess_worker(frame_data, regions, tesseract_config):
    
    frame_idx, frame = frame_data
    results = []
    
    for region in regions:
        try:
            x1, y1 = int(region["x"]), int(region["y"])
            w, h = int(region["width"]), int(region["height"])
            x2, y2 = x1 + w, y1 + h
            
            roi = frame[y1:y2, x1:x2]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
            
            text = pytesseract.image_to_string(thresh, config=tesseract_config)
            value = extract_number_optimized(text)
            results.append(value)
            
        except Exception as e:
            results.append(0)
    
    return frame_idx, results

def extract_frames_with_indices(video_path):

    capture = cv2.VideoCapture(video_path)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    fps = capture.get(cv2.CAP_PROP_FPS)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames > 1000:
        skip_frames = int(fps)
    else:
        skip_frames = max(1, int(fps // 2))
    
    frames_data = []
    frame_count = 0
    
    while True:
        ret, frame = capture.read()
        if not ret:
            break
            
        if frame_count % skip_frames == 0:
            # Store frame with its index for maintaining order
            frames_data.append((frame_count, frame.copy()))
        
        frame_count += 1
    
    capture.release()
    return frames_data