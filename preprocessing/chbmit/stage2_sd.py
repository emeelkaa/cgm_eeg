import os 
import pickle 
import numpy as np 
from tqdm import tqdm

BASE_DIR = "../datasets/chbmit/clean_signals"
OUT_DIR = "../datasets/chbmit/clean_segments_2"
os.makedirs(OUT_DIR, exist_ok=True)

SAMPLING_RATE = 256

CHANNELS = [
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
    "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2"
]

def z_normalize(data):
    data = np.nan_to_num(data, nan=0.0) 
    mean = np.mean(data, axis=-1, keepdims=True)
    std = np.std(data, axis=-1, keepdims=True)
    std[std == 0] = 1  # Prevent division by zero
    return (data - mean) / std

def sample_bckg(data, seizure_times, num_samples, segment_length):
    signal_length = data.shape[1]
    valid_mask = np.ones(signal_length, dtype=bool)
    for start, end in seizure_times:
        valid_mask[start:end] = False
    
    valid_starts = []
    for i in range(0, signal_length - segment_length + 1, segment_length):
        if np.all(valid_mask[i : i + segment_length]):
            valid_starts.append(i)
    
    if len(valid_starts) == 0:
        return []
    
    num_to_sample = min(num_samples, len(valid_starts))
    sampled_indices = np.random.choice(valid_starts, num_to_sample, replace=False)

    segments = []
    for idx in sampled_indices:
        segment = data[:, idx: idx + segment_length]
        if segment.shape[1] == segment_length:
            segments.append((segment, idx))
    
    return segments


if __name__ == "__main__":
    out_dir_seizures = os.path.join(OUT_DIR, "full_dataset")
    os.makedirs(out_dir_seizures, exist_ok=True)

    folders = os.listdir(BASE_DIR)
    for folder in folders:
        print(f"Processing {folder}...")
        folder_dir = os.path.join(BASE_DIR, folder)
        for f in tqdm(os.listdir(os.path.join(BASE_DIR, folder)), desc=f"Processing {folder}", unit="file"):
            with open(os.path.join(folder_dir, f), "rb") as in_file:
                record = pickle.load(in_file)
            signal = []
            for channel in CHANNELS:
                if channel in record:
                    signal.append(record[channel])
                else:
                    raise ValueError(f"Channel {channel} not found in record {record}")
            signal = np.array(signal)
            if "times" in record["metadata"]:
                seizure_times = record["metadata"]["times"]
            else:
                seizure_times = []
                
            if seizure_times == []:
                continue
            
            num_seizure_segments = 0

            for start, end in seizure_times:
                for i in range(start, end, SAMPLING_RATE * 5):
                    segment = signal[:, i : i + 10 * SAMPLING_RATE]
                    if segment.shape[1] == 10 * SAMPLING_RATE:
                        segment = z_normalize(segment)
                        with open(os.path.join(out_dir_seizures, f"{f.split('.')[0]}-s-{i}.pkl"), "wb") as out_f:
                            pickle.dump(
                                {"X": segment, "y": 1},
                                out_f,
                            )
                        num_seizure_segments += 1
            
            bckg_segments = sample_bckg(signal, seizure_times, num_seizure_segments * 3, 10 * SAMPLING_RATE)
            for bckg_segment in bckg_segments:
                segment, idx = bckg_segment
                segment = z_normalize(segment)
                with open(os.path.join(out_dir_seizures, f"{f.split('.')[0]}-b-{idx}.pkl"), "wb") as out_f:
                    pickle.dump(
                        {"X": segment, "y": 0},
                        out_f,
                    )
            
