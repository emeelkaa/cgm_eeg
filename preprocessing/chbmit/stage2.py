import os 
import pickle 
import numpy as np
from tqdm import tqdm

SAMPLING_RATE = 200

INPUT_DIR = "../datasets/chbmit/clean_signals"
OUTPUT_DIR = "../datasets/chbmit/clean_segments"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_PATS = ['chb01', 'chb02', 'chb03', 'chb04',
              'chb05', 'chb06', 'chb07', 'chb08',
              'chb09', 'chb10', 'chb11', 'chb12',
              'chb13', 'chb14', 'chb15', 'chb16',
              'chb17', 'chb18', 'chb19', 'chb20']

VAL_PATS = ["chb21", "chb22"]
TEST_PATS = ["chb23", "chb24"]

CHANNELS = ["FP1-F7", "F7-T7", "T7-P7", "P7-O1",
            "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
            "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
            "FP2-F4", "F4-C4", "C4-P4", "P4-O2"]

def z_normalize(data):
    data = np.nan_to_num(data, nan=0.0) 
    mean = np.mean(data, axis=-1, keepdims=True)
    std = np.std(data, axis=-1, keepdims=True)
    std[std == 0] = 1  # Prevent division by zero
    return (data - mean) / std

def sub_to_segments(folder, out_folder):
    for f in tqdm(os.listdir(os.path.join(INPUT_DIR, folder))):
        tqdm.write(f"Processing {folder}/{f}...")
        with open(os.path.join(INPUT_DIR, folder, f), "rb") as in_file:
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

        for i in range(0, signal.shape[1], SAMPLING_RATE * 10):
            segment = signal[:, i : i + 10 * SAMPLING_RATE]
            if segment.shape[1] == 10 * SAMPLING_RATE:
                label = 0

                for seizure_time in seizure_times:
                    if (
                        i < seizure_time[0] < i + 10 * SAMPLING_RATE
                        or i < seizure_time[1] < i + 10 * SAMPLING_RATE
                    ):
                        label = 1
                        break
                segment = z_normalize(segment)
                with open(os.path.join(out_folder, f"{f.split('.')[0]}-{i}.pkl"), "wb") as out_f:
                    pickle.dump(
                        {"X": segment, "y": label},
                        out_f,
                    )
                
        for idx, seizure_time in enumerate(seizure_times):
            for i in range(
                max(0, seizure_time[0] - SAMPLING_RATE), 
                min(seizure_time[1] + SAMPLING_RATE, signal.shape[1]),
                5 * SAMPLING_RATE,
            ):
                segment = signal[:, i : i + 10 * SAMPLING_RATE]
                label = 1
                segment = z_normalize(segment)
                with open(os.path.join(out_folder, f"{f.split('.')[0]}-s-{idx}-add-{i}.pkl"), "wb") as out_f:
                    pickle.dump(
                        {"X": segment, "y": label},
                        out_f,
                    )

if __name__ == "__main__":
    folders = os.listdir(INPUT_DIR)
    out_folders = []
    for folder in folders:
        if folder in TEST_PATS:
            out_folder = os.path.join(OUTPUT_DIR, "test")
        elif folder in VAL_PATS:
            out_folder = os.path.join(OUTPUT_DIR, "val")
        else:
            out_folder = os.path.join(OUTPUT_DIR, "train")
        
        if not os.path.exists(out_folder):
            os.makedirs(out_folder)

        out_folders.append(out_folder)
    
    for folder, out_folder in zip(folders, out_folders):
        sub_to_segments(folder, out_folder)