# Code adapted from the BIOT repository:
# https://github.com/ycq091044/BIOT

import os 
import pickle 
import numpy as np 
import mne 
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

SAMPLING_RATE = 200
NOTCH_FILTER = 50 
HIGHPASS_FILTER = 0.5 
LOWPASS_FILTER = 75

INPUT_DIR = "../datasets/chbmit/edf"
OUTPUT_DIR = "../datasets/chbmit/clean_signals"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def preprocess(raw, sfreq, notch_filter, highpass_filter, lowpass_filter):
    if raw.info["sfreq"] != sfreq:
        raw.resample(sfreq, npad='auto', verbose=False)
    raw.notch_filter(freqs=notch_filter, verbose=False)
    raw.filter(highpass_filter, lowpass_filter, fir_design='firwin', verbose=False)
    return raw

# process metadata (extract seizure info and timing)
def process_metadata(summary, filename):
    with open(summary, "r") as f:
        lines = f.readlines()

    metadata = {}
    times = []
    for i in range(len(lines)):
        line = lines[i].split()
        if len(line) == 3 and line[2] == filename:
            j = i + 1
            processed = False
            # look until we find the number of seizures in this file
            while not processed:
                if lines[j].split()[0] == "Number":
                    seizures = int(lines[j].split()[-1])
                    processed = True
                j = j + 1

            # if file has seizures get their start and end time
            if seizures > 0:
                j = i + 1
                for s in range(seizures):
                    processed = False
                    while not processed:
                        l = lines[j].split()
                        if l[0] == "Seizure" and "Start" in l:
                            start = int(l[-2]) * SAMPLING_RATE - 1  # seizure start (in samples)
                            end = (
                                int(lines[j + 1].split()[-2]) * SAMPLING_RATE - 1
                            )  # seizure end (in samples)
                            processed = True
                        j = j + 1
                    times.append((start, end))

            metadata["seizures"] = seizures  # total number of seizures
            metadata["times"] = times        # list of (start, end) time pairs
 
    return metadata 

def process_file(edf_path):
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False, exclude=['-', '.'])
    if 'T8-P8-0' in raw.ch_names and 'T8-P8-1' in raw.ch_names:
        raw.drop_channels(['T8-P8-1'])
        raw.rename_channels(mapping={'T8-P8-0': 'T8-P8'})
    
    raw = preprocess(raw, SAMPLING_RATE, NOTCH_FILTER, HIGHPASS_FILTER, LOWPASS_FILTER)
    channels = raw.ch_names
    signals = np.array(raw.get_data())

    clean_dict = {channel: signal for channel, signal in zip(channels, signals)}

    return channels, clean_dict

def start_process(patient, ref_num):
    os.makedirs(f"{OUTPUT_DIR}/chb{patient}", exist_ok=True)

    ref_file = f"{INPUT_DIR}/chb{patient}/chb{patient}_{ref_num}.edf"
    ref_channels, ref_clean_dict = process_file(ref_file)

    ref_metadata = process_metadata(
        f"{INPUT_DIR}/chb{patient}/chb{patient}-summary.txt",
        f"chb{patient}_{ref_num}.edf",
    )
    
    ref_metadata['channels'] = ref_channels
    ref_clean_dict['metadata'] = ref_metadata
    print(f"\nProcessed reference file for {patient}: {ref_channels}")
    target = f"{OUTPUT_DIR}/chb{patient}/chb{patient}_{ref_num}.pkl"
    with open(target, "wb") as out_f:
        pickle.dump(ref_clean_dict, out_f)
    
    remaining_files = [f for f in os.listdir(f"{INPUT_DIR}/chb{patient}") if f.endswith(".edf") and f != f"chb{patient}_{ref_num}.edf"]
    for file in remaining_files:
        channels, clean_dict = process_file(f"{INPUT_DIR}/chb{patient}/{file}")
        num = file.split('.')[0].split('_')[-1]
        metadata = process_metadata(
            f"{INPUT_DIR}/chb{patient}/chb{patient}-summary.txt",
            f"chb{patient}_{num}.edf",
        )
        if metadata == {}:
            print(f"No metadata found for {file}, skipping")
            continue
        
        metadata['channels'] = channels
        clean_dict['metadata'] = metadata
        print(f"Saving chb{patient}_{num} with {metadata['seizures']} seizures")
        target = f"{OUTPUT_DIR}/chb{patient}/chb{patient}_{num}.pkl"
        with open(target, "wb") as out_f:
            pickle.dump(clean_dict, out_f)

if __name__ == "__main__":
    parameters = [
        ("01", "01"),
        ("02", "01"),
        ("03", "01"),
        ("05", "01"),
        ("06", "01"),
        ("07", "01"),
        ("08", "02"),
        ("10", "01"),
        ("11", "01"),
        ("14", "01"),
        ("20", "01"),
        ("21", "01"),
        ("22", "01"),
        ("23", "06"),
        ("24", "01"),
        ("04", "07"),
        ("09", "02"),
        ("15", "02"),
        ("16", "01"),
        ("18", "02"),
        ("19", "02"),
    ]

    for patient, ref_num in tqdm(parameters, desc="Processing patients", unit="patient"):
        start_process(patient, ref_num)


