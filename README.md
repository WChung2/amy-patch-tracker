# Amy Patch Tracker: GPU-Accelerated Filament Picker

This tool performs high-speed filament detection on Cryo-EM micrographs using GPU acceleration (CuPy). It scans images for specific frequencies (e.g., 4.7Å amyloid cross-beta sheets) and filters micrographs based on patch scores.

## Prerequisites
* **OS:** Linux (Tested on Ubuntu/CentOS)
* **GPU:** NVIDIA GPU (CUDA 11.x or 12.x recommended)
* **Python:** 3.8 or higher

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/YOUR_USERNAME/amy-patch-tracker.git
    cd amy-patch-tracker
    ```

2.  **Create a virtual environment:**
    ```bash
    python3 -m venv venv
    ```

3.  **Activate the environment:**
    ```bash
    source venv/bin/activate
    ```

4.  **Install dependencies:**
    This will install \`cupy\`, \`mrcfile\`, \`matplotlib\`, etc.
    ```bash
    pip install -r requirements.txt
    ```

---

## Usage

### Step 1: Run Filament Picking (\`main.py\`)
This script loads motion-corrected MRC files, analyzes them on the GPU, and saves heatmaps and a STAR file of the results.

**Syntax:**
```bash
python main.py [INPUT_DIR] [OUTPUT_DIR] --workers [NUM_WORKERS] --threshold [THRESHOLD]
```

**Example:**
```bash
python main.py path_to_the_sparc_folder/S1/ 
               path_to_output_folder/ 
               --workers 12 --threshold 0.175
```
* **Input Dir:** Your cryoSPARC output folder after fly motioncorr processing.
* **Output Dir:** Folder where the STAR file and heatmaps will be saved.
* **--workers:** Number of parallel processes. (Recommendation: 2-3 workers per GPU).

### Step 2: Build RELION Data (\`build_relion.py\`)
Once picking is complete, run this script to link the selected micrographs (e.g., EER files) for further processing in RELION/CryoSPARC.

**Syntax:**
```bash
python build_relion.py --input_dir [RAW_DATA_PATH] --star_file [PICKED_STAR_PATH] --output_folder [DESTINATION]
```

**Example:**
```bash
python build_relion.py 
    --input_dir path_to_the_sparc_folder/S1/ 
    --star_file path_to_output_folder/picked_micrographs.star 
    --output_folder path_to_relion_frames/
```

---

## Troubleshooting

* **"CUDA error: initialization error":**
    If the script crashes immediately, try reducing the \`--workers\` count.
* **GPU Memory Warnings:**
    The script automatically recycles workers to prevent memory leaks. If you see "GPU Detected" printing repeatedly, this is normal behavior.
* **Permission Denied:**
    Ensure you have write permissions for the output directory.
