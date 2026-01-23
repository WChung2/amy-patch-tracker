import os
import argparse
import subprocess
import sys

def run_rsync_transfer(input_dir, star_file_path, output_folder):
    # 1. Setup paths
    input_movie_folder = os.path.join(input_dir, 'import_movies')
    
    # Define a temporary file to store the list of filenames
    temp_list_file = "files_to_copy_temp.txt"

    if not os.path.isdir(input_movie_folder):
        print(f"Error: Could not find 'import_movies' folder inside: {input_dir}")
        sys.exit(1)

    if not os.path.exists(output_folder):
        print(f"Creating output directory: {output_folder}")
        os.makedirs(output_folder)

    # 2. Parse the star file and generate the list
    print("Parsing star file...")
    valid_files = []
    
    try:
        with open(star_file_path, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            stripped = line.strip()
            # Skip headers/comments
            if not stripped or stripped.startswith(('data_', 'loop_', '_', '#')):
                continue

            columns = stripped.split()
            if not columns: continue
            mrc_filename = columns[0]

            # Convert to .eer
            if '_patch_aligned.mrc' in mrc_filename:
                eer_filename = mrc_filename.replace('_patch_aligned.mrc', '.eer')
            elif mrc_filename.endswith('.mrc'):
                eer_filename = mrc_filename.replace('.mrc', '.eer')
            else:
                continue

            valid_files.append(eer_filename)
            
    except FileNotFoundError:
        print(f"Error: Could not find star file: {star_file_path}")
        sys.exit(1)

    if not valid_files:
        print("No valid files found in star file.")
        sys.exit(1)

    # 3. Write the filenames to a temporary text file
    print(f"Found {len(valid_files)} files. Preparing rsync...")
    with open(temp_list_file, 'w') as f:
        for filename in valid_files:
            f.write(filename + '\n')

    # 4. Construct the rsync command
    # -a: archive mode (preserves timestamps/permissions)
    # -v: verbose
    # --progress: show progress bar
    # --files-from: read the list of files we just created
    rsync_cmd = [
        "rsync", 
        "-av", 
        "--progress", 
        f"--files-from={temp_list_file}", 
        input_movie_folder,   # Source directory
        output_folder         # Destination directory
    ]

    print("-" * 40)
    print(f"Starting Transfer of {len(valid_files)} files...")
    print("Command:", " ".join(rsync_cmd))
    print("-" * 40)

    # 5. Execute rsync
    try:
        subprocess.run(rsync_cmd, check=True)
        print("-" * 40)
        print("Success! All files synced.")
    except subprocess.CalledProcessError as e:
        print(f"Error: Rsync failed with code {e.returncode}")
    finally:
        # Cleanup: Remove the temporary list file
        if os.path.exists(temp_list_file):
            os.remove(temp_list_file)
            print("Cleaned up temporary file list.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy EER files using rsync based on a star file.")
    
    parser.add_argument('--input_dir', required=True, type=str, help="Directory of your cryoSPARC S folder (e.g. .../S1/)")
    parser.add_argument('--star_file', required=True, type=str, help="The star file from amy_tracker")
    parser.add_argument('--output_folder', required=True, type=str, help="Destination directory")

    args = parser.parse_args()

    run_rsync_transfer(args.input_dir, args.star_file, args.output_folder)