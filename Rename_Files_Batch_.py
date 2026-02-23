# Renames files in Folder
import os
def rename_files_in_folder(file_paths, old_string, new_string):
# List all files in the folder
    for file_path in file_paths:
        #print(file_path)
        for filename in os.listdir(file_path):
            # Check if the file contains the old_string
            if old_string in filename:
                # Generate the new filename by replacing old_string with new_string
                new_filename = filename.replace(old_string, new_string)
            
                # Build full file paths
                old_file_path = os.path.join(file_path, filename)
                new_file_path = os.path.join(file_path, new_filename)
            
                # Rename the file
                try:
                    os.rename(old_file_path, new_file_path)
                    print(f"Renamed: {filename} -> {new_filename}")
                except Exception as e:
                    print(f"Failed to rename {filename}: {e}")

# Define the folder path and the strings to replace
file_paths = [
    'E:/',

]

old_string = '2'
new_string = '24'

# Call the function
rename_files_in_folder(file_paths, old_string, new_string)

# Add prefix to files
import os
def add_prefix_to_files(file_paths, prefix):
    for file_path in file_paths:
        for filename in os.listdir(file_path):
            # Generate the new filename by adding the prefix
            new_filename = f"{prefix}{filename}"
            
            # Build full file paths
            old_file_path = os.path.join(file_path, filename)
            new_file_path = os.path.join(file_path, new_filename)
            
            # Rename the file
            try:
                os.rename(old_file_path, new_file_path)
                print(f"Renamed: {filename} -> {new_filename}")
            except Exception as e:
                print(f"Failed to rename {filename}: {e}")

# Usage
folders_to_process = [
    'E:/',

]
prefix_to_add = '_'

add_prefix_to_files(folders_to_process, prefix_to_add)

