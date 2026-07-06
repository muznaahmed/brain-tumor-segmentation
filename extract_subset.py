import zipfile
import os

zip_path = "brats20-dataset-training-validation.zip"
output_dir = "data_subset_45"
num_patients = 45

os.makedirs(output_dir, exist_ok=True)

with zipfile.ZipFile(zip_path, "r") as z:
    names = z.namelist()
    
    # Get unique patient folder prefixes, in order, first 5 only
    patient_prefixes = []
    for name in names:
        parts = name.split("/")
        if len(parts) >= 4 and "BraTS20_Training_" in parts[2]:
            prefix = "/".join(parts[:3]) + "/"
            if prefix not in patient_prefixes:
                patient_prefixes.append(prefix)
        if len(patient_prefixes) >= num_patients:
            break
    
    print(f"Extracting these {len(patient_prefixes)} patients:")
    for p in patient_prefixes:
        print(" ", p)
    
    # Extract only files belonging to those patients
    extracted_count = 0
    for name in names:
        if any(name.startswith(prefix) for prefix in patient_prefixes):
            z.extract(name, output_dir)
            extracted_count += 1
    
    print(f"\nExtracted {extracted_count} files to '{output_dir}/'")