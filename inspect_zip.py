import zipfile

zip_path = "brats20-dataset-training-validation.zip"

with zipfile.ZipFile(zip_path, "r") as z:
    names = z.namelist()
    print(f"Total files in zip: {len(names)}")
    print("\nFirst 15 entries:")
    for name in names[:15]:
        print(name)