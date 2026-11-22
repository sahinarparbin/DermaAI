
import zipfile
import json

model_path = "models/stage1_final_best.keras"

with zipfile.ZipFile(model_path, "r") as model_zip:

    for filename in ["config.json", "metadata.json"]:
        print("\n" + "=" * 50)
        print(filename)
        print("=" * 50)

        content = model_zip.read(filename).decode("utf-8")

        # Display the contents
        print(content)

        # Save a separate copy
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)

print("\nExtraction completed!")
