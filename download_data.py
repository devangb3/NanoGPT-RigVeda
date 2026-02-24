import os
from pathlib import Path

input_data = ""
work_dir = Path("data/rig_veda")
if work_dir.exists():
    for folder in sorted(work_dir.iterdir()):
        if folder.is_dir():
            for file in sorted(folder.iterdir()):
                if file.name.endswith("eng.txt"):
                    try:
                        input_data += file.read_text() + "\n"
                    except Exception as e:
                        print(f"Error occurred during reading file: {e}")


Path("data/input.txt").write_text(input_data, encoding="utf-8")
print(len(input_data))