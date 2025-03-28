import json

input_file = "./vege/data/train/metadata.jsonl"
output_file = "unique_classes.txt"

unique_classes = set()

with open(input_file, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        unique_classes.add(data["class"])


with open(output_file, "w") as f_out:
    for cls in sorted(unique_classes):
        f_out.write(cls + "\n")

print(f"Unique classes have been written to {output_file}")
