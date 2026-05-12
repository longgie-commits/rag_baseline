import json
import os

meta_path = os.path.join("index", "meta.json")
dump_path = "meta_dump.txt"

with open(meta_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# Sort by page number to create a natural reading flow
data.sort(key=lambda x: x.get("page_num", 0))

output_lines = []
output_lines.append(f"Total chunks: {len(data)}")

# Gather continuous texts per chapter roughly 
last_page = -1

for chunk in data:
    page = chunk.get("page_num", 0)
    text = chunk.get("text", "").strip()
    if not text:
        continue
    
    if page != last_page:
        output_lines.append(f"\n\n================ PAGE {page} ================\n")
        last_page = page
    
    # We truncate incredibly long repeating formulas to save token length
    # This dump is just for human reading context.
    output_lines.append(text)

with open(dump_path, "w", encoding="utf-8") as f:
    f.write("\n".join(output_lines))

print("Dump successful!")
