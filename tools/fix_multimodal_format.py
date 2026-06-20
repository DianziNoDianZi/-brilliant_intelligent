"""Fix multimodal data format: JSON array → JSONL."""
import json

p = 'D:/briliant_intelligent/data/multimodal_data.json'
with open(p, 'r', encoding='utf-8') as f:
    raw = f.read()

# Skip past the JSON array by finding matching bracket
depth = 0
array_end = -1
in_str = False
i = 0
while i < len(raw):
    c = raw[i]
    if c == '"' and (i == 0 or raw[i-1] != '\\'):
        in_str = not in_str
    if not in_str:
        if c == '[':
            depth += 1
        elif c == ']':
            depth -= 1
            if depth == 0:
                array_end = i + 1
                break
    i += 1

if array_end <= 0:
    print("No JSON array found")
    exit(1)

array_part = raw[:array_end]
remaining = raw[array_end:].strip()

data = json.loads(array_part)

# Write as JSONL
with open(p, 'w', encoding='utf-8') as f:
    for item in data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')

# Append remaining auto-logged lines
remaining_records = 0
if remaining:
    for line in remaining.split('\n'):
        line = line.strip()
        if line:
            try:
                json.loads(line)
                with open(p, 'a', encoding='utf-8') as f:
                    f.write(line + '\n')
                remaining_records += 1
            except:
                pass

print(f"Converted: {len(data)} array records + {remaining_records} auto-logged lines")
