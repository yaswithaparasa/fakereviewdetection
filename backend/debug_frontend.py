"""Remove the batch tab block and fix any remaining batchText/batchRes references from app.jsx."""

path = r"c:\Users\yaswi\OneDrive\Desktop\Finalc17 (2)\mani-c17\frontend\src\app.jsx"

with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

print(f"Total lines before: {len(lines)}")

# Print lines 185-220 to understand structure
for i, l in enumerate(lines[184:], 185):
    print(f"{i}: {l[:120]!r}")
