"""Surgically remove the batch tab remnant lines 195-214 from app.jsx."""

path = r"c:\Users\yaswi\OneDrive\Desktop\Finalc17 (2)\mani-c17\frontend\src\app.jsx"

with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Lines 195-214 (0-indexed: 194-213) are the batch JSX remnants
# Line 194 (index 194) starts with `<div style={{ minHeight...` mixed with batchText junk
# Line 215 (index 214) is blank, then 216-220 are proper closing JSX

# Keep:  lines 0-193 (behavioralInputs array + return start)
# Skip:  lines 194-213 (corrupted batch tab JSX)  
# Keep:  lines 214-219 (the proper closing: </main></div></></>)

# But lines 216-220 have the real closing, and line 194 is the REAL
# <div style={{ minHeight...}} corrupted with batch code appended.
# We need to replace line 194 with the clean version.

clean_closing = [
    '      <div style={{ minHeight:"100vh", position:"relative", zIndex:1 }}>\n',
    '\n',
    '        </main>\n',
    '      </div>\n',
    '    </>\n',
    '  );\n',
    '}\n',
]

# Lines 0..193 are good, then add clean closing
new_lines = lines[:194] + clean_closing

with open(path, 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print(f"Done. Total lines now: {len(new_lines)}")
print("Last 10 lines:")
for l in new_lines[-10:]:
    print(repr(l))
