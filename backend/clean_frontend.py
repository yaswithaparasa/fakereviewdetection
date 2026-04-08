"""Clean frontend: remove batch tab, URL analyzer, and related state/functions."""
import re

path = r"c:\Users\yaswi\OneDrive\Desktop\Finalc17 (2)\mani-c17\frontend\src\app.jsx"

with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

content = ''.join(lines)

# ── 1. Remove state variable lines for batch/url ──────────────────────────────
remove_vars = [
    "batchText", "setBatchText",
    "batchRes",  "setBatchRes",
    "urlInput",  "setUrlInput",
    "urlLoading","setUrlLoading",
    "urlRes",    "setUrlRes",
    "urlError",  "setUrlError",
    "urlMode",   "setUrlMode",
    "htmlInput", "setHtmlInput",
    "reviewsText","setReviewsText",
]

new_lines = []
for line in lines:
    # Drop any useState line that declares one of the removed variables
    if "useState" in line and any(v in line for v in remove_vars):
        continue
    new_lines.append(line)

content = ''.join(new_lines)

# ── 2. Remove isUrl helper ────────────────────────────────────────────────────
content = re.sub(r'\n\s*const isUrl = \(text\).*?;', '', content)

# ── 3. Remove runBatch function (useCallback) ─────────────────────────────────
# Find and remove the runBatch useCallback block
content = re.sub(
    r'\n  const runBatch = useCallback\(async \(\) => \{.*?\}, \[batchText.*?\]\);',
    '', content, flags=re.DOTALL
)

# ── 4. Remove runUrlScan function (useCallback) ───────────────────────────────
content = re.sub(
    r'\n  const runUrlScan = useCallback\(async \(\) => \{.*?\}, \[urlInput.*?\]\);',
    '', content, flags=re.DOTALL
)

# ── 5. Remove the tabs UI (keep only single tab always visible) ───────────────
# Replace the two-tab pill nav with nothing (remove the whole div.flex tabs block)
content = re.sub(
    r'\s*\{/\* Tabs \*/\}.*?\{/\* .* SINGLE TAB .*?\*\/\}',
    '\n          {/* Single Review Analyser */}',
    content, flags=re.DOTALL
)

# ── 6. Remove the batch tab conditional block entirely ────────────────────────
content = re.sub(
    r'\n\s*\{/\* .* BATCH TAB .*?\*\/\}\s*\{tab===.batch. &&.*?\}\s*\)',
    '',
    content, flags=re.DOTALL
)

# ── 7. Remove UrlReviewRow and BatchRow component usage refs if any stray remaining
content = re.sub(r'\n/\* .* Batch row .*?\*/.*?^}', '', content, flags=re.DOTALL | re.MULTILINE)

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Done! Remaining mentions check:")
print("  batchText:", "batchText" in content)
print("  runBatch:", "runBatch" in content)
print("  urlInput:", "urlInput" in content)
print("  runUrlScan:", "runUrlScan" in content)
print("  tab===.batch:", 'tab==="batch"' in content)
