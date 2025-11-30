import os
import site

# Add user site packages if not in site packages
import sys
# sys.path usually includes site packages
paths = sys.path

print("Searching in paths:", paths)

found = False
for p_dir in paths:
    if not os.path.exists(p_dir) or not os.path.isdir(p_dir):
        continue
    for root, dirs, files in os.walk(p_dir):
        if "langchain" not in root: # Optimization: only search in langchain related folders
            continue
        for file in files:
            if file.endswith(".py"):
                try:
                    with open(os.path.join(root, file), "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                        if "class ConversationSummaryBufferMemory" in content:
                            print("FOUND IN:", os.path.join(root, file))
                            found = True
                except:
                    pass
if not found:
    print("Not found.")
