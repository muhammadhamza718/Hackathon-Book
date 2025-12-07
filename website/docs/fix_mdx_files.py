import os
import re
def fix_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return
    # Split by code blocks to avoid touching code
    # This regex splits by triple backticks
    parts = re.split(r'(```[\s\S]*?```)', content)
    
    new_parts = []
    modified = False
    
    for part in parts:
        if part.startswith('```') and part.endswith('```'):
            new_parts.append(part)
        else:
            # This is prose (or inline code)
            # We need to be careful not to break inline code `...`
            # For simplicity, we'll iterate through the text and escape {token}
            # avoiding inline code.
            
            # Helper to replace in text while skipping inline code
            def replace_in_prose(text):
                # Split by inline code
                segments = re.split(r'(`[^`]*`)', text)
                fixed_segments = []
                local_mod = False
                for seg in segments:
                    if seg.startswith('`') and seg.endswith('`'):
                        fixed_segments.append(seg)
                    else:
                        # Regex to find {token} where token starts with a letter
                        # We use lookbehind to ensure it's not already escaped
                        # Pattern: not backslash, {, letter/underscore, word chars
                        def repl(m):
                            nonlocal local_mod
                            local_mod = True
                            return '\\' + m.group(0)
                        
                        # Python re lookbehind requires fixed width, so handle manually
                        # search for {var
                        new_seg = re.sub(r'(?<!\\)\{([a-zA-Z_]\w*)', repl, seg)
                        fixed_segments.append(new_seg)
                
                return "".join(fixed_segments), local_mod
            new_part, part_mod = replace_in_prose(part)
            if part_mod:
                modified = True
            new_parts.append(new_part)
    if modified:
        new_content = "".join(new_parts)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Fixed: {filepath}")
def main():
    root_dir = r"F:\Courses\Hamza\Hackathon-2\hackathon-book\website\docs"
    print(f"Scanning {root_dir}...")
    
    count = 0
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".md") or file.endswith(".mdx"):
                filepath = os.path.join(subdir, file)
                fix_file(filepath)
                count += 1
    
    print(f"Scanned {count} files.")
if __name__ == "__main__":
    main()