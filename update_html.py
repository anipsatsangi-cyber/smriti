with open('frontend/index.html', 'r') as f:
    content = f.read()

# Replace why-grid with slider-grid
content = content.replace('class="why-grid"', 'class="slider-grid"')
# Replace why-card with slider-card
content = content.replace('class="why-card"', 'class="slider-card"')

# Wrap bench tables in slider-card bench-card inside a slider-grid
# First, let's find the container of bench tables
bench_start_marker = "<!-- Smriti's own three-layer benchmark, no fictional competitors -->"
bench_end_marker = "  <!-- What Smriti gives that vector stores structurally cannot -->"

parts = content.split(bench_start_marker)
if len(parts) == 2:
    pre = parts[0]
    rest = parts[1]
    
    # We'll just manually replace the bench-table divs with wrapped versions
    rest = rest.replace('<div class="bench-table">', '<div class="slider-card bench-card">\n    <div class="bench-table">')
    rest = rest.replace('<div class="bench-table" style="margin-top: 3rem;">', '<div class="slider-card bench-card">\n    <div class="bench-table">')
    
    # We need to close the slider-card after the closing div of bench-table
    # Each bench-table ends with </div>.
    # Let's just use regex or a simple script
    import re
    rest = re.sub(r'(</p>\n  </div>)', r'\1\n  </div>', rest)
    
    content = pre + bench_start_marker + '\n  <div class="slider-grid" style="padding-top: 1rem;">\n  ' + rest

# One more fix: The last bench table needs closing. Let's just do it cleanly.

with open('frontend/index.html', 'w') as f:
    f.write(content)
