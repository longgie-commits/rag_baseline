import codecs
import re

with codecs.open('build_bench.py', 'r', 'utf-8-sig') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if '"id": "14"' in line or '"id": "15"' in line:
        line = re.sub(r'"type":\s*"[^"]+"', '"type": "figure_table_dependent"', line)
    else:
        line = line.replace('"type": "definition"', '"type": "text_only"')
        line = line.replace('"type": "fact"', '"type": "text_only"')
        line = line.replace('"type": "comparison"', '"type": "cross_segment_reasoning"')
        line = line.replace('"type": "reasoning"', '"type": "cross_segment_reasoning"')
        line = line.replace('"type": "application"', '"type": "cross_segment_reasoning"')
        line = line.replace('"type": "formula_extraction"', '"type": "formula_dependent"')
        line = line.replace('"type": "symbol_reasoning"', '"type": "formula_dependent"')
    new_lines.append(line)

with codecs.open('build_bench.py', 'w', 'utf-8') as f:
    f.writelines(new_lines)
