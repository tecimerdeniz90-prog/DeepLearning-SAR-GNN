import json

with open('Deep Neural Network - Application.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

with open('nb_output.txt', 'w', encoding='utf-8') as out:
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = "".join(cell['source'])
            if 'YOUR CODE' in source:
                out.write(f"--- Cell {i} ---\n")
                out.write(source)
                out.write("\n\n")
