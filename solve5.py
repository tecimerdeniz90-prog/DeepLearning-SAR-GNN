import json

with open('Deep Neural Network - Application.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

replacements = {
    "def two_layer_model(X, Y, layers_dims": [
        "    # YOUR CODE STARTS HERE\n",
        "    parameters = initialize_parameters(n_x, n_h, n_y)\n",
        "    # YOUR CODE ENDS HERE\n",
        "        # YOUR CODE STARTS HERE\n",
        "        A1, cache1 = linear_activation_forward(X, W1, b1, activation='relu')\n",
        "        A2, cache2 = linear_activation_forward(A1, W2, b2, activation='sigmoid')\n",
        "        # YOUR CODE ENDS HERE\n",
        "        # YOUR CODE STARTS HERE\n",
        "        cost = compute_cost(A2, Y)\n",
        "        # YOUR CODE ENDS HERE\n",
        "        # YOUR CODE STARTS HERE\n",
        "        dA1, dW2, db2 = linear_activation_backward(dA2, cache2, activation='sigmoid')\n",
        "        dA0, dW1, db1 = linear_activation_backward(dA1, cache1, activation='relu')\n",
        "        # YOUR CODE ENDS HERE\n",
        "        # YOUR CODE STARTS HERE\n",
        "        parameters = update_parameters(parameters, grads, learning_rate)\n",
        "        # YOUR CODE ENDS HERE\n"
    ],
    "def L_layer_model(X, Y, layers_dims": [
        "    # YOUR CODE STARTS HERE\n",
        "    parameters = initialize_parameters_deep(layers_dims)\n",
        "    # YOUR CODE ENDS HERE\n",
        "        # YOUR CODE STARTS HERE\n",
        "        AL, caches = L_model_forward(X, parameters)\n",
        "        # YOUR CODE ENDS HERE\n",
        "        # YOUR CODE STARTS HERE\n",
        "        cost = compute_cost(AL, Y)\n",
        "        # YOUR CODE ENDS HERE\n",
        "        # YOUR CODE STARTS HERE\n",
        "        grads = L_model_backward(AL, Y, caches)\n",
        "        # YOUR CODE ENDS HERE\n",
        "        # YOUR CODE STARTS HERE\n",
        "        parameters = update_parameters(parameters, grads, learning_rate)\n",
        "        # YOUR CODE ENDS HERE\n"
    ]
}

def apply_replacements(cell_source, key):
    rep_blocks = []
    current_block = []
    in_rep = False
    for line in replacements[key]:
        if 'YOUR CODE STARTS HERE' in line:
            in_rep = True
            current_block = [line]
        elif 'YOUR CODE ENDS HERE' in line:
            current_block.append(line)
            rep_blocks.append(current_block)
            in_rep = False
        elif in_rep:
            current_block.append(line)
            
    out_source = []
    i = 0
    block_idx = 0
    while i < len(cell_source):
        line = cell_source[i]
        if 'YOUR CODE STARTS HERE' in line:
            while i < len(cell_source) and 'YOUR CODE ENDS HERE' not in cell_source[i]:
                i += 1
            if block_idx < len(rep_blocks):
                out_source.extend(rep_blocks[block_idx])
                block_idx += 1
        else:
            out_source.append(line)
        i += 1
    return out_source

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if 'YOUR CODE' in source:
            for key in replacements:
                if key in source:
                    cell['source'] = apply_replacements(cell['source'], key)
                    break

with open('Deep Neural Network - Application.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
