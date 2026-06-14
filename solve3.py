import json

with open('Planar_data_classification_with_one_hidden_layer.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

replacements = {
    "shape_X = ...": [
        "    # YOUR CODE STARTS HERE\n",
        "    shape_X = X.shape\n",
        "    shape_Y = Y.shape\n",
        "    m = X.shape[1]\n",
        "    # YOUR CODE ENDS HERE\n"
    ],
    "def layer_sizes(X, Y):": [
        "    # YOUR CODE STARTS HERE\n",
        "    n_x = X.shape[0]\n",
        "    n_h = 4\n",
        "    n_y = Y.shape[0]\n",
        "    # YOUR CODE ENDS HERE\n"
    ],
    "def initialize_parameters(n_x, n_h, n_y):": [
        "    # YOUR CODE STARTS HERE\n",
        "    W1 = np.random.randn(n_h, n_x) * 0.01\n",
        "    b1 = np.zeros((n_h, 1))\n",
        "    W2 = np.random.randn(n_y, n_h) * 0.01\n",
        "    b2 = np.zeros((n_y, 1))\n",
        "    # YOUR CODE ENDS HERE\n"
    ],
    "def forward_propagation(X, parameters):": [
        "    # YOUR CODE STARTS HERE\n",
        "    W1 = parameters['W1']\n",
        "    b1 = parameters['b1']\n",
        "    W2 = parameters['W2']\n",
        "    b2 = parameters['b2']\n",
        "    # YOUR CODE ENDS HERE\n",
        "    # YOUR CODE STARTS HERE\n",
        "    Z1 = np.dot(W1, X) + b1\n",
        "    A1 = np.tanh(Z1)\n",
        "    Z2 = np.dot(W2, A1) + b2\n",
        "    A2 = sigmoid(Z2)\n",
        "    # YOUR CODE ENDS HERE\n"
    ],
    "def compute_cost(A2, Y):": [
        "    # YOUR CODE STARTS HERE\n",
        "    logprobs = np.multiply(np.log(A2), Y) + np.multiply(np.log(1 - A2), 1 - Y)\n",
        "    cost = -1/m * np.sum(logprobs)\n",
        "    # YOUR CODE ENDS HERE\n"
    ],
    "def backward_propagation(parameters, cache, X, Y):": [
        "    # YOUR CODE STARTS HERE\n",
        "    W1 = parameters['W1']\n",
        "    W2 = parameters['W2']\n",
        "    # YOUR CODE ENDS HERE\n",
        "    # YOUR CODE STARTS HERE\n",
        "    A1 = cache['A1']\n",
        "    A2 = cache['A2']\n",
        "    # YOUR CODE ENDS HERE\n",
        "    # YOUR CODE STARTS HERE\n",
        "    dZ2 = A2 - Y\n",
        "    dW2 = 1/m * np.dot(dZ2, A1.T)\n",
        "    db2 = 1/m * np.sum(dZ2, axis=1, keepdims=True)\n",
        "    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))\n",
        "    dW1 = 1/m * np.dot(dZ1, X.T)\n",
        "    db1 = 1/m * np.sum(dZ1, axis=1, keepdims=True)\n",
        "    # YOUR CODE ENDS HERE\n"
    ],
    "def update_parameters(parameters, grads, learning_rate": [
        "    # YOUR CODE STARTS HERE\n",
        "    W1 = copy.deepcopy(parameters['W1'])\n",
        "    b1 = parameters['b1']\n",
        "    W2 = copy.deepcopy(parameters['W2'])\n",
        "    b2 = parameters['b2']\n",
        "    # YOUR CODE ENDS HERE\n",
        "    # YOUR CODE STARTS HERE\n",
        "    dW1 = grads['dW1']\n",
        "    db1 = grads['db1']\n",
        "    dW2 = grads['dW2']\n",
        "    db2 = grads['db2']\n",
        "    # YOUR CODE ENDS HERE\n",
        "    # YOUR CODE STARTS HERE\n",
        "    W1 = W1 - learning_rate * dW1\n",
        "    b1 = b1 - learning_rate * db1\n",
        "    W2 = W2 - learning_rate * dW2\n",
        "    b2 = b2 - learning_rate * db2\n",
        "    # YOUR CODE ENDS HERE\n"
    ],
    "def nn_model(X, Y, n_h, num_iterations = 10000": [
        "    # YOUR CODE STARTS HERE\n",
        "    parameters = initialize_parameters(n_x, n_h, n_y)\n",
        "    # YOUR CODE ENDS HERE\n",
        "        # YOUR CODE STARTS HERE\n",
        "        A2, cache = forward_propagation(X, parameters)\n",
        "        cost = compute_cost(A2, Y)\n",
        "        grads = backward_propagation(parameters, cache, X, Y)\n",
        "        parameters = update_parameters(parameters, grads)\n",
        "        # YOUR CODE ENDS HERE\n"
    ],
    "def predict(parameters, X):": [
        "    # YOUR CODE STARTS HERE\n",
        "    A2, cache = forward_propagation(X, parameters)\n",
        "    predictions = (A2 > 0.5)\n",
        "    # YOUR CODE ENDS HERE\n"
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

with open('Planar_data_classification_with_one_hidden_layer.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    
print("3. Notebook başarıyla çözüldü!")
