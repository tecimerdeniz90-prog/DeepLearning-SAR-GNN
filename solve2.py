import json

with open('Logistic_Regression_with_a_Neural_Network_mindset.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

replacements = {
    "m_train = ": [
        "    # YOUR CODE STARTS HERE\n",
        "    m_train = train_set_x_orig.shape[0]\n",
        "    m_test = test_set_x_orig.shape[0]\n",
        "    num_px = train_set_x_orig.shape[1]\n",
        "    # YOUR CODE ENDS HERE\n"
    ],
    "train_set_x_flatten = ...": [
        "    # YOUR CODE STARTS HERE\n",
        "    train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T\n",
        "    test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T\n",
        "    # YOUR CODE ENDS HERE\n"
    ],
    "def sigmoid(z):": [
        "    # YOUR CODE STARTS HERE\n",
        "    s = 1 / (1 + np.exp(-z))\n",
        "    # YOUR CODE ENDS HERE\n"
    ],
    "def initialize_with_zeros(dim):": [
        "    # YOUR CODE STARTS HERE\n",
        "    w = np.zeros((dim, 1))\n",
        "    b = 0.0\n",
        "    # YOUR CODE ENDS HERE\n"
    ],
    "def propagate(w, b, X, Y):": [
        "    # YOUR CODE STARTS HERE\n",
        "    A = sigmoid(np.dot(w.T, X) + b)\n",
        "    cost = -1/m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))\n",
        "    # YOUR CODE ENDS HERE\n",
        "    # YOUR CODE STARTS HERE\n",
        "    dw = 1/m * np.dot(X, (A - Y).T)\n",
        "    db = 1/m * np.sum(A - Y)\n",
        "    # YOUR CODE ENDS HERE\n"
    ],
    "def optimize": [
        "        # YOUR CODE STARTS HERE\n",
        "        grads, cost = propagate(w, b, X, Y)\n",
        "        # YOUR CODE ENDS HERE\n",
        "        # YOUR CODE STARTS HERE\n",
        "        w = w - learning_rate * grads['dw']\n",
        "        b = b - learning_rate * grads['db']\n",
        "        # YOUR CODE ENDS HERE\n"
    ],
    "def predict(w, b, X):": [
        "    # YOUR CODE STARTS HERE\n",
        "    A = sigmoid(np.dot(w.T, X) + b)\n",
        "    # YOUR CODE ENDS HERE\n",
        "        # YOUR CODE STARTS HERE\n",
        "        if A[0, i] > 0.5:\n",
        "            Y_prediction[0, i] = 1\n",
        "        else:\n",
        "            Y_prediction[0, i] = 0\n",
        "        # YOUR CODE ENDS HERE\n"
    ],
    "def model(X_train, Y_train, X_test, Y_test": [
        "    # YOUR CODE STARTS HERE\n",
        "    w, b = initialize_with_zeros(X_train.shape[0])\n",
        "    params, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)\n",
        "    w = params['w']\n",
        "    b = params['b']\n",
        "    Y_prediction_test = predict(w, b, X_test)\n",
        "    Y_prediction_train = predict(w, b, X_train)\n",
        "    # YOUR CODE ENDS HERE\n"
    ]
}

def apply_replacements(cell_source, key):
    # This function handles multiple start/end blocks within a single cell
    # First we figure out how many blocks we have in our replacement
    # Our replacements above have multiple blocks for propagate, optimize, predict.
    # We will just replace the i-th block in source with i-th block in replacement
    
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
            
    # Now find all start/end in source
    out_source = []
    i = 0
    block_idx = 0
    while i < len(cell_source):
        line = cell_source[i]
        if 'YOUR CODE STARTS HERE' in line:
            # We skip lines until ENDS HERE
            while i < len(cell_source) and 'YOUR CODE ENDS HERE' not in cell_source[i]:
                i += 1
            # append the replacement block
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

with open('Logistic_Regression_with_a_Neural_Network_mindset.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    
print("2. Notebook başarıyla çözüldü!")
