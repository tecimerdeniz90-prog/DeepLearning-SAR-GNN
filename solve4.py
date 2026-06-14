import json

with open('Building_your_Deep_Neural_Network_Step_by_Step.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

replacements = {
    "def initialize_parameters(n_x, n_h, n_y):": [
        "    # YOUR CODE STARTS HERE\n",
        "    W1 = np.random.randn(n_h, n_x) * 0.01\n",
        "    b1 = np.zeros((n_h, 1))\n",
        "    W2 = np.random.randn(n_y, n_h) * 0.01\n",
        "    b2 = np.zeros((n_y, 1))\n",
        "    # YOUR CODE ENDS HERE\n"
    ],
    "def initialize_parameters_deep(layer_dims):": [
        "        # YOUR CODE STARTS HERE\n",
        "        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01\n",
        "        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))\n",
        "        # YOUR CODE ENDS HERE\n"
    ],
    "def linear_forward(A, W, b):": [
        "    # YOUR CODE STARTS HERE\n",
        "    Z = np.dot(W, A) + b\n",
        "    # YOUR CODE ENDS HERE\n"
    ],
    "def linear_activation_forward(A_prev, W, b, activation):": [
        "        # YOUR CODE STARTS HERE\n",
        "        Z, linear_cache = linear_forward(A_prev, W, b)\n",
        "        A, activation_cache = sigmoid(Z)\n",
        "        # YOUR CODE ENDS HERE\n",
        "        # YOUR CODE STARTS HERE\n",
        "        Z, linear_cache = linear_forward(A_prev, W, b)\n",
        "        A, activation_cache = relu(Z)\n",
        "        # YOUR CODE ENDS HERE\n"
    ],
    "def L_model_forward(X, parameters):": [
        "        # YOUR CODE STARTS HERE\n",
        "        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation='relu')\n",
        "        caches.append(cache)\n",
        "        # YOUR CODE ENDS HERE\n",
        "    # YOUR CODE STARTS HERE\n",
        "    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation='sigmoid')\n",
        "    caches.append(cache)\n",
        "    # YOUR CODE ENDS HERE\n"
    ],
    "def compute_cost(AL, Y):": [
        "    # YOUR CODE STARTS HERE\n",
        "    cost = -1/m * np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL))\n",
        "    # YOUR CODE ENDS HERE\n"
    ],
    "def linear_backward(dZ, cache):": [
        "    # YOUR CODE STARTS HERE\n",
        "    dW = 1/m * np.dot(dZ, A_prev.T)\n",
        "    db = 1/m * np.sum(dZ, axis=1, keepdims=True)\n",
        "    dA_prev = np.dot(W.T, dZ)\n",
        "    # YOUR CODE ENDS HERE\n"
    ],
    "def linear_activation_backward(dA, cache, activation):": [
        "        # YOUR CODE STARTS HERE\n",
        "        dZ = relu_backward(dA, activation_cache)\n",
        "        dA_prev, dW, db = linear_backward(dZ, linear_cache)\n",
        "        # YOUR CODE ENDS HERE\n",
        "        # YOUR CODE STARTS HERE\n",
        "        dZ = sigmoid_backward(dA, activation_cache)\n",
        "        dA_prev, dW, db = linear_backward(dZ, linear_cache)\n",
        "        # YOUR CODE ENDS HERE\n"
    ],
    "def L_model_backward(AL, Y, caches):": [
        "    # YOUR CODE STARTS HERE\n",
        "    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))\n",
        "    # YOUR CODE ENDS HERE\n",
        "    # YOUR CODE STARTS HERE\n",
        "    current_cache = caches[L-1]\n",
        "    dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dAL, current_cache, activation='sigmoid')\n",
        "    grads['dA' + str(L-1)] = dA_prev_temp\n",
        "    grads['dW' + str(L)] = dW_temp\n",
        "    grads['db' + str(L)] = db_temp\n",
        "    # YOUR CODE ENDS HERE\n",
        "        # YOUR CODE STARTS HERE\n",
        "        current_cache = caches[l]\n",
        "        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads['dA' + str(l + 1)], current_cache, activation='relu')\n",
        "        grads['dA' + str(l)] = dA_prev_temp\n",
        "        grads['dW' + str(l + 1)] = dW_temp\n",
        "        grads['db' + str(l + 1)] = db_temp\n",
        "        # YOUR CODE ENDS HERE\n"
    ],
    "def update_parameters(params, grads, learning_rate):": [
        "        # YOUR CODE STARTS HERE\n",
        "        parameters['W' + str(l+1)] = parameters['W' + str(l+1)] - learning_rate * grads['dW' + str(l+1)]\n",
        "        parameters['b' + str(l+1)] = parameters['b' + str(l+1)] - learning_rate * grads['db' + str(l+1)]\n",
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

with open('Building_your_Deep_Neural_Network_Step_by_Step.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
