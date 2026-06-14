import json

with open('Python_Basics_with_Numpy.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

replacements = {
    "def basic_sigmoid(x):": [
        "    # YOUR CODE STARTS HERE\n",
        "    s = 1 / (1 + math.exp(-x))\n",
        "    # YOUR CODE ENDS HERE\n"
    ],
    "def sigmoid(x):": [
        "    # YOUR CODE STARTS HERE\n",
        "    s = 1 / (1 + np.exp(-x))\n",
        "    # YOUR CODE ENDS HERE\n"
    ],
    "def sigmoid_derivative(x):": [
        "    # YOUR CODE STARTS HERE\n",
        "    s = 1 / (1 + np.exp(-x))\n",
        "    ds = s * (1 - s)\n",
        "    # YOUR CODE ENDS HERE\n"
    ],
    "def image2vector(image):": [
        "    # YOUR CODE STARTS HERE\n",
        "    v = image.reshape(image.shape[0] * image.shape[1] * image.shape[2], 1)\n",
        "    # YOUR CODE ENDS HERE\n"
    ],
    "def normalize_rows(x):": [
        "    # YOUR CODE STARTS HERE\n",
        "    x_norm = np.linalg.norm(x, ord=2, axis=1, keepdims=True)\n",
        "    x = x / x_norm\n",
        "    # YOUR CODE ENDS HERE\n"
    ],
    "def softmax(x):": [
        "    # YOUR CODE STARTS HERE\n",
        "    x_exp = np.exp(x)\n",
        "    x_sum = np.sum(x_exp, axis=1, keepdims=True)\n",
        "    s = x_exp / x_sum\n",
        "    # YOUR CODE ENDS HERE\n"
    ],
    "def L1(yhat, y):": [
        "    # YOUR CODE STARTS HERE\n",
        "    loss = np.sum(np.abs(yhat - y))\n",
        "    # YOUR CODE ENDS HERE\n"
    ],
    "def L2(yhat, y):": [
        "    # YOUR CODE STARTS HERE\n",
        "    loss = np.sum(np.square(yhat - y))\n",
        "    # YOUR CODE ENDS HERE\n"
    ]
}

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if 'YOUR CODE' in source:
            for key, rep_lines in replacements.items():
                if key in source:
                    # Find indices
                    start_idx = -1
                    end_idx = -1
                    for i, line in enumerate(cell['source']):
                        if 'YOUR CODE STARTS HERE' in line:
                            start_idx = i
                        elif 'YOUR CODE ENDS HERE' in line:
                            end_idx = i
                            break
                    if start_idx != -1 and end_idx != -1:
                        cell['source'] = cell['source'][:start_idx] + rep_lines + cell['source'][end_idx+1:]
                    break

with open('Python_Basics_with_Numpy.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
    
print("Notebook başarıyla çözüldü ve kaydedildi!")
