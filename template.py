import os

# Define folder structure
folders = [
    "my_nn_project/data/processed",
    "my_nn_project/models",
    "my_nn_project/utils",
    "my_nn_project/notebooks"
]

# Define files with optional starter text
files = {
    "my_nn_project/data/dataset.csv": "id,feature1,feature2,target\n",  # placeholder
    "my_nn_project/models/model.py": "# Define your neural network here\n",
    "my_nn_project/utils/data_utils.py": "# Functions for loading & preprocessing data\n",
    "my_nn_project/utils/train_utils.py": "# Functions for training & evaluation\n",
    "my_nn_project/notebooks/exploration.ipynb": "",  # empty notebook
    "my_nn_project/main.py": "# Entry point: import utils, train model, evaluate\n",
    "my_nn_project/requirements.txt": "torch\npandas\nscikit-learn\n",
    "my_nn_project/README.md": "# My NN Project\n\nProject description here.\n",
    "my_nn_project/.gitignore": "*.pyc\n__pycache__/\n.ipynb_checkpoints/\n"
}

# Create folders
for folder in folders:
    os.makedirs(folder, exist_ok=True)

# Create files
for file, content in files.items():
    with open(file, "w", encoding="utf-8") as f:
        f.write(content)

print("✅ Project structure created successfully!")
