import os

APE_PROMPT_LIST = []

current_dir = os.path.dirname(os.path.abspath(__file__))

for file in os.listdir(current_dir):
    if file.endswith(".prompt"):
        APE_PROMPT_LIST.append(os.path.splitext(file)[0])