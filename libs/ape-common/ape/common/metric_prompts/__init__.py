import os
from ape.common.prompt import Prompt
import promptfile as pf

current_dir = os.path.dirname(os.path.abspath(__file__))

if os.path.exists(current_dir):
    ApeMetricPrompts = pf.Client(base_path=current_dir, prompt_class=Prompt)
else:
    raise FileNotFoundError(f"Prompts directory not found at {current_dir}")
