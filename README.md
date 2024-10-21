<div align="center">
    <a href="https://www.weavel.ai/ape">
        <img src="https://www.dropbox.com/scl/fi/h7e7lunf2x8g0teeqlrlt/Ape-Logo.png?rlkey=fc9fzxye4mls00cluv08f4vus&st=pfjsapa3&raw=1" title="Logo" style="width:200px; padding: 20px;" />
    </a>
    <h1>Ape: Open-Source Hub for Prompt Optimization</h1>
    <div>
        <a href="https://pypi.org/project/ape-core" target="_blank">
            <img src="https://img.shields.io/pypi/v/ape-core.svg" alt="PyPI Version"/>
        </a>
    </div>
</div>

## About

**Ape** is an open-source library that serves as a centralized hub for prompt optimization techniques.  
Our goal is to provide easy-to-use implementations of various prompt engineering methods, facilitating benchmarking, experimentation, and collaborative research within the community.

## Features

- **Modular Architecture**: Easily extendable classes and types in `ape-common` for building custom prompt optimization techniques.
- **Comprehensive Implementations**: 1 file clean implementations of state-of-the-art methods by inheriting from a unified `Trainer` class.
- **Benchmarking Suite**: A diverse set of benchmarks to evaluate performance across different tasks:
  - **bird-bench** (SQL)
  - **gpqa** (Reasoning)
  - **MATH** (Mathematical Reasoning)
  - **boolq** (Question Answering)
  - **NYT** (Classification)
  - More Benchmarks will be added soon
- **Community Collaboration**: A dedicated space in `ape-core/trainer/community` for proposing new architectures and sharing innovative ideas.

## Implemented Techniques

### Paper Implementations (`ape-core/trainer/paper`)

- **DSPy-MIPRO**
- **EvoPrompt**

### Community Contributions (`ape-core/trainer/community`)

- **Few-Shot Trainer**
- **TextGradient Trainer**
- **TextGrad-Evo Trainer**
- **Optuna Trainer**
- **Expel Trainer**

## Installation

```bash
pip install ape-core
```

## How to run

```python
from ape import Prompt
from ape.trainer import FewShotTrainer

student_prompt = Prompt(
    messages=messages,
    model="gpt-4o-mini",
    temperature=0.0,
    response_format=json_schema
)

trainer = FewShotTrainer(
    generator=Generator(), # You should implement your own generator
    metric=Metric(), # You should implement your own metric
    # global_metric=GlobalMetric(), # If you want to use specific metric like MICRO-F1, you should implement your own global metric
)

optimized_prompt, report = await trainer.train(
    prompt=student_prompt,
    trainset=trainset,
    testset=testset,
)
```

To enable syntax highlighting for .prompt files, consider using the Promptfile IntelliSense extension for VS Code.

## Getting Started

Explore the ape-core/trainer/paper directory to find implementations of various prompt optimization techniques. Each subdirectory corresponds to a specific paper and contains:

README.md: An overview of the method.
paper_name_trainer.py: The implementation of the technique inheriting from the Trainer class.

## Contributing

We welcome contributions to enhance Ape's capabilities and expand its collection of prompt optimization techniques. There are three main types of contributions you can make:

### 1. Paper Implementation

If you want to implement a new paper method, follow these steps:

1. Create a new directory under `ape-core/trainer/paper` with the name of the paper.
2. Implement the method by creating a new class that inherits from the `Trainer` class. Ensure the implementation is contained in a single file to maintain simplicity and clarity.
3. Add a `README.md` file in the same directory to describe the paper and its method.
4. Submit a pull request with the new implementation.

### 2. Paper Benchmarking

While paper benchmarking is still under development, this will eventually involve running benchmarks on various datasets using the methods in `ape-core/trainer`. Stay tuned for updates on how to contribute to this area.

### 3. Community Research

Community research contributions are focused on innovating beyond existing methods. This type of contribution is still under development.

For more information on contributing, please see the [`CONTRIBUTING.md`](CONTRIBUTING.md) file.

## Help and Support

If you have any questions, feedback, or suggestions, feel free to:

Raise an issue in the issue tracker.
Join the Weavel Community Discord to connect with other users and contributors.

## License

Ape is released under the MIT License.

## Acknowledgments

Special thanks to the Stanford NLP's DSPy project for inspiration and foundational ideas.
