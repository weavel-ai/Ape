<div align="center">
    <a href="https://www.weavel.ai/ape">
        <img src="https://www.dropbox.com/scl/fi/h7e7lunf2x8g0teeqlrlt/Ape-Logo.png?rlkey=fc9fzxye4mls00cluv08f4vus&st=pfjsapa3&raw=1" title="Logo" style="width:200px; padding: 20px;" />
    </a>
    <h1>Ape: Open-Source Hub for AI Prompt Engineering</h1>
    <div style="display: flex; justify-content: center; gap: 10px;">
        <a href="https://www.ycombinator.com/companies/weavel">
            <img
                src="https://img.shields.io/badge/Y%20Combinator-S24-orange?style=flat-square"
                alt="Y Combinator S24"
            />
        </a>
        <a href="https://github.com/weavel-ai/Ape/blob/main/LICENSE" target="_blank">
            <img src="https://img.shields.io/pypi/l/ape-common.svg" alt="License" />
        </a>
        <a href="https://pypi.org/project/ape-core" target="_blank">
            <img src="https://img.shields.io/pypi/v/ape-core.svg" alt="PyPI Version"/>
        </a>
    </div>
</div>

## About

**Ape (AI prompt engineer)** is a prompt optimization library with implementations of various state-of-the-art prompt optimization methods.  
**Ape** focuses on easier benchmarking, experimentation, and collaborative research of various techniques within the community. Ape makes it easy to apply and compare different prompt optimization techniques.

[Read the docs →](https://ape-prompt.vercel.app)

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

### Community Implementations (`ape-core/trainer/community`)

- **Few-Shot Trainer**
- **TextGradient Trainer**
- **TextGrad-Evo Trainer**
- **Optuna Trainer**
- **Expel Trainer**

## Experiment Results

If you want to see the experiment results of methods over various benchmarks, please refer to the [Experiment Results](./experiments/benchmarks/RESULT.md) file.

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

If you want to see the tutorial code to run Ape, please refer to the [Example Experiment Code](./experiments/EXAMPLE_EXPERIMENT.ipynb).

## Contributing

We welcome contributions to enhance Ape's capabilities and expand its collection of prompt optimization techniques. There are four main types of contributions you can make:

### 1. Paper Implementation Contributions

We aim to implement every paper on prompt optimization or automated prompt engineering.
If you want to implement a new paper, please refer to the [`CONTRIBUTING.md`](CONTRIBUTING.md) file for more information.

### 2. Benchmark Contributions

All prompt optimization methods will be evaluated on various benchmarks to understand the strengths and weaknesses of each approach.
Currently, we have 5 benchmarks: bird-bench, gpqa, math, boolq, and NYT.

### 3. Community Research Contributions

Community research contributions focus on innovating beyond existing methods.

### 4. Other Contributions

These contributions include bug fixes, documentation improvements, experiment management, and more.

**For more information on contributing, please see the [`CONTRIBUTING.md`](CONTRIBUTING.md) file.**

## Help and Support

If you have any questions, feedback, or suggestions, feel free to:

Raise an issue in the issue tracker.
Join the Weavel Community Discord to connect with other users and contributors.

## License

Ape is released under the MIT License.

## Acknowledgments

Special thanks to the Stanford NLP's DSPy project for inspiration and foundational ideas.

## References

- [DSPy](https://github.com/stanfordnlp/DSPy)
- [EvoPrompt](https://arxiv.org/abs/2309.08532)
- [TextGrad](https://github.com/microsoft/TextGrad)
- [Expel](https://arxiv.org/abs/2308.10144)
- [adalflow](https://github.com/SylphAI-Inc/AdalFlow)
