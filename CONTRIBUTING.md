# Welcome Contributors

We welcome contributions to enhance Ape's capabilities and improve its performance. To report bugs, create a [GitHub issue](https://github.com/weavel-ai/Ape/issues).

> Before contributing, read through the existing issues and pull requests to see if someone else is already working on something similar. That way you can avoid duplicating efforts.

To contribute, please follow these steps:

1. Fork the Ape repository on GitHub.
2. Create a new branch for your feature or bug fix.
3. Make your changes and ensure that the code passes all tests.
4. Submit a pull request describing your changes and their benefits.

## Pull Request Guidelines

When submitting a pull request, please follow these guidelines:

1. **Title**: please include following prefixes:

   - `Feature:` for new features
   - `Fix:` for bug fixes
   - `Docs:` for documentation changes
   - `Refactor:` for code refactoring
   - `Improve:` for performance improvements
   - `Other:` for other changes

   for example:

   - `Feature: added new feature to the code`
   - `Fix: fixed the bug in the code`

2. **Description**: Provide a clear and detailed description of your changes in the pull request. Explain the problem you are solving, the approach you took, and any potential side effects or limitations of your changes.
3. **Documentation**: Update the relevant documentation to reflect your changes. This includes the README file, code comments, and any other relevant documentation.
4. **Dependencies**: If your changes require new dependencies, ensure that they are properly documented and added to the `requirements.txt` or `package.json` files.
5. if the pull request does not meet the above guidelines, it may be closed without merging.

**Note**: Please ensure that you have the latest version of the code before creating a pull request. If you have an existing fork, just sync your fork with the latest version of the Ape repository.

Please adhere to the coding conventions, maintain clear documentation, and provide thorough testing for your contributions.

## Contribution Types

There are four main types of contributions you can make to the Ape project:

### 1. Paper Implementation Contribution

For contributions implementing a new method related to prompt optimization or automated prompt engineering, follow these steps:

1. Create a new directory under `libs/ape-core/ape/core/trainer/paper` named after the paper you are implementing.
2. Implement the method in a single Python file by creating a class that inherits from the Trainer class. Keep the implementation simple and contained in one file.
3. Add a README.md file in the same directory to describe the paper and its method. Use the format provided in [`libs/ape-core/ape/core/trainer/papers/EXAMPLE_README.md`](./libs/ape-core/ape/core/trainer/papers/EXAMPLE_README.md).
4. Submit a pull request following the guidelines above.
5. Any prompts required for the implementation should be added to the `libs/ape-core/ape/core/core_prompts` directory.

To submit an issue for this type of contribution, use the "Paper Implementation Contribution" template.

### 2. Benchmark Contribution

Benchmark contributions involve selecting a relevant dataset and implementing a Jupyter Notebook to test various methods on that dataset. The dataset should reflect real-world use cases of LLMs.

To contribute, follow these steps:

1. Create a directory under `experiments/benchmarks` with the name of the dataset.
2. Include a `DESCRIPTION.md` file in the directory, following the format in `experiments/benchmarks/EXAMPLE_BENCHMARK_DESCRIPTION.md`.
3. Implement a `run_benchmark.ipynb` file for testing the dataset on different methods.

To submit an issue for benchmark contributions, use the "Benchmark Contribution" template.

### 3. Community Research Contribution

Community Research contributions involve proposing new prompt optimization algorithms based on existing research and experimental results. These contributions will be collaboratively developed through discussions on Discord.

You can propose new ideas by submitting an issue using the "Community Research Suggestion" template.

### 4. Other Contributions

This category includes bug fixes, experimental management, performance improvements, or other general enhancements. For these contributions, submit an issue using the "Bug Report" template.
