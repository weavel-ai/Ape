"""
Ape: your AI prompt engineer
"""

from setuptools import setup, find_namespace_packages

# Read README.md for the long description
with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="ape-core",
    version="0.6.4",
    packages=find_namespace_packages(),
    include_package_data=True,
    entry_points={},
    description="Ape: your AI prompt engineer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="weavel",
    url="https://github.com/weavel-ai/Ape",
    install_requires=[
        "pydantic",
        "pyyaml",
        "optuna",
        "psycopg2-binary",
        "numpy",
        "promptfile",
        "structlog",
        "pandas",
        "numpy",
        "sqlalchemy",
        "litellm",
        "nest_asyncio",
        "rich",
        "pysbd",
    ],
    python_requires=">=3.8.10",
    keywords=[
        "prompt",
        "prompt engineering",
        "AI prompt engineer",
        "llm",
    ],
)
