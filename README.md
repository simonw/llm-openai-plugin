# llm-openai

[![PyPI](https://img.shields.io/pypi/v/llm-openai.svg)](https://pypi.org/project/llm-openai/)
[![Changelog](https://img.shields.io/github/v/release/simonw/llm-openai?include_prereleases&label=changelog)](https://github.com/simonw/llm-openai/releases)
[![Tests](https://github.com/simonw/llm-openai/actions/workflows/test.yml/badge.svg)](https://github.com/simonw/llm-openai/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/simonw/llm-openai/blob/main/LICENSE)

LLM plugin for OpenAI

> [!WARNING]  
> This is a very early alpha.

## Installation

Install this plugin in the same environment as [LLM](https://llm.datasette.io/).
```bash
llm install llm-openai
```
## Usage

Run this to see the models - they start with the `openai/` prefix:

```bash
llm models -q openai/
```

## Development

To set up this plugin locally, first checkout the code. Then create a new virtual environment:
```bash
cd llm-openai
python -m venv venv
source venv/bin/activate
```
Now install the dependencies and test dependencies:
```bash
llm install -e '.[test]'
```
To run the tests:
```bash
python -m pytest
```
