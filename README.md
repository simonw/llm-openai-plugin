# llm-openai-plugin

[![PyPI](https://img.shields.io/pypi/v/llm-openai-plugin.svg)](https://pypi.org/project/llm-openai-plugin/)
[![Changelog](https://img.shields.io/github/v/release/simonw/llm-openai-plugin?include_prereleases&label=changelog)](https://github.com/simonw/llm-openai-plugin/releases)
[![Tests](https://github.com/simonw/llm-openai-plugin/actions/workflows/test.yml/badge.svg)](https://github.com/simonw/llm-openai-plugin/actions/workflows/test.yml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/simonw/llm-openai-plugin/blob/main/LICENSE)

[LLM](https://llm.datasette.io/) plugin for [OpenAI models](https://platform.openai.com/docs/models).

This plugin **is a preview**. LLM currently ships with OpenAI models as part of its default collection, implemented using the [Chat Completions API](https://platform.openai.com/docs/guides/responses-vs-chat-completions).

This plugin implements those same models using the new [Responses API](https://platform.openai.com/docs/api-reference/responses).

Currently the only reason to use this plugin over the LLM defaults is to access [o1-pro](https://platform.openai.com/docs/models/o1-pro), which can only be used via the Responses API.

## Installation

Install this plugin in the same environment as [LLM](https://llm.datasette.io/).
```bash
llm install llm-openai-plugin
```
## Usage

To run a prompt against `o1-pro` do this:

```bash
llm -m openai/o1-pro "Convince me that pelicans are the most noble of birds"
```

Run this to see a full list of models - they start with the `openai/` prefix:

```bash
llm models -q openai/
```

Here's the output of that command:

<!-- [[[cog
import cog
from llm import cli
from click.testing import CliRunner
runner = CliRunner()
result = runner.invoke(cli.cli, ["models", "-q", "openai/"])
cog.out(
    "```\n{}\n```".format(result.output.strip())
)
]]] -->
```
OpenAI: openai/gpt-4o
OpenAI: openai/gpt-4o-mini
OpenAI: openai/gpt-4.5-preview
OpenAI: openai/gpt-4.5-preview-2025-02-27
OpenAI: openai/o3-mini
OpenAI: openai/o1-mini
OpenAI: openai/o1
OpenAI: openai/o1-pro
OpenAI: openai/gpt-4.1
OpenAI: openai/gpt-4.1-2025-04-14
OpenAI: openai/gpt-4.1-mini
OpenAI: openai/gpt-4.1-mini-2025-04-14
OpenAI: openai/gpt-4.1-nano
OpenAI: openai/gpt-4.1-nano-2025-04-14
```
<!-- [[[end]]] -->
Add `--options` to see a full list of options that can be provided to each model.

## Development

To set up this plugin locally, first checkout the code. Then create a new virtual environment:
```bash
cd llm-openai-plugin
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

This project uses [pytest-recording](https://github.com/kiwicom/pytest-recording) to record OpenAI API responses for the tests, and [syrupy](https://github.com/syrupy-project/syrupy) to capture snapshots of their results.

If you add a new test that calls the API you can capture the API response and snapshot like this:
```bash
PYTEST_OPENAI_API_KEY="$(llm keys get openai)" pytest --record-mode once --snapshot-update
```
Then review the new snapshots in `tests/__snapshots__/` to make sure they look correct.
