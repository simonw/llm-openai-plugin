import json
import llm
import os
from pydantic import BaseModel
import pytest

API_KEY = os.environ.get("PYTEST_OPENAI_API_KEY", None) or "badkey"


def test_plugin_is_installed():
    model_ids = [model.model_id for model in llm.get_models()]
    assert "openai/gpt-4o-mini" in model_ids


@pytest.mark.parametrize(
    "options",
    (
        {"max_output_tokens": 24},
        {"temperature": 0.5},
        {"top_p": 0.5},
        {"store": True},
        {"truncation": "auto"},
    ),
)
@pytest.mark.vcr
def test_options(options, snapshot, vcr):
    model = llm.get_model("openai/gpt-4o-mini")
    response = model.prompt("say hi", key=API_KEY, stream=False, **options)
    assert response.text() == snapshot
    # Was the option sent to the API?
    api_input = json.loads(vcr.requests[0].body)
    assert all(item in api_input.items() for item in options.items())
    usage = response.usage()
    assert usage.input == 27


@pytest.mark.asyncio
@pytest.mark.vcr
async def test_async_model(snapshot):
    model = llm.get_async_model("openai/gpt-4o-mini")
    response = await model.prompt("say hi", key=API_KEY)
    output = await response.text()
    assert output == snapshot
    usage = await response.usage()
    assert usage.input == 27
    assert usage.output == 11


class Dog(BaseModel):
    name: str
    age: int
    bio: str


@pytest.mark.asyncio
@pytest.mark.vcr
async def test_async_model_schema(snapshot):
    model = llm.get_async_model("openai/gpt-4o-mini")
    response = await model.prompt("invent a dog", key=API_KEY, schema=Dog)
    output = await response.text()
    assert json.loads(output) == snapshot


@pytest.mark.vcr
def test_tools(snapshot):
    model = llm.get_model("openai/gpt-5-mini")

    def simple_tool(number):
        "A simple tool"
        return "This is a simple tool, {}".format(number)

    chain_response = model.chain(
        "Call simple_tool passing 5",
        tools=[simple_tool],
        key=API_KEY
    )
    output = chain_response.text()
    assert output == snapshot
