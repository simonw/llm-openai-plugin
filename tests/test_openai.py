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


@pytest.mark.vcr
def test_conversation_chaining(vcr):
    """Test that conversations use previous_response_id for chaining"""
    model = llm.get_model("openai/gpt-4o-mini")
    conversation = model.conversation()

    # First prompt - no chaining
    response1 = conversation.prompt("What is 7+7?", key=API_KEY, stream=False)
    text1 = response1.text()
    assert "14" in text1

    # Check first request didn't use previous_response_id
    first_request = json.loads(vcr.requests[0].body)
    assert "previous_response_id" not in first_request
    assert "input" in first_request

    # Second prompt - should use chaining
    response2 = conversation.prompt("Add 3 to that", key=API_KEY, stream=False)
    text2 = response2.text()
    assert "17" in text2

    # Check second request used previous_response_id
    second_request = json.loads(vcr.requests[1].body)
    assert "previous_response_id" in second_request
    # Should have previous_response_id from first response
    assert second_request["previous_response_id"] == response1.response_json["id"]
    # Input should only contain the new message (not full history)
    assert len(second_request["input"]) == 1
    assert second_request["input"][0]["content"] == "Add 3 to that"


@pytest.mark.vcr
def test_conversation_chaining_with_store_false(vcr):
    """Test that store=False falls back to full conversation history"""
    model = llm.get_model("openai/gpt-4o-mini")
    conversation = model.conversation()

    # First prompt with store=True
    response1 = conversation.prompt("What is 8+8?", key=API_KEY, stream=False, store=True)
    text1 = response1.text()
    assert "16" in text1

    # Second prompt with store=False - should NOT use chaining
    response2 = conversation.prompt("Subtract 5", key=API_KEY, stream=False, store=False)
    text2 = response2.text()
    assert "11" in text2

    # Check second request did NOT use previous_response_id
    second_request = json.loads(vcr.requests[1].body)
    assert "previous_response_id" not in second_request
    # Input should contain full conversation history
    assert len(second_request["input"]) >= 2


@pytest.mark.vcr
def test_chained_response_stored_correctly():
    """Test that chained responses have previous_response_id in response_json"""
    model = llm.get_model("openai/gpt-4o-mini")
    conversation = model.conversation()

    # First response
    response1 = conversation.prompt("Say 'first'", key=API_KEY, stream=False)
    response1.text()

    # Second response - chained
    response2 = conversation.prompt("Say 'second'", key=API_KEY, stream=False)
    response2.text()

    # Verify response_json contains previous_response_id
    assert response2.response_json is not None
    assert "previous_response_id" in response2.response_json
    assert response2.response_json["previous_response_id"] == response1.response_json["id"]
