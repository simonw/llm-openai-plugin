from enum import Enum
import json
from typing import AsyncGenerator, Iterator, Optional, Dict, List

import llm
from llm import (
    AsyncKeyModel,
    KeyModel,
    hookimpl,
    Options,
    Prompt,
    Response,
    Conversation,
)
from llm.utils import simplify_usage_dict
import openai
from pydantic import Field, create_model


@hookimpl
def register_models(register):
    models = {
        "gpt-4o": {"vision": True, "supports_tools": True},
        "gpt-4o-mini": {"vision": True, "supports_tools": True},
        "gpt-4.5-preview": {"vision": True, "supports_tools": True},
        "gpt-4.5-preview-2025-02-27": {"vision": True, "supports_tools": True},
        "o3-mini": {"reasoning": True, "supports_tools": True},
        "o1-mini": {"reasoning": True, "supports_tools": True, "schemas": False},
        "o1": {"reasoning": True, "vision": True, "supports_tools": True},
        "o1-pro": {
            "reasoning": True,
            "vision": True,
            "supports_tools": True,
            "streaming": False,
        },
        # GPT-4.1
        "gpt-4.1": {"vision": True, "supports_tools": True},
        "gpt-4.1-2025-04-14": {"vision": True, "supports_tools": True},
        "gpt-4.1-mini": {"vision": True, "supports_tools": True},
        "gpt-4.1-mini-2025-04-14": {"vision": True, "supports_tools": True},
        "gpt-4.1-nano": {"vision": True, "supports_tools": True},
        "gpt-4.1-nano-2025-04-14": {"vision": True, "supports_tools": True},
        # April 16th 2025
        "o3": {
            "vision": True,
            "reasoning": True,
            "supports_tools": True,
            "streaming": False,
        },
        "o3-2025-04-16": {
            "vision": True,
            "reasoning": True,
            "supports_tools": True,
            "streaming": False,
        },
        "o3-streaming": {"vision": True, "reasoning": True, "supports_tools": True},
        "o3-2025-04-16-streaming": {
            "vision": True,
            "reasoning": True,
            "supports_tools": True,
        },
        "o4-mini": {"vision": True, "reasoning": True, "supports_tools": True},
        "o4-mini-2025-04-16": {
            "vision": True,
            "reasoning": True,
            "supports_tools": True,
        },
        # May 16th 2025
        "codex-mini-latest": {
            "vision": True,
            "reasoning": True,
            "supports_tools": True,
        },
        # June 10th 2025
        "o3-pro": {"vision": True, "reasoning": True, "supports_tools": True},
    }
    for model_id, options in models.items():
        register(
            ResponsesModel(model_id, **options),
            AsyncResponsesModel(model_id, **options),
        )


# --------------------------------------------------------------------- #
#                                Options                                #
# --------------------------------------------------------------------- #
class TruncationEnum(str, Enum):
    auto = "auto"
    disabled = "disabled"


class ImageDetailEnum(str, Enum):
    low = "low"
    high = "high"
    auto = "auto"


class ReasoningEffortEnum(str, Enum):
    minimal = "minimal"
    low = "low"
    medium = "medium"
    high = "high"


class BaseOptions(Options):
    max_output_tokens: Optional[int] = Field(
        description=(
            "An upper bound for the number of tokens that can be generated for a "
            "response, including visible output tokens and reasoning tokens."
        ),
        ge=0,
        default=None,
    )
    temperature: Optional[float] = Field(
        description=(
            "What sampling temperature to use, between 0 and 2. Higher values like "
            "0.8 will make the output more random, while lower values like 0.2 will "
            "make it more focused and deterministic."
        ),
        ge=0,
        le=2,
        default=None,
    )
    top_p: Optional[float] = Field(
        description=(
            "An alternative to sampling with temperature, called nucleus sampling, "
            "where the model considers the results of the tokens with top_p "
            "probability mass."
        ),
        ge=0,
        le=1,
        default=None,
    )
    store: Optional[bool] = Field(
        description="Whether to store the generated model response for later retrieval.",
        default=None,
    )
    truncation: Optional[TruncationEnum] = Field(
        description=(
            "Truncation strategy. If 'auto' and the conversation exceeds the context "
            "window, older content is truncated automatically."
        ),
        default=None,
    )


class VisionOptions(Options):
    image_detail: Optional[ImageDetailEnum] = Field(
        description=(
            "low = fixed token cost per image, high = cost scales with image size, "
            "auto = model decides. Default: low."
        ),
        default=None,
    )


class ReasoningOptions(Options):
    reasoning_effort: Optional[ReasoningEffortEnum] = Field(
        description=(
            "low/medium/high – allows you to limit the amount of reasoning the model "
            "performs, trading quality for speed and cost."
        ),
        default=None,
    )


# --------------------------------------------------------------------- #
#                            Shared utilities                           #
# --------------------------------------------------------------------- #
def combine_options(*mixins):
    # reversed() keeps CLI option order intuitive
    return create_model("CombinedOptions", __base__=tuple(reversed(mixins)))


def additional_properties_false(input_dict: dict) -> dict:
    """
    Recursively add 'additionalProperties': False anywhere a dict contains 'properties'.
    """
    result = {}
    for key, value in input_dict.items():
        if isinstance(value, dict):
            result[key] = additional_properties_false(value)
        elif isinstance(value, list):
            result[key] = [
                additional_properties_false(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            result[key] = value
    if "properties" in input_dict:
        result["additionalProperties"] = False
    return result


def _attachment(attachment, image_detail):
    """
    Produce the new Responses-endpoint attachment structure.
    """
    url = attachment.url
    base64_content = ""
    if not url or attachment.resolve_type().startswith("audio/"):
        base64_content = attachment.base64_content()
        url = f"data:{attachment.resolve_type()};base64,{base64_content}"
    if attachment.resolve_type() == "application/pdf":
        if not base64_content:
            base64_content = attachment.base64_content()
        return {
            "type": "input_file",
            "filename": f"{attachment.id()}.pdf",
            "file_data": f"data:application/pdf;base64,{base64_content}",
        }
    if attachment.resolve_type().startswith("image/"):
        return {
            "type": "input_image",
            "image_url": url,
            "detail": image_detail,
        }
    # audio
    format_ = "wav" if attachment.resolve_type() == "audio/wav" else "mp3"
    return {
        "type": "input_audio",
        "input_audio": {"data": base64_content, "format": format_},
    }


# --------------------------------------------------------------------- #
#                           Shared model logic                          #
# --------------------------------------------------------------------- #
class _SharedResponses:
    needs_key = "openai"
    key_env_var = "OPENAI_API_KEY"

    def __init__(
        self,
        model_name,
        vision=False,
        streaming=True,
        schemas=True,
        reasoning=False,
        supports_tools=False,
    ):
        self.model_id = "openai/" + model_name
        streaming_suffix = "-streaming"
        if model_name.endswith(streaming_suffix):
            model_name = model_name[: -len(streaming_suffix)]
        self.model_name = model_name
        self.can_stream = streaming
        self.supports_schema = schemas
        self.supports_tools = supports_tools

        option_mixins = [BaseOptions]
        self.vision = vision
        if vision:
            self.attachment_types = {
                "image/png",
                "image/jpeg",
                "image/webp",
                "image/gif",
                "application/pdf",
            }
            option_mixins.append(VisionOptions)
        if reasoning:
            option_mixins.append(ReasoningOptions)
        self.Options = combine_options(*option_mixins)

    # ------------------------- usage tracking ------------------------ #
    def set_usage(self, response: Response, usage):
        if not usage:
            return
        if not isinstance(usage, dict):
            usage = usage.model_dump()
        input_tokens = usage.pop("input_tokens")
        output_tokens = usage.pop("output_tokens")
        usage.pop("total_tokens", None)
        response.set_usage(
            input=input_tokens, output=output_tokens, details=simplify_usage_dict(usage)
        )

    # ------------------------- message building ---------------------- #
    def _build_messages(
        self, prompt: Prompt, conversation: Optional[Conversation]
    ) -> List[Dict]:
        messages: List[Dict] = []
        current_system = None
        image_detail = None
        if self.vision:
            image_detail = prompt.options.image_detail or "low"

        # Walk the previous conversation
        if conversation is not None:
            for prev_response in conversation.responses:
                # system prompt
                if (
                    prev_response.prompt.system
                    and prev_response.prompt.system != current_system
                ):
                    messages.append(
                        {"role": "system", "content": prev_response.prompt.system}
                    )
                    current_system = prev_response.prompt.system

                # user message (+ attachments)
                if prev_response.attachments:
                    attachment_message = []
                    if prev_response.prompt.prompt:
                        attachment_message.append(
                            {"type": "input_text", "text": prev_response.prompt.prompt}
                        )
                    for attachment in prev_response.attachments:
                        attachment_message.append(_attachment(attachment, image_detail))
                    messages.append({"role": "user", "content": attachment_message})
                elif prev_response.prompt.prompt:
                    messages.append(
                        {"role": "user", "content": prev_response.prompt.prompt}
                    )

                # tool results that the user ran
                for tool_result in prev_response.prompt.tool_results:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_result.tool_call_id,
                            "content": tool_result.output,
                        }
                    )

                # assistant text
                prev_text = prev_response.text_or_raise()
                if prev_text:
                    messages.append({"role": "assistant", "content": prev_text})

                # assistant tool calls
                prev_tool_calls = prev_response.tool_calls_or_raise()
                if prev_tool_calls:
                    messages.append(
                        {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "type": "function",
                                    "id": tc.tool_call_id,
                                    "function": {
                                        "name": tc.name,
                                        "arguments": json.dumps(tc.arguments),
                                    },
                                }
                                for tc in prev_tool_calls
                            ],
                        }
                    )

        # Current system prompt (if changed)
        if prompt.system and prompt.system != current_system:
            messages.append({"role": "system", "content": prompt.system})

        # tool results provided with the current prompt
        for tool_result in prompt.tool_results:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_result.tool_call_id,
                    "content": tool_result.output,
                }
            )

        # Current user message
        if not prompt.attachments:
            messages.append({"role": "user", "content": prompt.prompt or ""})
        else:
            attachment_message = []
            if prompt.prompt:
                attachment_message.append({"type": "input_text", "text": prompt.prompt})
            for attachment in prompt.attachments:
                attachment_message.append(_attachment(attachment, image_detail))
            messages.append({"role": "user", "content": attachment_message})

        return messages

    # ------------------------ kwargs building ------------------------ #
    def _build_kwargs(
        self, prompt: Prompt, conversation: Optional[Conversation]
    ) -> Dict:
        messages = self._build_messages(prompt, conversation)
        kwargs: Dict = {"model": self.model_name, "input": messages}

        # Options
        for option_name in (
            "max_output_tokens",
            "temperature",
            "top_p",
            "store",
            "truncation",
        ):
            value = getattr(prompt.options, option_name, None)
            if value is not None:
                kwargs[option_name] = value

        # Response format using schema
        if self.supports_schema and prompt.schema:
            kwargs["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": "output",
                    "schema": additional_properties_false(prompt.schema),
                }
            }

        # Tool definitions
        if self.supports_tools and prompt.tools:
            tools_list = []
            for tool in prompt.tools:
                entry: Dict = {
                    "type": "function",
                    "name": tool.name,
                    "description": tool.description or None,
                    "parameters": tool.input_schema,
                }
                # include strict if specified
                if getattr(tool, "strict", None) is not None:
                    entry["strict"] = tool.strict
                tools_list.append(entry)
            kwargs["tools"] = tools_list

        return kwargs

    # ----------------------- streaming handling ---------------------- #
    def _handle_event(self, event, response: Response):
        """
        Return any delta text that should be yielded, update response & usage.
        """
        # 1. Normal text deltas
        if event.type == "response.output_text.delta":
            return event.delta

        # 2. Final, non-streaming payload
        if event.type == "response.completed":
            response.response_json = event.response.model_dump()
            self.set_usage(response, event.response.usage)
            # Save any tool calls (non-streaming path)
            for tc in getattr(event.response, "tool_calls", []) or []:
                response.add_tool_call(
                    llm.ToolCall(
                        tool_call_id=tc.id,
                        name=tc.function.name,
                        arguments=json.loads(tc.function.arguments),
                    )
                )
            return None

        # Other event types we ignore for delta output
        return None

    def _finish_non_streaming_response(self, response: Response, client_response):
        """
        Populate response object for non-streaming mode.
        """
        response.response_json = client_response.model_dump()
        self.set_usage(response, client_response.usage)
        # Capture tool calls (if any)
        for tc in getattr(client_response, "tool_calls", []) or []:
            response.add_tool_call(
                llm.ToolCall(
                    tool_call_id=tc.id,
                    name=tc.function.name,
                    arguments=json.loads(tc.function.arguments),
                )
            )


# --------------------------------------------------------------------- #
#                           Sync implementation                         #
# --------------------------------------------------------------------- #
class ResponsesModel(_SharedResponses, KeyModel):
    def execute(
        self,
        prompt: Prompt,
        stream: bool,
        response: Response,
        conversation: Optional[Conversation],
        key: Optional[str],
    ) -> Iterator[str]:
        client = openai.OpenAI(api_key=self.get_key(key))
        kwargs = self._build_kwargs(prompt, conversation)
        kwargs["stream"] = stream

        if stream:
            # Accumulate tool calls from streaming events
            tool_calls: Dict[int, Dict] = {}
            for event in client.responses.create(**kwargs):
                # New function call started
                if event.type == "response.output_item.added":
                    item = event.item
                    if item.type == "function_call":
                        tool_calls[event.output_index] = {
                            "id": item.id,
                            "call_id": item.call_id,
                            "name": item.name,
                            "arguments": "",
                        }
                # Arguments delta
                elif event.type == "response.function_call_arguments.delta":
                    idx = event.output_index
                    if idx in tool_calls:
                        tool_calls[idx]["arguments"] += event.delta
                # Function call done
                elif event.type == "response.function_call_arguments.done":
                    idx = event.output_index
                    item = event.item
                    tool_calls[idx] = {
                        "id": item.id,
                        "call_id": item.call_id,
                        "name": item.name,
                        "arguments": item["arguments"],
                    }

                # Handle normal text deltas and final event
                delta = self._handle_event(event, response)
                if delta is not None:
                    yield delta

            # After streaming finished, persist tool call list
            for tc in tool_calls.values():
                response.add_tool_call(
                    llm.ToolCall(
                        tool_call_id=tc["call_id"],
                        name=tc["name"],
                        arguments=json.loads(tc["arguments"]),
                    )
                )
        else:
            client_response = client.responses.create(**kwargs)
            # yield any final text (if present)
            if getattr(client_response, "output_text", None) is not None:
                yield client_response.output_text
            self._finish_non_streaming_response(response, client_response)


# --------------------------------------------------------------------- #
#                          Async implementation                         #
# --------------------------------------------------------------------- #
class AsyncResponsesModel(_SharedResponses, AsyncKeyModel):
    async def execute(
        self,
        prompt: Prompt,
        stream: bool,
        response: Response,
        conversation: Optional[Conversation],
        key: Optional[str],
    ) -> AsyncGenerator[str, None]:
        client = openai.AsyncOpenAI(api_key=self.get_key(key))
        kwargs = self._build_kwargs(prompt, conversation)
        kwargs["stream"] = stream

        if stream:
            tool_calls: Dict[int, Dict] = {}
            async for event in await client.responses.create(**kwargs):
                # New function call started
                if event.type == "response.output_item.added":
                    item = event.item
                    if item.type == "function_call":
                        tool_calls[event.output_index] = {
                            "id": item.id,
                            "call_id": item.call_id,
                            "name": item.name,
                            "arguments": "",
                        }
                # Arguments delta
                elif event.type == "response.function_call_arguments.delta":
                    idx = event.output_index
                    if idx in tool_calls:
                        tool_calls[idx]["arguments"] += event.delta
                # Function call done
                elif event.type == "response.function_call_arguments.done":
                    idx = event.output_index
                    item = event.item
                    tool_calls[idx] = {
                        "id": item.id,
                        "call_id": item.call_id,
                        "name": item.name,
                        "arguments": item["arguments"],
                    }

                # Handle normal text deltas and final event
                delta = self._handle_event(event, response)
                if delta is not None:
                    yield delta

            # After streaming finished, persist tool call list
            for tc in tool_calls.values():
                response.add_tool_call(
                    llm.ToolCall(
                        tool_call_id=tc["call_id"],
                        name=tc["name"],
                        arguments=json.loads(tc["arguments"]),
                    )
                )
        else:
            client_response = await client.responses.create(**kwargs)
            if getattr(client_response, "output_text", None) is not None:
                yield client_response.output_text
            self._finish_non_streaming_response(response, client_response)
