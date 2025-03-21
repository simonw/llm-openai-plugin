from enum import Enum
from llm import (
    AsyncKeyModel,
    KeyModel,
    hookimpl,
    Options,
    Prompt,
    Response,
    Conversation,
)
import openai
from pydantic import Field, create_model
from typing import AsyncGenerator, Iterator, Optional


@hookimpl
def register_models(register):
    models = {
        "gpt-4o": {"vision": True},
        "gpt-4o-mini": {"vision": True},
        "o3-mini": {"reasoning": True},
        "o1-mini": {"reasoning": True},
        "o1": {"reasoning": True, "vision": True},
        "o1-pro": {"reasoning": True, "vision": True, "streaming": False},
    }
    for model_id, options in models.items():
        register(
            ResponsesModel(model_id, **options),
            AsyncResponsesModel(model_id, **options),
        )


class TruncationEnum(str, Enum):
    auto = "auto"
    disabled = "disabled"


class ImageDetailEnum(str, Enum):
    low = "low"
    high = "high"
    auto = "auto"


class ReasoningEffortEnum(str, Enum):
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
            "probability mass. So 0.1 means only the tokens comprising the top "
            "10% probability mass are considered. Recommended to use top_p or "
            "temperature but not both."
        ),
        ge=0,
        le=1,
        default=None,
    )
    store: Optional[bool] = Field(
        description=(
            "Whether to store the generated model response for later retrieval via API."
        ),
        default=None,
    )
    truncation: Optional[TruncationEnum] = Field(
        description=(
            "The truncation strategy to use for the model response. If 'auto' and the "
            "context of this response and previous ones exceeds the model's context "
            "window size, the model will truncate the response to fit the context "
            "window by dropping input items in the middle of the conversation."
        ),
        default=None,
    )


class VisionOptions(Options):
    image_detail: Optional[ImageDetailEnum] = Field(
        description=(
            "low = fixed tokens per image. high = more tokens for larger images. "
            "auto = model decides. Default is low."
        ),
        default=None,
    )


class ReasoningOptions(Options):
    reasoning_effort: Optional[ReasoningEffortEnum] = Field(
        description=(
            "Constraints effort on reasoning for reasoning models. Currently supported "
            "values are low, medium, and high. Reducing reasoning effort can result in "
            "faster responses and fewer tokens used on reasoning in a response."
        ),
        default=None,
    )


class _SharedResponses:
    needs_key = "openai"
    key_env_var = "OPENAI_API_KEY"

    def __init__(self, model_name, vision=False, streaming=True, reasoning=False):
        self.model_id = "openai/" + model_name
        self.model_name = model_name
        self.can_stream = streaming
        options = [BaseOptions]
        if vision:
            self.attachment_types = {
                "image/png",
                "image/jpeg",
                "image/webp",
                "image/gif",
                "application/pdf",
            }
            options.append(VisionOptions)
        if reasoning:
            options.append(ReasoningOptions)
        self.Options = combine_options(*options)

    def __str__(self):
        return f"OpenAI Responses: {self.model_id}"

    def build_messages(self, prompt, conversation):
        messages = []
        current_system = None
        image_detail = prompt.options.image_detail or "low"
        if conversation is not None:
            for prev_response in conversation.responses:
                if (
                    prev_response.prompt.system
                    and prev_response.prompt.system != current_system
                ):
                    messages.append(
                        {"role": "system", "content": prev_response.prompt.system}
                    )
                    current_system = prev_response.prompt.system
                if prev_response.attachments:
                    attachment_message = []
                    if prev_response.prompt.prompt:
                        attachment_message.append(
                            {"type": "input_text", "text": prev_response.prompt.prompt}
                        )
                    for attachment in prev_response.attachments:
                        attachment_message.append(_attachment(attachment, image_detail))
                    messages.append({"role": "user", "content": attachment_message})
                else:
                    messages.append(
                        {"role": "user", "content": prev_response.prompt.prompt}
                    )
                messages.append(
                    {"role": "assistant", "content": prev_response.text_or_raise()}
                )
        if prompt.system and prompt.system != current_system:
            messages.append({"role": "system", "content": prompt.system})
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

    def build_kwargs(self, prompt, messages):
        kwargs = {"model": self.model_name, "input": messages}
        for option in (
            "max_output_tokens",
            "temperature",
            "top_p",
            "store",
            "truncation",
        ):
            value = getattr(prompt.options, option, None)
            if value is not None:
                kwargs[option] = value
        return kwargs


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
        messages = self.build_messages(prompt, conversation)
        kwargs = self.build_kwargs(prompt, messages)
        kwargs["stream"] = stream
        if stream:
            for event in client.responses.create(**kwargs):
                if event.type == "response.output_text.delta":
                    yield event.delta
                elif event.type == "response.completed":
                    response.response_json = event.response.model_dump()
        else:
            client_response = client.responses.create(**kwargs)
            yield client_response.output_text
            response.response_json = client_response.model_dump()


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
        messages = self.build_messages(prompt, conversation)
        kwargs = self.build_kwargs(prompt, messages)
        kwargs["stream"] = stream
        if stream:
            completion = await client.responses.create(**kwargs)
            async for event in completion:
                if event.type == "response.output_text.delta":
                    yield event.delta
                elif event.type == "response.completed":
                    response.response_json = event.response.model_dump()
        else:
            client_response = await client.responses.create(**kwargs)
            yield client_response.output_text
            response.response_json = client_response.model_dump()


def _attachment(attachment, image_detail):
    url = attachment.url
    base64_content = ""
    if not url or attachment.resolve_type().startswith("audio/"):
        base64_content = attachment.base64_content()
        url = f"data:{attachment.resolve_type()};base64,{base64_content}"
    if attachment.resolve_type() == "application/pdf":
        if not base64_content:
            base64_content = attachment.base64_content()
        return {
            "type": "file",
            "file": {
                "filename": f"{attachment.id()}.pdf",
                "file_data": f"data:application/pdf;base64,{base64_content}",
            },
        }
    if attachment.resolve_type().startswith("image/"):
        return {"type": "input_image", "image_url": url, "detail": image_detail}
    else:
        format_ = "wav" if attachment.resolve_type() == "audio/wav" else "mp3"
        return {
            "type": "input_audio",
            "input_audio": {
                "data": base64_content,
                "format": format_,
            },
        }


def combine_options(*mixins):
    # reversed() here makes --options display order correct
    return create_model("CombinedOptions", __base__=tuple(reversed(mixins)))
