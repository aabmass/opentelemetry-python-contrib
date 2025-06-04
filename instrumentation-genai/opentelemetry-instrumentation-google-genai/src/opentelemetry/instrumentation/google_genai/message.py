# Copyright The OpenTelemetry Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging

from google.genai.types import (
    Candidate,
    Content,
    Part,
)

from .message_models import (
    BlobPart,
    ChatMessage,
    Choice,
    FileDataPart,
    InputMessages,
    MessagePart,
    OutputMessages,
    SystemMessage,
    TextPart,
    ToolCallPart,
    ToolCallResponsePart,
    UnknownPart,
)

_logger = logging.getLogger(__name__)


def to_input_messages(
    *,
    contents: list[Content],
) -> InputMessages:
    return InputMessages(
        messages=[_to_chat_message(content) for content in contents]
    )


def to_output_message(
    *,
    candidates: list[Candidate],
) -> OutputMessages:
    choices = [
        Choice(
            finish_reason=candidate.finish_reason or "",
            message=_to_chat_message(candidate.content)
            if candidate.content
            else None,
        )
        for candidate in candidates
    ]
    return OutputMessages(choices=choices)


def to_system_instruction(
    *,
    contents: list[Content],
) -> SystemMessage | None:
    return SystemMessage(
        messages=[_to_chat_message(content) for content in contents]
    )


def _to_chat_message(
    content: Content,
) -> ChatMessage:
    parts = content.parts or []
    return ChatMessage(
        role=content.role, parts=[_to_part(part) for part in parts]
    )


def _to_part(part: Part) -> MessagePart:
    if (text := part.text) is not None:
        return TextPart(content=text)

    if data := part.inline_data:
        return BlobPart(mime_type=data.mime_type or "", data=data.data or b"")

    if data := part.file_data:
        return FileDataPart(
            mime_type=data.mime_type or "", file_uri=data.file_uri or ""
        )

    if call := part.function_call:
        return ToolCallPart(
            id=call.id or "", name=call.name or "", arguments=call.args
        )

    if response := part.function_response:
        return ToolCallResponsePart(
            id=response.id or "",
            name=response.name or "",
            result=response.response,
        )

    _logger.info("Unknown part dropped from telemetry %s", part)
    return UnknownPart()
