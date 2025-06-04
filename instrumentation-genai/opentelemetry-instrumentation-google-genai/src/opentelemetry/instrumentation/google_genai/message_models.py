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

# Copied and adapted from
# https://gist.github.com/lmolkova/09ba0de7f68280f1eac27a6acfd9b1a6?permalink_comment_id=5578799#gistcomment-5578799

from typing import Annotated, Any, List, Literal, Optional, Union

from pydantic import Base64Encoder, BaseModel, EncodedBytes, Field


class Base64OneWayEncoder(Base64Encoder):
    @classmethod
    def decode(cls, data: bytes) -> bytes:
        """NoOp"""
        return data


Base64EncodedBytes = Annotated[
    bytes, EncodedBytes(encoder=Base64OneWayEncoder)
]


class TextPart(BaseModel):
    type: Literal["text"] = "text"
    content: str


class BlobPart(BaseModel):
    type: Literal["blob"] = "blob"
    mime_type: str
    data: Base64EncodedBytes


class FileDataPart(BaseModel):
    type: Literal["file_data"] = "file_data"
    mime_type: str
    file_uri: str


class ToolCallPart(BaseModel):
    type: Literal["tool_call"] = "tool_call"
    id: str
    name: str
    arguments: Optional[Any]


class ToolCallResponsePart(BaseModel):
    type: Literal["tool_call_response"] = "tool_call_response"
    id: str
    name: Optional[str]
    result: Optional[Any]


class UnknownPart(BaseModel):
    type: Literal["unknown"] = "unknown"


MessagePart = Annotated[
    Union[
        TextPart,
        ToolCallPart,
        ToolCallResponsePart,
        # Add other message part types here as needed,
        # e.g. image URL, image blob, audio URL, structured output, hosted tool call, etc.
        FileDataPart,
        BlobPart,
        UnknownPart,
    ],
    Field(discriminator="type"),
]


class ChatMessage(BaseModel):
    role: str | None
    parts: List[MessagePart]


class SystemMessage(BaseModel):
    messages: List[ChatMessage]


class InputMessages(BaseModel):
    messages: List[ChatMessage]


class Choice(BaseModel):
    finish_reason: str
    message: ChatMessage | None


class OutputMessages(BaseModel):
    choices: List[Choice]
