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

import pytest
from google.genai import Client, types

from ..common.otel_assertions import OTelAssertions


@pytest.mark.vcr
def test_span(
    genai_client: Client,
    otel_assertions: OTelAssertions,
    instrument_with_content,
):
    genai_client.models.generate_content(
        model="gemini-2.0-flash", contents="Does this work?"
    )
    otel_assertions.assert_has_span_named("generate_content gemini-2.0-flash")

    span = otel_assertions.get_span_named("generate_content gemini-2.0-flash")
    assert span.attributes["gen_ai.system"] == "vertex_ai"
    assert span.attributes["gen_ai.operation.name"] == "GenerateContent"

    assert (
        span.attributes["code.function.name"]
        == "google.genai.Models.generate_content"
    )


@pytest.mark.vcr
def test_genai_system_vertex(
    genai_client: Client,
    otel_assertions: OTelAssertions,
    instrument_with_content,
):
    genai_client.models.generate_content(
        model="gemini-2.0-flash", contents="Does this work?"
    )
    span = otel_assertions.get_span_named("generate_content gemini-2.0-flash")
    assert span.attributes["gen_ai.system"] == "vertex_ai"


@pytest.mark.vcr
def test_genai_system_gemini(
    genai_client_gemini: Client,
    otel_assertions: OTelAssertions,
    instrument_with_content,
):
    genai_client_gemini.models.generate_content(
        model="gemini-2.0-flash", contents="Does this work?"
    )
    span = otel_assertions.get_span_named("generate_content gemini-2.0-flash")
    assert span.attributes["gen_ai.system"] == "gemini"


@pytest.mark.vcr
def test_token_counts(
    genai_client: Client,
    otel_assertions: OTelAssertions,
    instrument_with_content,
):
    genai_client.models.generate_content(
        model="gemini-2.0-flash", contents="Does this work?"
    )
    span = otel_assertions.get_span_named("generate_content gemini-2.0-flash")
    assert span.attributes["gen_ai.usage.input_tokens"] == 4
    assert span.attributes["gen_ai.usage.output_tokens"] == 264


def call_with_system_prompt(genai_client: Client, vcr) -> None:
    with vcr.use_cassette("test_system_prompt"):
        genai_client.models.generate_content(
            model="gemini-2.0-flash",
            contents="Does this work?",
            config={"system_instruction": "foo"},
        )


def test_record_system_prompt(
    genai_client: Client,
    otel_assertions: OTelAssertions,
    vcr,
    instrument_with_content,
):
    call_with_system_prompt(genai_client, vcr)
    otel_assertions.assert_has_event_named("gen_ai.system.message")
    event_record = otel_assertions.get_event_named("gen_ai.system.message")
    assert event_record.attributes["gen_ai.system"] == "vertex_ai"
    assert event_record.body["content"] == "foo"


def test_does_not_record_system_prompt_if_disabled(
    genai_client: Client,
    otel_assertions: OTelAssertions,
    vcr,
    instrument_no_content,
):
    call_with_system_prompt(genai_client, vcr)
    # NOTE: this should still emit an event, it should just have content elided
    otel_assertions.assert_does_not_have_event_named("gen_ai.system.message")


def test_record_user_prompt(
    genai_client: Client,
    otel_assertions: OTelAssertions,
    vcr,
    instrument_with_content,
):
    call_with_system_prompt(genai_client, vcr)
    otel_assertions.assert_has_event_named("gen_ai.user.message")
    event_record = otel_assertions.get_event_named("gen_ai.user.message")
    assert event_record.attributes["gen_ai.system"] == "vertex_ai"
    assert event_record.body["content"] == "Does this work?"


def test_does_not_record_user_prompt_if_disabled(
    genai_client: Client,
    otel_assertions: OTelAssertions,
    vcr,
    instrument_no_content,
):
    call_with_system_prompt(genai_client, vcr)
    # NOTE: this should still emit an event, it should just have content elided
    otel_assertions.assert_does_not_have_event_named("gen_ai.user.message")


def test_record_response(
    genai_client: Client,
    otel_assertions: OTelAssertions,
    vcr,
    instrument_with_content,
):
    call_with_system_prompt(genai_client, vcr)
    # NOTE: this should be gen_ai.choice event
    otel_assertions.assert_has_event_named("gen_ai.assistant.message")
    event_record = otel_assertions.get_event_named("gen_ai.assistant.message")
    assert event_record.attributes["gen_ai.system"] == "vertex_ai"
    # NOTE: this should not contain all of this, just the actual content
    assert event_record.body["content"] == {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "video_metadata": None,
                            "thought": None,
                            "code_execution_result": None,
                            "executable_code": None,
                            "file_data": None,
                            "function_call": None,
                            "function_response": None,
                            "inline_data": None,
                            "text": 'To answer whether "this" works, I need to know what "this" refers to. Please provide me with more context. For example:\n\n*   **What is the code or process you\'re referring to?** (e.g., "Does this Python code work?", "Does this button on the website work?")\n*   **Show me the code or describe the process.** (e.g., Paste the code snippet, explain the steps you\'re taking on the website)\n*   **What are you trying to achieve?** (e.g., "I\'m trying to sort a list", "I\'m trying to submit a form")\n\nOnce I have this information, I can help you determine if it works as expected and, if not, help you troubleshoot the issue.\n',
                        }
                    ],
                    "role": "model",
                },
                "citation_metadata": None,
                "finish_message": None,
                "token_count": None,
                "avg_logprobs": -0.3037491986851492,
                "finish_reason": types.FinishReason.STOP,
                "grounding_metadata": None,
                "index": None,
                "logprobs_result": None,
                "safety_ratings": None,
            }
        ],
        "model_version": "gemini-2.0-flash",
        "prompt_feedback": None,
        "usage_metadata": {
            "cached_content_token_count": None,
            "candidates_token_count": 167,
            "prompt_token_count": 5,
            "total_token_count": 172,
        },
        "automatic_function_calling_history": [],
        "parsed": None,
    }


def test_does_not_record_response_if_disabled(
    genai_client: Client,
    otel_assertions: OTelAssertions,
    vcr,
    instrument_no_content,
):
    call_with_system_prompt(genai_client, vcr)
    # NOTE: this should still emit an event, it should just have content elided
    otel_assertions.assert_does_not_have_event_named("gen_ai.user.message")
