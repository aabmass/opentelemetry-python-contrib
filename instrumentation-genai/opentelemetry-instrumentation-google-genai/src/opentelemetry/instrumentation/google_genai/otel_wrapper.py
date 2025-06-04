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

import google.genai

from opentelemetry._events import Event
from opentelemetry.instrumentation.google_genai.uploader import (
    upload_to_storage,
)
from opentelemetry.util.types import _ExtendedAttributes, AnyValue
from opentelemetry.semconv._incubating.metrics import gen_ai_metrics
from opentelemetry.semconv.schemas import Schemas

from .version import __version__ as _LIBRARY_VERSION
from .message_models import InputMessages, OutputMessages, SystemMessage

_logger = logging.getLogger(__name__)

_SCOPE_NAME = "opentelemetry.instrumentation.google_genai"
_PYPI_PACKAGE_NAME = "opentelemetry-instrumentation-google-genai"
_SCHEMA_URL = Schemas.V1_30_0.value
_SCOPE_ATTRIBUTES = {
    "gcp.client.name": "google.genai",
    "gcp.client.repo": "googleapis/python-genai",
    "gcp.client.version": google.genai.__version__,
    "pypi.package.name": _PYPI_PACKAGE_NAME,
}


class OTelWrapper:
    def __init__(self, tracer, event_logger, meter):
        self._tracer = tracer
        self._event_logger = event_logger
        self._meter = meter
        self._operation_duration_metric = (
            gen_ai_metrics.create_gen_ai_client_operation_duration(meter)
        )
        self._token_usage_metric = (
            gen_ai_metrics.create_gen_ai_client_token_usage(meter)
        )

    @staticmethod
    def from_providers(tracer_provider, event_logger_provider, meter_provider):
        return OTelWrapper(
            tracer_provider.get_tracer(
                _SCOPE_NAME, _LIBRARY_VERSION, _SCHEMA_URL, _SCOPE_ATTRIBUTES
            ),
            event_logger_provider.get_event_logger(
                _SCOPE_NAME, _LIBRARY_VERSION, _SCHEMA_URL, _SCOPE_ATTRIBUTES
            ),
            meter=meter_provider.get_meter(
                _SCOPE_NAME, _LIBRARY_VERSION, _SCHEMA_URL, _SCOPE_ATTRIBUTES
            ),
        )

    def start_as_current_span(self, *args, **kwargs):
        return self._tracer.start_as_current_span(*args, **kwargs)

    @property
    def operation_duration_metric(self):
        return self._operation_duration_metric

    @property
    def token_usage_metric(self):
        return self._token_usage_metric

    def log_system_prompt(self, attributes, body):
        _logger.debug("Recording system prompt.")
        event_name = "gen_ai.system.message"
        self._log_event(event_name, attributes, body)

    def log_user_prompt(self, attributes, body):
        _logger.debug("Recording user prompt.")
        event_name = "gen_ai.user.message"
        self._log_event(event_name, attributes, body)

    def log_completion_details(
        self,
        *,
        attributes: _ExtendedAttributes,
        system_instructions: SystemMessage | None,
        input_messages: InputMessages,
        output_messages: OutputMessages,
        # request_model: str,
        # response_model: str,
        # input_tokens: int,
        # output_tokens: int,
    ) -> None:
        _logger.debug("Recording completion details event.")
        event_name = "gen_ai.completion.details"

        body = {
            "gen_ai.input.messages": input_messages.model_dump(mode="json"),
            "gen_ai.output.messages": output_messages.model_dump(mode="json"),
        }
        if system_instructions:
            body["gen_ai.system.instructions"] = (
                system_instructions.model_dump(mode="json")
            )

        event = Event(event_name, body=body, attributes=attributes)
        self._event_logger.emit(event)

    def log_completion_details_refs(
        self,
        *,
        attributes: _ExtendedAttributes,
        system_instructions: SystemMessage | None,
        input_messages: InputMessages,
        output_messages: OutputMessages,
        response_id: str,
        # request_model: str,
        # response_model: str,
        # input_tokens: int,
        # output_tokens: int,
    ) -> None:
        _logger.debug("Recording completion details event as ref.")
        event_name = "gen_ai.completion.details"

        input_messages_ref = upload_to_storage(
            f"{response_id}_input.json",
            input_messages.model_dump(mode="json"),
        )
        output_messages_ref = upload_to_storage(
            f"{response_id}_output.json",
            output_messages.model_dump(mode="json"),
        )
        body = {
            "gen_ai.input.messages_ref": input_messages_ref,
            "gen_ai.output.messages_ref": output_messages_ref,
        }
        if system_instructions:
            system_instructions_ref = upload_to_storage(
                f"{response_id}_system_instructions.json",
                system_instructions.model_dump(mode="json"),
            )
            body["gen_ai.system.instructions_ref"] = system_instructions_ref

        event = Event(event_name, body=body, attributes=attributes)
        self._event_logger.emit(event)

    def log_response_content(self, attributes, body):
        _logger.debug("Recording response.")
        event_name = "gen_ai.choice"
        self._log_event(event_name, attributes, body)

    def _log_event(self, event_name, attributes, body):
        event = Event(event_name, body=body, attributes=attributes)
        self._event_logger.emit(event)
