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


from opentelemetry.sdk._logs.export import (
    InMemoryLogExporter,
)
from opentelemetry.sdk.metrics.export import (
    InMemoryMetricReader,
)
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)


class _LogWrapper:
    def __init__(self, log_data):
        self._log_data = log_data

    @property
    def scope(self):
        return self._log_data.instrumentation_scope

    @property
    def resource(self):
        return self._log_data.log_record.resource

    @property
    def attributes(self):
        return self._log_data.log_record.attributes

    @property
    def body(self):
        return self._log_data.log_record.body

    def __str__(self):
        return self._log_data.log_record.to_json()


class _MetricDataPointWrapper:
    def __init__(self, resource, scope, metric):
        self._resource = resource
        self._scope = scope
        self._metric = metric

    @property
    def resource(self):
        return self._resource

    @property
    def scope(self):
        return self._scope

    @property
    def metric(self):
        return self._metric

    @property
    def name(self):
        return self._metric.name

    @property
    def data(self):
        return self._metric.data


class OTelAssertions:
    def __init__(
        self,
        logs_exporter: InMemoryLogExporter,
        span_exporter: InMemorySpanExporter,
        metric_reader: InMemoryMetricReader,
    ):
        self._logs = logs_exporter
        self._traces = span_exporter
        self._metrics = metric_reader
        self._spans = []
        self._finished_logs = []
        self._metrics_data = []

    def get_finished_logs(self):
        for log_data in self._logs.get_finished_logs():
            self._finished_logs.append(_LogWrapper(log_data))
        return self._finished_logs

    def get_finished_spans(self):
        for span in self._traces.get_finished_spans():
            self._spans.append(span)
        return self._spans

    def get_metrics_data(self):
        data = self._metrics.get_metrics_data()
        if data is not None:
            for resource_metric in data.resource_metrics:
                resource = resource_metric.resource
                for scope_metrics in resource_metric.scope_metrics:
                    scope = scope_metrics.scope
                    for metric in scope_metrics.metrics:
                        wrapper = _MetricDataPointWrapper(
                            resource, scope, metric
                        )
                        self._metrics_data.append(wrapper)
        return self._metrics_data

    def get_span_named(self, name):
        for span in self.get_finished_spans():
            if span.name == name:
                return span
        return None

    def assert_has_span_named(self, name):
        span = self.get_span_named(name)
        finished_spans = [span.name for span in self.get_finished_spans()]
        assert span is not None, (
            f'Could not find span named "{name}"; finished spans: {finished_spans}'
        )

    def get_event_named(self, event_name):
        for event in self.get_finished_logs():
            event_name_attr = event.attributes.get("event.name")
            if event_name_attr is None:
                continue
            if event_name_attr == event_name:
                return event
        return None

    def assert_has_event_named(self, name):
        event = self.get_event_named(name)
        finished_logs = self.get_finished_logs()
        assert event is not None, (
            f'Could not find event named "{name}"; finished logs: {finished_logs}'
        )

    def assert_does_not_have_event_named(self, name):
        event = self.get_event_named(name)
        assert event is None, f"Unexpected event: {event}"

    def get_metrics_data_named(self, name):
        results = []
        for entry in self.get_metrics_data():
            if entry.name == name:
                results.append(entry)
        return results

    def assert_has_metrics_data_named(self, name):
        data = self.get_metrics_data_named(name)
        assert len(data) > 0
