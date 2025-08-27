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

"""Generic uploading module which could be part of a GenAI utils

Has a global uploader object using `fsspec` under the hood to work with all sorts of local and
cloud storage.
"""

import contextlib
import io
import json
import logging
import os
import posixpath
from abc import ABC, abstractmethod
from pathlib import Path

import fsspec
from google.cloud import storage

from opentelemetry.instrumentation.utils import suppress_instrumentation
from opentelemetry.trace import get_tracer
from opentelemetry.util._once import Once
from opentelemetry.util.types import AnyValue
from opentelemetry.instrumentation.utils import suppress_instrumentation

OTEL_PYTHON_GENAI_UPLOADER_PATH = "OTEL_PYTHON_GENAI_UPLOADER_PATH"
"""
.. envvar:: OTEL_PYTHON_GENAI_UPLOADER_PATH

An `fsspec.open` compatible URI/path for uploading prompts and responses. Can be a local path
like ``file:./prompts`` or a cloud storage URI such as ``gcs://my_bucket``. See
`fsspec <https://filesystem-spec.readthedocs.io/en/latest/index.html>`_.
"""

_logger = logging.getLogger(__name__)
_tracer = get_tracer(__name__)


class Uploader(ABC):
    @abstractmethod
    def upload(self, *, path: str, value: AnyValue) -> str:
        pass


# Another impl using GCS client directly. I tested this and it was not any faster but could use
# more benchmarking vs fsspec
class GcsUploader(Uploader):
    def __init__(
        self, *, bucket_path: str, client: storage.Client | None = None
    ) -> None:
        self._client = client or storage.Client()
        self._bucket_path = bucket_path

    def upload(self, *, path: str, value: AnyValue) -> str:
        blob = self._client.bucket(self._bucket_path).blob(str(path))
        with blob.open("w") as file:
            json.dump(value, file)
        _logger.debug("GcsUploader uploaded to %s", file)
        return f"gs://{blob.bucket.name}/{blob.name}"


class FsSpecUploader(Uploader):
    def __init__(self, *, base_path: str) -> None:
        self._base_path = base_path

    # @_tracer.start_as_current_span("fsspec_upload")
    def upload(self, *, path: str, value: AnyValue) -> str:
        open_file = fsspec.open(posixpath.join(self._base_path, path), "w")
        with open_file as file:
            json.dump(value, file)

        _logger.debug("Uploaded to %s", open_file)
        return open_file.full_name


_uploader: Uploader | None = None
_uploader_once: Once = Once()


def _get_uploader() -> Uploader | None:
    if _uploader:
        return _uploader

    if uploader_path := os.environ.get(OTEL_PYTHON_GENAI_UPLOADER_PATH):
        try:
            set_uploader(FsSpecUploader(base_path=uploader_path))
        except ValueError:
            _logger.exception(
                "Failed to create uploader from %s=%s",
                OTEL_PYTHON_GENAI_UPLOADER_PATH,
                uploader_path,
            )

    return _uploader


def set_uploader(uploader: Uploader) -> None:
    def do_set():
        global _uploader
        _uploader = uploader
        _logger.debug("Set global uploader %s", uploader)

    _uploader_once.do_once(do_set)


def upload_to_storage(filename: str, value: AnyValue) -> str:
    """Given an AnyValue, uploads it to storage (local filesystem, GCS, S3, etc.)"""
    # TODO: remove this, just suppressing instrumentation to avoid noise in the demo
    with suppress_instrumentation():
        if not (uploader := _get_uploader()):
            return "/dev/null"
        return uploader.upload(path=Path(filename), value=value)
