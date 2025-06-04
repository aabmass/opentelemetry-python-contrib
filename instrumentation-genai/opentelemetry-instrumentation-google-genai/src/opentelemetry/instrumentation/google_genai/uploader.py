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

from abc import ABC, abstractmethod
import io
import json
import logging
from pathlib import Path

from fsspec import AbstractFileSystem
from fsspec.asyn import AsyncFileSystem, AbstractBufferedFile

from opentelemetry.trace import get_tracer
from opentelemetry.util._once import Once
from opentelemetry.util.types import AnyValue
from google.cloud import storage
from google.cloud.storage import transfer_manager


_logger = logging.getLogger(__name__)
_tracer = get_tracer(__name__)


# TODO: try to make async or do background buffering + async


class Uploader(ABC):
    @abstractmethod
    def upload(self, *, path: Path, value: AnyValue) -> str:
        pass


class GcsUploader(Uploader):
    def __init__(
        self, *, bucket_path: str, client: storage.Client | None = None
    ) -> None:
        self._client = client or storage.Client()
        self._bucket_path = bucket_path

    def upload(self, *, path: Path, value: AnyValue) -> str:
        blob = self._client.bucket(self._bucket_path).blob(str(path))
        with blob.open("w") as file:
            json.dump(value, file)
        _logger.debug("GcsUploader uploaded to %s", file)
        return f"gs://{blob.bucket.name}/{blob.name}"


class FsSpecUploader(Uploader):
    def __init__(self, *, fs: AbstractFileSystem, base_path: Path) -> None:
        self._fs = fs
        self._base_path = base_path

    @_tracer.start_as_current_span("fsspec_upload")
    def upload(self, *, path: Path, value: AnyValue) -> str:
        with self._fs.open(self._base_path / path, "wb") as file:
            # Encode to utf8 bytes while writing to the file
            text_wrapper = io.TextIOWrapper(
                file, encoding="utf-8", write_through=True
            )
            json.dump(value, text_wrapper)
        _logger.debug("Uploaded to %s", file)
        return file.full_name


_uploader: Uploader | None = None
_uploader_once: Once = Once()


def _get_uploader() -> Uploader | None:
    return _uploader


def set_uploader(uploader: Uploader) -> None:
    def do_set():
        global _uploader
        _uploader = uploader

    _uploader_once.do_once(do_set)


def upload_to_storage(filename: str, value: AnyValue) -> str:
    """Given an AnyValue, uploads it to storage (local filesystem, GCS, S3, etc.)"""
    if not (uploader := _get_uploader()):
        return "/dev/null"
    return uploader.upload(path=Path(filename), value=value)
