import google.auth
import google.auth.transport.grpc
import google.auth.transport.requests
import grpc
from google.auth.transport.grpc import AuthMetadataPlugin

from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
    OTLPSpanExporter,
)


class OTLPSpanExporterGoogleAuth(OTLPSpanExporter):
    """Small OTLP span exporter shim to add google credentials so tokens refresh"""

    def __init__(self) -> None:
        credentials, project_id = google.auth.default()
        request = google.auth.transport.requests.Request()
        auth_metadata_plugin = AuthMetadataPlugin(
            credentials=credentials, request=request
        )
        channel_creds = grpc.composite_channel_credentials(
            grpc.ssl_channel_credentials(),
            grpc.metadata_call_credentials(auth_metadata_plugin),
        )

        super().__init__(credentials=channel_creds)
        print("Created OTLP wrapped exporter", self)
