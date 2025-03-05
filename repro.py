# /// script
# requires-python = ">=3.9, <3.13"
# dependencies = [
#     "google-cloud-aiplatform >= 1.64",
#     "numpy>=2",
#     "shapely>=2",
# ]
# ///

import vertexai
from vertexai.generative_models import (
    Content,
    GenerativeModel,
    Part,
)


def main() -> None:
    vertexai.init(api_transport="rest")
    model = GenerativeModel("gemini-1.5-flash-002")
    res = list(
        model.generate_content(
            [
                Content(
                    role="invalid_role",
                    parts=[Part.from_text("Say this is a test")],
                )
            ],
            stream=True,
        )
    )
    print(res)


if __name__ == "__main__":
    main()
