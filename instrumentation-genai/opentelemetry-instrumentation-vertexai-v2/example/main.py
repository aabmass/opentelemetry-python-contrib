import vertexai
from vertexai.generative_models import GenerativeModel

PROJECT_ID = "otel-starter-project"
vertexai.init(project=PROJECT_ID, location="us-central1")

model = GenerativeModel("gemini-1.5-flash-002")

response = model.generate_content("Write a short poem on OpenTelemetry.")

print(response.text)
# Example response:
# **Emphasizing the Dried Aspect:**
# * Everlasting Blooms
# * Dried & Delightful
# * The Petal Preserve
# ...
