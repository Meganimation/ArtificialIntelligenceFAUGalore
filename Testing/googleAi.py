from google import genai
from google.genai import types
GIT P
client = genai.Client(vertexai=True, project="newprj-490404", location="us-central1")

# 2. Define your "Academic" Safety Settings
# This is how to prevent the model from 'over-refusing' research
academic_safety = [
    types.SafetySetting(
        category="HARM_CATEGORY_DANGEROUS_CONTENT",
        threshold="BLOCK_NONE", # This allows technical/medical data
    ),
    types.SafetySetting(
        category="HARM_CATEGORY_HARASSMENT",
        threshold="BLOCK_NONE",
    )
]

# 3. Generate Factual Research Content
response = client.models.generate_content(
    model="gemini-2.5-flash", # Use 2.5 instead of 2.0
    contents="Analyze the neurological impact of localized glutamate excitotoxicity in the prefrontal cortex.",
    config=types.GenerateContentConfig(
        safety_settings=academic_safety,
        temperature=0.1, # Low temperature = Higher factual consistency
    )
)

print(response.text)