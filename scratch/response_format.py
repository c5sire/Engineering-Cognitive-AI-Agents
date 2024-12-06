from litellm import completion
from pydantic import BaseModel


# Define your Pydantic model
class CalendarEvent(BaseModel):
  name: str
  date: str
  participants: list[str]


# Use LiteLLM to call the model with the response format using the workaround
response = completion(
  model="gpt-4o",
  messages=[
    {
      "role": "system",
      "content": "Extract the event information.",
    },
    {
      "role": "user",
      "content": "Alice and Bob are going to a science fair on Friday.",
    },
  ],
  response_format=CalendarEvent,
)

calendar_event = CalendarEvent.model_validate_json(
  response.choices[0].message.content
)
print(calendar_event)
