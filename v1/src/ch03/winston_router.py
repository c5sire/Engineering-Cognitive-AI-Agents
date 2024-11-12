# winston_router.py
"""
ChainLit-based Conversational AI Assistant with Intent
Routing

This module represents a significant evolution from the
earlier implementation, introducing intent
classification and routing capabilities. Key
enhancements and differences include:

1. Intent Classification: Implements a system to
   classify user intents, allowing for more targeted
   and efficient response handling.

2. Specialized Intent Handlers: Introduces dedicated
   handlers for different types of intents (weather,
   task, help, general), enabling more context-specific
   responses.

3. Flexible Intent Routing: Uses a dictionary-based
   routing system to map intents to their respective
   handlers, improving modularity and extensibility.

4. Enhanced Prompt Management: Incorporates additional
   prompt templates for various intents and intent
   classification methods (basic, chain-of-thought,
   few-shot, multi-intent).

5. Generalized Intent Handling: Implements a
   `handle_intent` function that provides a common
   structure for processing different types of intents.

6. Weather Function Integration: Maintains the weather
   function calling capability but integrates it more
   seamlessly into the intent-based structure.

7. Improved Code Organization: Restructures the code to
   accommodate the new intent classification and
   routing system, enhancing readability and
   maintainability.

8. Expanded Configurability: Introduces more
   environment variables and prompt templates, allowing
   for greater customization of the assistant's
   behavior.

While retaining core functionalities like streaming
responses, conversation history maintenance, and
personalized greetings, this updated version provides a
more sophisticated and adaptable framework for handling
user interactions. It represents a shift towards a more
intelligent and context-aware conversational AI system.
"""

import ast
import json
import os
from datetime import datetime
from typing import Any, cast

import chainlit as cl
from dotenv import load_dotenv
from litellm.types.utils import (
  ChatCompletionMessageToolCall,
  Function,
)

from ch03.intent_classifiers import classify_intent
from ch03.llm import (
  LLMParams,
  Message,
  call_llm,
  call_llm_streaming,
)
from ch03.prompt import Prompt, load_prompt

#

_ = load_dotenv()

WINSTON_PROMPT = load_prompt("ch03/persona/winston")
GREETING_PROMPT = load_prompt("ch03/persona/greeting")
FUNCTION_CALL_RESPONSE_PROMPT = load_prompt(
  "ch03/internal/function_call_response"
)
WEATHER_INTENT_PROMPT = load_prompt(
  "ch03/intent/weather"
)
TASK_INTENT_PROMPT = load_prompt("ch03/intent/task")
HELP_INTENT_PROMPT = load_prompt("ch03/intent/help")
GENERAL_INTENT_PROMPT = load_prompt(
  "ch03/intent/general"
)
BASIC_INTENT_PROMPT = load_prompt(
  "ch03/intent/classifier/basic"
)
COT_INTENT_PROMPT = load_prompt(
  "ch03/intent/classifier/cot"
)
FEW_SHOT_INTENT_PROMPT = load_prompt(
  "ch03/intent/classifier/few_shot"
)
MULTI_INTENT_PROMPT = load_prompt(
  "ch03/intent/classifier/multi"
)

USER_NAME = os.getenv("USER_NAME", "User")


# Example dummy function for weather
def get_current_weather(
  location: str,
  unit: str = "fahrenheit",
) -> str:
  """Get the current weather in a given location"""
  weather_info = {
    "location": location,
    "temperature": "72"
    if unit.lower() == "fahrenheit"
    else "22",
    "unit": unit,
    "forecast": ["sunny", "windy"],
  }
  return json.dumps(weather_info)


async def call_llm_and_tool(
  messages: list[Message],
  params: LLMParams | None = None,
  tools: list[dict[str, Any]] | None = None,
  suppress_output: bool = False,
) -> str | Message:
  """
  Call the LLM, optionally update the chat UI with streaming,
  execute a tool if selected, and return the response.

  Parameters
  ----------
  messages : list[Message]
      The conversation history.
  params : LLMParams | None, optional
      The LLM parameters, by default None
  tools : list[dict[str, Any]] | None, optional
      The tools to call, by default None
  suppress_output : bool, optional
      Whether to suppress streaming output, by default False

  Returns
  -------
  Union[str, Message]:
      Either the LLM response or the tool response
  """
  if params is None:
    params = LLMParams()

  if not suppress_output:
    response_generator = call_llm_streaming(
      messages=messages,
      params=params,
      tools=tools,
    )

    ui_msg = None
    function_msg: Message | None = None
    async for chunk in response_generator:
      if chunk["type"] == "token":
        if not ui_msg:
          ui_msg = cl.Message(content="")
          _ = await ui_msg.send()
        await ui_msg.stream_token(chunk["data"])
      elif chunk["type"] == "tool_call" and tools:
        tool_call = cast(
          ChatCompletionMessageToolCall,
          chunk["data"],
        )
        function_msg = await call_tool(
          tool_call.function
        )

    if not function_msg and ui_msg:
      await ui_msg.update()
    return function_msg or (
      ui_msg.content if ui_msg else ""
    )

  response = await call_llm(
    messages=messages,
    params=params,
    tools=tools,
  )

  function_result = (
    await call_tool(response)
    if isinstance(
      response, ChatCompletionMessageToolCall
    )
    else None
  )
  return function_result or str(response)


@cl.step(type="tool")
async def call_tool(
  function: Function,
) -> Message:
  """
  Call the tool function and update the message history with the function

  Parameters
  ----------
  tool_call : FunctionCall
      The function call to make

  Returns
  -------
  Message
      The function message
  """
  function_name = function.name
  if not function_name:
    raise ValueError("Function name is required")

  arguments = ast.literal_eval(function.arguments)

  current_step = cl.context.current_step
  if not current_step:
    raise ValueError("Current step is required")

  current_step.name = function_name
  current_step.input = arguments

  function_response = get_current_weather(
    location=arguments.get("location"),
    unit=arguments.get("unit", "fahrenheit"),
  )

  current_step.output = function_response
  current_step.language = "json"

  return cast(
    Message,
    {
      "role": "function",
      "name": function_name,
      "content": function_response,
    },
  )


@cl.on_chat_start
async def start_chat() -> None:
  """
  Start the chat session by setting the initial
  history
  """
  history = [
    cast(
      Message,
      {
        "role": "system",
        "content": WINSTON_PROMPT.template,
      },
    ),
  ]
  cl.user_session.set("history", history)

  current_time = datetime.now()
  time_of_day = (
    "morning"
    if 0 <= current_time.hour < 12
    else "afternoon"
    if 12 <= current_time.hour < 18
    else "evening"
  )

  greeting_messages = history.copy()
  greeting_messages.append(
    cast(
      Message,
      {
        "role": "user",
        "content": GREETING_PROMPT.render(
          user_name=USER_NAME,
          time_of_day=time_of_day,
        ),
      },
    ),
  )

  greeting_response = await call_llm_and_tool(
    messages=greeting_messages,
    params=GREETING_PROMPT.params,
  )
  assert isinstance(greeting_response, str)
  history.append(
    cast(
      Message,
      {
        "role": "assistant",
        "content": greeting_response,
      },
    ),
  )
  cl.user_session.set("history", history)


@cl.on_message
async def handle_message(message: cl.Message) -> None:
  """
  Receive a message from the user, classify intent,
  and route to appropriate handler
  """
  history: list[Message] = cl.user_session.get(
    "history"
  )  # type: ignore
  assert isinstance(history, list)
  history.append(
    cast(
      Message,
      {
        "role": "user",
        "content": message.content,
      },
    ),
  )

  intent = (
    await classify_intent(
      messages=history,
      prompt=MULTI_INTENT_PROMPT,
    )
  )[0]

  intent_handlers = {
    "weather": handle_weather_intent,
    "task": handle_task_intent,
    "help": handle_help_intent,
    "general": handle_general_intent,
  }

  handler = intent_handlers.get(
    intent, handle_general_intent
  )
  new_messsages = await handler(history)
  history.extend(new_messsages)


async def handle_intent(
  messages: list[Message],
  prompt: Prompt,
  prompt_vars: dict[str, Any] | None = None,
  tools: list[dict[str, Any]] | None = None,
  suppress_output: bool = False,
) -> list[Message]:
  """Generalized handler for intents"""

  # If no custom prompt variables are provided, use
  # the user_message
  if prompt_vars is None:
    user_message = messages[-1]["content"]
    prompt_vars = {"user_message": user_message}

  # Render the prompt with the provided variables
  response_prompt = prompt.render(**prompt_vars)

  # Generate a new list of messages, dropping the last
  # (actual) user message since it's already been
  # incorporated into the prompt and add the prompt
  # into the list
  tmp_messages = messages[:-1] + [
    cast(
      Message,
      {
        "role": "user",
        "content": response_prompt,
      },
    )
  ]

  # Call the LLM and execute a tool if selected
  response = await call_llm_and_tool(
    messages=tmp_messages,
    params=prompt.params,
    tools=tools,
    suppress_output=suppress_output,
  )

  # If a tool call was executed, update the chat
  # history with a prose response
  if isinstance(response, dict):
    # Create some temporary messages to provide
    # context to the LLM
    tmp_messages = tmp_messages + [
      response,
      cast(
        Message,
        {
          "role": "user",
          "content": FUNCTION_CALL_RESPONSE_PROMPT.template,
        },
      ),
    ]
    response = await call_llm_and_tool(
      messages=messages[:-1] + tmp_messages,
      params=FUNCTION_CALL_RESPONSE_PROMPT.params,
      suppress_output=suppress_output,
    )

  return [
    cast(
      Message,
      {
        "role": "assistant",
        "content": str(response),
      },
    )
  ]


async def handle_weather_intent(
  messages: list[Message],
  suppress_output: bool = False,
) -> list[Message]:
  """Handle the weather intent"""
  tools = [
    {
      "type": "function",  # Add this line
      "function": {  # Wrap the existing content in a 'function' key
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {
              "type": "string",
              "description": "The city and state, e.g. San Francisco, CA",
            },
            "unit": {
              "type": "string",
              "enum": ["celsius", "fahrenheit"],
            },
          },
          "required": ["location"],
        },
      },
    }
  ]

  return await handle_intent(
    messages=messages,
    prompt=WEATHER_INTENT_PROMPT,
    tools=tools,
    suppress_output=suppress_output,
  )


async def handle_task_intent(
  messages: list[Message],
  suppress_output: bool = False,
) -> list[Message]:
  """Handle the task intent"""
  return await handle_intent(
    messages,
    TASK_INTENT_PROMPT,
    suppress_output=suppress_output,
  )


async def handle_help_intent(
  messages: list[Message],
  suppress_output: bool = False,
) -> list[Message]:
  """Handle the help intent"""
  return await handle_intent(
    messages,
    HELP_INTENT_PROMPT,
    suppress_output=suppress_output,
  )


async def handle_general_intent(
  messages: list[Message],
  suppress_output: bool = False,
) -> list[Message]:
  """Handle the general intent"""
  return await handle_intent(
    messages,
    GENERAL_INTENT_PROMPT,
    suppress_output=suppress_output,
  )
