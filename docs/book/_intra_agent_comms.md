In the current Winston architecture, agents can communicate and collaborate in several ways. Let me break down the main mechanisms:

1. **Through the System Interface**
   The `System` protocol provides methods for agent-to-agent communication. Here's a simplified example:

```python
async def invoke_conversation(
    self,
    agent_id: str,
    message: str,
    context: dict[str, Any] | None = None,
) -> AsyncIterator[Response]:
    """Invoke a conversation with another agent."""
```

2. **Shared Workspace**
   Agents can communicate indirectly through a shared workspace. This is demonstrated in the `CognitiveAgent` class:

```python
# From examples/ch03/winston_cognitive.py
async def process(self, message: Message) -> AsyncIterator[Response]:
    # Create message with shared workspace ID for sub-agents
    sub_message = Message(
        content=message.content,
        metadata={
            **message.metadata,
            "shared_workspace": self.workspace_path,
        },
    )

    # Route to appropriate agent
    if self.multimodal_agent.can_handle(message):
        agent = self.multimodal_agent
    elif self.reasoning_agent.can_handle(message):
        agent = self.reasoning_agent
    # ...

    # Process with selected agent
    async for response in agent.process(sub_message):
        yield response
```

3. **Direct Service Invocation**
   Here's an example of how you could implement a direct service invocation pattern:

```python
class ServiceAgent(BaseAgent):
    async def provide_service(self, request: Message) -> Response:
        """Provide a specific service to other agents."""
        # Process the request
        response = await self.generate_response(request)
        return response

class ClientAgent(BaseAgent):
    async def process(self, message: Message) -> AsyncIterator[Response]:
        # Get service agent from system
        service_agent = self.system._agents.get("service_agent_id")

        # Prepare service request
        service_request = Message(
            content="Service request content",
            metadata={
                "requesting_agent": self.id,
                "service_type": "specific_service"
            }
        )

        # Invoke service
        result = await service_agent.provide_service(service_request)

        # Process result
        yield Response(content=f"Service result: {result.content}")
```

4. **Event-Based Communication**
   The system also supports event-based communication:

```python
class EventEmittingAgent(BaseAgent):
    async def process(self, message: Message) -> AsyncIterator[Response]:
        # Do some processing

        # Emit event for other agents
        await self.system.emit_event(
            "event_type",
            {"some": "data"}
        )

# In System implementation
async def emit_event(self, event_type: str, data: Any) -> None:
    """Emit event to subscribed agents."""
    subscribers = self._event_subscribers.get(event_type, set())
    for agent_id in subscribers:
        message = Message(
            content=data,
            metadata={
                "pattern": MessagePattern.EVENT,
                "event_type": event_type,
            }
        )
        async for _ in self.route_message(agent_id, message):
            pass
```

To implement a new service-oriented interaction, you could:

1. Define a service interface:

```python
class AnalysisService(Protocol):
    async def analyze(self, data: str) -> Response:
        """Analyze provided data."""
        ...

class AnalysisAgent(BaseAgent, AnalysisService):
    async def analyze(self, data: str) -> Response:
        analysis_message = Message(
            content=data,
            metadata={"type": "analysis_request"}
        )
        return await self.generate_response(analysis_message)

class ConsumerAgent(BaseAgent):
    async def process(self, message: Message) -> AsyncIterator[Response]:
        # Get analysis service
        analysis_agent = self.system._agents.get("analysis_agent_id")

        # Use service
        analysis_result = await analysis_agent.analyze(message.content)

        # Process result
        yield Response(
            content=f"Analysis complete: {analysis_result.content}"
        )
```

This architecture allows for flexible communication patterns while maintaining clear boundaries and responsibilities between agents. The shared workspace provides persistent context, while direct service invocation and event emission enable real-time interaction.
