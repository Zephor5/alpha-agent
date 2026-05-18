"""Prompt builder for explicit memory context."""

from __future__ import annotations

from alpha_agent.llm.base import ChatMessage
from alpha_agent.memory.models import RetrievedContext


class PromptBuilder:
    """Build transparent OpenAI-style chat prompts from memory context."""

    system_prompt = """Identity: Alpha Agent.

Behavior rules:
- Be concise but useful.
- Use memory context when relevant, but do not overfit to it.
- Do not claim uncertain memories as certain.
- Prefer asking clarifying questions only when necessary.
- Keep the runtime understandable and avoid hidden agent behavior."""

    def build(self, user_message: str, context: RetrievedContext) -> list[ChatMessage]:
        """Build messages compatible with chat completions APIs."""

        user_content = "\n\n".join(
            [
                self._working_memory_section(context),
                self._semantic_section(context),
                self._episodic_section(context),
                self._procedural_section(context),
                "## Current User Message\n" + user_message,
            ]
        )
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]

    def rough_token_estimate(self, messages: list[ChatMessage]) -> int:
        """Estimate prompt tokens with a simple character-based approximation."""

        return sum(len(message["content"]) for message in messages) // 4

    def _working_memory_section(self, context: RetrievedContext) -> str:
        if not context.working_memory:
            return "## Working Memory\n- None"
        lines = [f"- {item.content}" for item in context.working_memory]
        return "## Working Memory\n" + "\n".join(lines)

    def _semantic_section(self, context: RetrievedContext) -> str:
        if not context.semantic_memories:
            return "## Relevant User Facts\n- None"
        lines = [
            f"- ({memory.confidence:.2f}) {memory.content}" for memory in context.semantic_memories
        ]
        return "## Relevant User Facts\n" + "\n".join(lines)

    def _episodic_section(self, context: RetrievedContext) -> str:
        if not context.episodic_memories:
            return "## Relevant Episodes\n- None"
        lines = [
            f"- ({memory.salience:.2f}) {memory.summary}" for memory in context.episodic_memories
        ]
        return "## Relevant Episodes\n" + "\n".join(lines)

    def _procedural_section(self, context: RetrievedContext) -> str:
        if not context.procedural_memories:
            return "## Relevant Skills\n- None"
        lines = [
            f"- {memory.name}: {memory.description}\n{memory.procedure_markdown}"
            for memory in context.procedural_memories
        ]
        return "## Relevant Skills\n" + "\n".join(lines)
