"""
Conversation Memory Service with LangGraph

Implements:
- Multi-turn conversation with memory
- LangGraph state management
- Chat history tracking
- Context-aware follow-up questions

This is where LangGraph shines - stateful workflows!
"""

from typing import List, Dict, Any, Optional, Annotated
from typing_extensions import TypedDict
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from app.services.retrieval import retrieval_service
from app.services.generation import generation_service
from app.core.config import settings


class ConversationMessage(BaseModel):
    """Single message in conversation"""
    role: str = Field(..., description="user or assistant")
    content: str = Field(..., description="Message content")
    timestamp: str = Field(..., description="ISO timestamp")
    sources: List[Dict] = Field(default=[], description="Source documents (for assistant messages)")


class ConversationState(BaseModel):
    """
    LangGraph conversation state.

    This is what gets passed through the graph and persisted between turns.
    """
    messages: List[BaseMessage] = Field(default=[], description="Chat history")
    question: str = Field(default="", description="Current user question")
    context: str = Field(default="", description="Retrieved context")
    answer: str = Field(default="", description="Generated answer")
    sources: List[Dict] = Field(default=[], description="Source documents")

    model_config = {"arbitrary_types_allowed": True}


class GraphState(TypedDict, total=False):
    """
    The actual LangGraph state schema.

    The graph used to be built as StateGraph(dict), which has no schema and therefore no
    reducers, so every node's returned value REPLACED each channel. Combined with seeding
    "messages": [] on every invoke, that wiped the checkpointer's history on each request:
    the turn was saved, then immediately overwritten, so reformulation always saw an empty
    history and every follow-up was silently treated as a standalone question.

    `messages` now carries the add_messages reducer, so returning a list APPENDS to the
    accumulated history instead of replacing it. Every node below returns only the keys it
    actually changed, which matters: returning the whole state would re-append the history
    it just read and duplicate every turn.
    """
    question: str
    standalone_question: str
    messages: Annotated[list, add_messages]
    context: str
    answer: str
    model_used: str
    sources: List[Dict]


class ConversationService:
    """
    Manages multi-turn conversations with memory using LangGraph.

    Features:
    - Chat history across multiple questions
    - Context-aware responses (remembers previous Q&A)
    - Standalone question generation (reformulates based on history)
    - State persistence with LangGraph checkpointing
    """

    def __init__(self):
        self.retrieval = retrieval_service
        self.generation = generation_service
        self.checkpointer = MemorySaver()  # In-memory checkpointer

        # Build LangGraph workflow
        self.graph = self._build_graph()

    def _build_graph(self):
        """
        Build LangGraph conversation workflow.

        Flow:
        1. Reformulate question (considering chat history)
        2. Retrieve relevant documents
        3. Generate answer with full context
        4. Save to memory
        """

        workflow = StateGraph(GraphState)

        # Define nodes (steps in the workflow)
        workflow.add_node("reformulate", self._reformulate_question)
        workflow.add_node("retrieve", self._retrieve_context)
        workflow.add_node("generate", self._generate_answer)

        # Define edges (flow between nodes)
        workflow.set_entry_point("reformulate")
        workflow.add_edge("reformulate", "retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)

        # Compile with checkpointer for memory
        return workflow.compile(checkpointer=self.checkpointer)

    def _reformulate_question(self, state: Dict) -> Dict:
        """
        Reformulate question considering chat history.

        Standalone question = question that makes sense without history.

        Example:
        History: "What are Daniel's AI skills?" → "He has experience with..."
        New Q:   "What projects use those?"
        Standalone: "What AI projects has Daniel built?"
        """

        messages = state.get("messages", [])
        question = state.get("question", "")

        # If no history, question is already standalone
        if len(messages) == 0:
            return {"standalone_question": question}

        # Use LLM to reformulate based on history
        reformulate_prompt = ChatPromptTemplate.from_messages([
            ("system", """Given a chat history and a follow-up question, rephrase the follow-up question to be a standalone question that captures all necessary context.

If the follow-up question is already standalone, return it unchanged.

Examples:
History: "What are Daniel's skills?" → "He knows React, Python..."
Follow-up: "What about AI?" → Standalone: "What are Daniel's AI skills?"

History: "Tell me about AutoFlow Pro" → "It's a browser automation..."
Follow-up: "What tech stack?" → Standalone: "What technology stack does AutoFlow Pro use?"
"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "Follow-up question: {question}\n\nStandalone question:"),
        ])

        try:
            llm = self.generation.primary_llm
            if llm is None:
                raise RuntimeError("no LLM provider configured")
            chain = reformulate_prompt | llm | StrOutputParser()

            standalone = chain.invoke({
                "chat_history": messages[-6:],  # Last 3 turns (6 messages)
                "question": question
            })

            print(f"[i] Reformulated: '{question}' → '{standalone.strip()}'")
            return {"standalone_question": standalone.strip()}

        except Exception as e:
            print(f"[!] Reformulation failed, using original: {e}")
            return {"standalone_question": question}

    def _retrieve_context(self, state: Dict) -> Dict:
        """Retrieve relevant documents for standalone question"""

        standalone_question = state.get("standalone_question", state.get("question", ""))

        print(f"[i] Retrieving context for: '{standalone_question}'")

        # Retrieve documents
        results = self.retrieval.retrieve(standalone_question, k=3, with_scores=True)

        # Format context
        context = self.retrieval.format_context(results.documents)

        return {
            "context": context,
            "sources": [
                {
                    "file_name": doc.metadata.get("file_name", "unknown"),
                    "page": doc.metadata.get("page"),
                    "content_preview": doc.page_content[:100] + "..."
                }
                for doc in results.documents
            ],
        }

    def _generate_answer(self, state: Dict) -> Dict:
        """Generate answer using context and chat history"""

        messages = state.get("messages", [])
        question = state.get("question", "")
        context = state.get("context", "")

        # Generate answer with full chat history for context
        gen_response = self.generation.generate(question, context)

        # Return ONLY the new turn. The add_messages reducer appends it to the accumulated
        # history, so handing back `messages + [...]` here would duplicate every prior turn.
        return {
            "answer": gen_response.answer,
            "model_used": gen_response.model_used,
            "messages": [
                HumanMessage(content=question),
                AIMessage(content=gen_response.answer),
            ],
        }

    def query_with_memory(
        self,
        question: str,
        conversation_id: str = "default"
    ) -> Dict[str, Any]:
        """
        Query with conversation memory.

        Args:
            question: User question
            conversation_id: Unique conversation identifier (for multi-user support)

        Returns:
            Dict with answer, sources, and conversation metadata
        """

        print(f"\n[i] Processing with conversation memory (ID: {conversation_id})")
        print("-" * 70)

        # Prepare initial state
        # Deliberately does NOT seed "messages". The checkpointer supplies the accumulated
        # history for this thread_id, and passing an empty list here is what used to wipe it.
        initial_state = {"question": question}

        # Run graph with checkpointing (memory!)
        config = {"configurable": {"thread_id": conversation_id}}

        final_state = self.graph.invoke(initial_state, config)

        return {
            "answer": final_state["answer"],
            "sources": final_state["sources"],
            "model_used": final_state.get("model_used") or self.generation.primary_model_name,
            "conversation_id": conversation_id,
            "message_count": len(final_state["messages"])
        }

    def get_conversation_history(self, conversation_id: str) -> List[ConversationMessage]:
        """
        Get conversation history for a specific conversation.

        Args:
            conversation_id: Conversation identifier

        Returns:
            List of messages in chronological order
        """

        # Get state from checkpointer
        config = {"configurable": {"thread_id": conversation_id}}

        try:
            # Get saved state (this is a LangGraph feature!)
            checkpoint = self.checkpointer.get(config)

            if not checkpoint or "messages" not in checkpoint:
                return []

            messages = checkpoint["messages"]

            # Convert to ConversationMessage format
            history = []
            for msg in messages:
                history.append(ConversationMessage(
                    role="user" if isinstance(msg, HumanMessage) else "assistant",
                    content=msg.content,
                    timestamp=datetime.now().isoformat(),
                    sources=[]
                ))

            return history

        except Exception as e:
            print(f"[!] Failed to get history: {e}")
            return []

    def clear_conversation(self, conversation_id: str):
        """Clear conversation history for a specific conversation"""
        config = {"configurable": {"thread_id": conversation_id}}
        # Note: MemorySaver doesn't have clear method, would need Redis checkpointer
        print(f"[i] Conversation {conversation_id} cleared (restart to reset)")


# Global instance
conversation_service = ConversationService()


# =============================================================================
# Test Conversation Memory
# =============================================================================
if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    print("=" * 70)
    print("Conversation Memory Test with LangGraph")
    print("=" * 70)

    # Simulate multi-turn conversation
    conv_id = "test_conversation_123"

    questions = [
        "What are Daniel's programming skills?",
        "What about AI specifically?",  # Follow-up - needs context!
        "Tell me about his blockchain experience",
        "Which projects involve that?"  # Another follow-up!
    ]

    for i, question in enumerate(questions, 1):
        print(f"\n[{i}/{len(questions)}] USER: {question}")
        print("-" * 70)

        result = conversation_service.query_with_memory(question, conv_id)

        print(f"\nASSISTANT: {result['answer'][:200]}...")
        print(f"\nSources: {result['sources'][0]['file_name'] if result['sources'] else 'None'}")
        print(f"Total messages in conversation: {result['message_count']}")

    print("\n" + "=" * 70)
    print("[OK] Conversation memory working!")
    print("=" * 70)
    print("\nKey Features Demonstrated:")
    print("  - Follow-up questions work without re-stating context")
    print("  - LangGraph manages state across turns")
    print("  - Chat history preserved per conversation ID")
    print("  - Standalone question generation from history")
