"""
Exercise 1: LangChain Basics
Learn: Basic LLM invocation, prompt templates, simple chains

Complete the TODOs below to practice LangChain fundamentals
"""

import os
import sys
from dotenv import load_dotenv

# Add parent directory to path to import from test_setup
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

load_dotenv()

from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

print("=" * 60)
print("Exercise 1: LangChain Basics")
print("=" * 60)

# Initialize LLM
llm = OllamaLLM(model="llama3")

# Part 1: Simple Invocation
print("\n📝 Part 1: Simple LLM Invocation")
print("-" * 60)

# TODO: Ask the LLM to explain RAG in one sentence
# Hint: Use llm.invoke("your question here")
response = llm.invoke("Explain RAG in one sentence")
print(f"Response: {response}\n")


# Part 2: Prompt Templates
print("\n📝 Part 2: Prompt Templates")
print("-" * 60)

# TODO: Create a prompt template with two variables: {role} and {topic}
# Example template: "You are a {role}. Explain {topic} in simple terms."
template = ChatPromptTemplate.from_template(
    "You are a {role}. Explain {topic} in simple terms."
)

# TODO: Create a chain: template | llm | StrOutputParser()
chain = template | llm | StrOutputParser()

# TODO: Invoke the chain with role="teacher" and topic="vector databases"
result = chain.invoke({"role": "teacher", "topic": "vector databases"})
print(f"Response: {result}\n")


# Part 3: Experiment!
print("\n📝 Part 3: Your Turn - Experiment!")
print("-" * 60)

# TODO: Create your own prompt template for a code explainer
# Variables: {programming_language}, {code_snippet}
# Example: "Explain this {programming_language} code: {code_snippet}"

code_template = ChatPromptTemplate.from_template(
    "Explain this {programming_language} code in simple terms:\n\n{code_snippet}\n\nExplanation:"
)

code_chain = code_template | llm | StrOutputParser()

sample_code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""

explanation = code_chain.invoke({
    "programming_language": "Python",
    "code_snippet": sample_code
})

print(f"Code Explanation:\n{explanation}\n")


# Challenge: Multi-step Chain
print("\n🎯 Challenge: Multi-Step Chain")
print("-" * 60)

# TODO: Create a two-step chain:
# Step 1: Generate 3 project ideas about AI automation
# Step 2: Pick the most interesting one and explain why

# Hint: You'll need two templates and chain them together
step1_template = ChatPromptTemplate.from_template(
    "Generate exactly 3 project ideas about {topic}. Number them 1, 2, 3."
)

step2_template = ChatPromptTemplate.from_template(
    "From these project ideas:\n{ideas}\n\nPick the most interesting and explain why in 2 sentences."
)

# Build the chain
multi_step_chain = (
    {"ideas": step1_template | llm}
    | step2_template
    | llm
    | StrOutputParser()
)

result = multi_step_chain.invoke({"topic": "AI automation for developers"})
print(f"Best Project Idea:\n{result}\n")


print("=" * 60)
print("✅ Exercise 1 Complete!")
print("=" * 60)
print("\nKey Learnings:")
print("1. LLMs are invoked with simple .invoke() calls")
print("2. Prompt templates make prompts reusable with variables")
print("3. Chains connect components with the | operator (LCEL)")
print("4. Output parsers format LLM responses")
print("\nNext: Try exercises/02_chains.py to learn about complex chains!")
