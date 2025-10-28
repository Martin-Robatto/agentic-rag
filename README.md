# Agentic RAG with LangGraph 🦜🔍

Implementation of Reflective RAG, Self-RAG & Adaptive RAG tailored towards developers and production-oriented applications for learning LangGraph🦜🕸️.

This repository contains a refactored version of the original [LangChain's Cookbook](https://github.com/mistralai/cookbook/tree/main/third_party/langchain),

See Original YouTube video:[Advance RAG control flow with Mistral and LangChain](https://www.youtube.com/watch?v=sgnrL7yo1TE)

of [Sophia Young](https://x.com/sophiamyang) from Mistral & [Lance Martin](https://x.com/RLanceMartin) from LangChain

## Features

- **Refactored Notebooks**: The original LangChain notebooks have been refactored to enhance readability, maintainability, and usability for developers.
- **Production-Oriented**: The codebase is designed with a focus on production readiness, allowing developers to seamlessly transition from experimentation to deployment.
- **Test Coverage**: Comprehensive test coverage ensures the reliability and stability of the application, enabling developers to validate their implementations effectively.
- **Documentation**: Detailed documentation and branches guides developers through setting up the environment, understanding the codebase, and utilizing LangGraph effectively.

## Environment Variables

To run this project, you will need to add the following environment variables to your .env file:

```bash
OPENAI_API_KEY=your_openai_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here  # For web search capabilities
LANGCHAIN_API_KEY=your_langchain_api_key_here  # Optional, for tracing
LANGCHAIN_TRACING_V2=true                      # Optional
LANGCHAIN_PROJECT=agentic-rag                  # Optional
```

> **Important Note**: If you enable tracing by setting `LANGCHAIN_TRACING_V2=true`, you must have a valid LangSmith API key set in `LANGCHAIN_API_KEY`. Without a valid API key, the application will throw an error.

## Prerequisites

- Python 3.10+
- Basic understanding of LLMs and RAG systems
- Familiarity with Python and vector databases (helpful but not required)

## Run Locally

Clone the project

```bash
  git clone https://github.com/Martin-Robatto/agentic-rag.git
```

Go to the project directory

```bash
  cd agentic-rag
```

Install dependencies

```bash
  pipenv install
```

Start the Agentic Rag flow

```bash
  pipenv run start
```

## Running Tests

To run tests, run the following command

```bash
  pipenv run test
```
## Acknowledgements

Original LangChain repository: [LangChain Cookbook](https://github.com/mistralai/cookbook/tree/main/third_party/langchain)
By [Sophia Young](https://x.com/sophiamyang) from Mistral & [Lance Martin](https://x.com/RLanceMartin) from LangChain