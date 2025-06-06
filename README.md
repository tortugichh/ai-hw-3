# Multi-Agent System with Google ADK, Langchain, and LlamaIndex

This project implements a multi-agent system using the Google Agent Developer Kit (ADK). It includes a search agent powered by Langchain and SerpAPI, and a summarization agent utilizing LlamaIndex and OpenAI.

## Technologies Used

*   Google Agent Developer Kit (ADK)
*   Langchain
*   LlamaIndex
*   SerpAPI
*   OpenAI
*   Python-dotenv

## Setup

1.  Clone the repository:

    ```bash
    git clone <repository_url>
    cd ai-hw-3
    ```

2.  Create and activate a virtual environment:

    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

3.  Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Configuration

Create a `.env` file in the root directory of the project with the following API keys:

```dotenv
GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY"
SERPAPI_API_KEY="YOUR_SERPAPI_API_KEY"
OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
```

Replace the placeholder values with your actual API keys.

## How to Run

Run the main script:

```bash
python main.py
```

The script will prompt you to enter a search query or use a default one.

## Project Structure

*   `main.py`: Entry point of the application, orchestrates the workflow.
*   `adk_workflow.py`: Defines the multi-agent workflow using Google ADK.
*   `langchain_search_adk_agent.py`: Implements the search agent using Langchain and SerpAPI.
*   `llama_index_summarize_adk_agent.py`: Implements the summarization agent using LlamaIndex and OpenAI.
*   `requirements.txt`: Lists the project dependencies.
*   `.env`: Configuration file for API keys (should not be committed to Git).
*   `.gitignore`: Specifies intentionally untracked files that Git should ignore. 