LLM Model API for Contentstack
This project provides a complete backend API built with Python and FastAPI that can connect to a Contentstack space, index its content into a vector database, and provide a RAG-powered chat interface to "talk" to your content.

Features
Content-Model Agnostic: Automatically discovers and indexes all content types in your stack.

Vector Search: Uses ChromaDB to store text embeddings for fast, semantic search.

RAG Pipeline: Implements a Retrieval-Augmented Generation pipeline to provide contextually accurate answers.

Fast LLM Responses: Leverages the Groq API for near-instant language model inference.

Simple Setup: Ready to run with just a few setup steps.

How It Works
/index Endpoint: A developer provides their Contentstack credentials. The API fetches all content, splits it into chunks, creates vector embeddings using OpenAI, and stores them in a dedicated ChromaDB collection.

/chat Endpoint: A user sends a question. The API creates an embedding for the question, queries ChromaDB to find the most relevant content chunks, and then passes this context along with the original question to an LLM (Groq Llama3) to generate a precise answer.

Setup and Installation
1. Prerequisites
Python 3.8+

A Contentstack account with a Stack API Key, Delivery Token, and Environment name.

An OpenAI API Key (for creating embeddings).

A Groq API Key (for fast LLM responses).

2. Clone the Repository
Clone or download the project files into a local directory.

3. Create a Virtual Environment
It's highly recommended to use a virtual environment to manage dependencies.

# Navigate to your project directory
cd /path/to/your/project

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
.\venv\Scripts\activate

4. Install Dependencies
Install all the required Python libraries from the requirements.txt file.

pip install -r requirements.txt

5. Configure Environment Variables
Create a .env file in the root of your project by copying the example file.

cp .env.example .env

Now, open the .env file and add your actual API keys from OpenAI and Groq.

6. Run the API Server
You can run the application directly from your terminal.

uvicorn main:app --reload

The API will now be running at http://127.0.0.1:8000. The --reload flag means the server will automatically restart if you make any changes to the code.

How to Use the API
You can interact with the API using any HTTP client (like Postman or curl), but the easiest way is to use the built-in interactive documentation.

Open your browser and navigate to http://127.0.0.1:8000/docs.

Index Your Content:

Expand the POST /index endpoint.

Click "Try it out".

Fill in the stack_api_key, delivery_token, environment, and a unique collection_name (e.g., "my-travel-blog").

Click "Execute". This might take a few moments depending on the amount of content in your stack.

Chat with Your Content:

Expand the POST /chat endpoint.

Click "Try it out".

In the question field, ask something related to your content.

In the collection_name field, use the exact same name you used during indexing (e.g., "my-travel-blog").

Click "Execute" and see the AI-powered response!