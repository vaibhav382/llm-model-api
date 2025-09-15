import os
import uvicorn
import contentstack
import chromadb
import google.generativeai as genai  
from groq import Groq
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import traceback
from fastapi.middleware.cors import CORSMiddleware


load_dotenv()

app = FastAPI(
    title="LLM Model API for Contentstack",
    description="An API that indexes Contentstack content and provides a RAG-powered chat interface.",
    version="1.2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"], 
)


try:
 
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
except Exception as e:
    print(f"API Key Error: Please ensure GEMINI_API_KEY and GROQ_API_KEY are set. Error: {e}")

try:
    chroma_client = chromadb.PersistentClient(path="./chroma_data")
except Exception as e:
    raise RuntimeError(f"Failed to initialize ChromaDB: {e}")



class IndexRequest(BaseModel):
    stack_api_key: str = Field(..., description="API Key for the Contentstack Stack.")
    delivery_token: str = Field(..., description="Delivery Token for the environment.")
    environment: str = Field(..., description="The environment name (e.g., 'main', 'development').")
    collection_name: str = Field(..., description="A unique name for this content source.")
    groq_model: str = Field("gemma2-9b-it", description="The Groq model to use for chat completions (e.g., 'llama3-70b-8192').")

class ChatRequest(BaseModel):
    question: str = Field(..., description="The user's question.")
    collection_name: str = Field(..., description="The unique name of the indexed content source.")



def parse_rich_text_editor(node):
    all_text = []
    if isinstance(node, dict):
        if "text" in node:
            all_text.append(node["text"])
        if "children" in node and isinstance(node["children"], list):
            for child_node in node["children"]:
                all_text.extend(parse_rich_text_editor(child_node))
    return all_text

def get_text_chunks(text: str, chunk_size: int = 1000, overlap: int = 100):
    if not text: return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def get_embedding(text_chunks: list[str], task_type: str, model="models/embedding-001"):
    if not text_chunks: return []
    try:
        result = genai.embed_content(
            model=model,
            content=text_chunks,
            task_type=task_type
        )
        return result['embedding']
    except Exception as e:
        print(f"Error getting Gemini embedding: {e}")
        return []


@app.get("/")
def read_root():
    return {"message": "LLM Model API is active."}

@app.post("/index")
def index_contentstack_data(request: IndexRequest):
    print("\n--- Starting Indexing Process ---")
    try:
        stack = contentstack.Stack(
            api_key=request.stack_api_key,
            delivery_token=request.delivery_token,
            environment=request.environment
        )
        
        content_types_response = stack.content_type().find()
        content_types_list = content_types_response.get('content_types', [])
        
        if not content_types_list:
             print("No content types found in the stack.")
             raise HTTPException(status_code=404, detail="No content types found in the stack.")
        
        content_type_uids = [ct['uid'] for ct in content_types_list]
        print(f"Found {len(content_type_uids)} content types: {content_type_uids}")

        collection = chroma_client.get_or_create_collection(
            name=request.collection_name,
            metadata={"groq_model": request.groq_model}
        )
        
        all_chunks = []
        all_metadatas = []
        
        for uid in content_type_uids:
            print(f"\nProcessing content type: '{uid}'")
            query_response = stack.content_type(uid).query().find()
            entries_list = query_response.get('entries', [])
            print(f"Found {len(entries_list)} entries.")
            
            for entry_dict in entries_list:
                body_content = entry_dict.get('body')
                full_text = ""
                
                if isinstance(body_content, dict):
                    text_list = parse_rich_text_editor(body_content)
                    full_text = " ".join(text_list)
                elif isinstance(body_content, str):
                    full_text = body_content
                
                if full_text:
                    print(f"  - Extracted {len(full_text)} characters from 'body' of entry: '{entry_dict.get('title', entry_dict.get('uid'))}'")
                    chunks = get_text_chunks(full_text)
                    
                    for i, chunk in enumerate(chunks):
                        all_chunks.append(chunk)
                        all_metadatas.append({
                            "source_uid": entry_dict.get('uid', 'unknown_uid'),
                            "content_type": uid,
                            "title": entry_dict.get('title', 'No Title'),
                            "chunk_index": i
                        })
                else:
                    print(f"  - No text found in 'body' of entry: '{entry_dict.get('title', entry_dict.get('uid'))}'")


        if not all_chunks:
            print("--- Indexing complete. No text content found to index. ---")
            return {"message": "No text content found to index."}
        
        print(f"\nCreating embeddings for {len(all_chunks)} text chunks...")
        embeddings = get_embedding(all_chunks, task_type="RETRIEVAL_DOCUMENT")
        if not embeddings:
            raise HTTPException(status_code=500, detail="Failed to create text embeddings.")
        
        ids = [f"{meta['source_uid']}_{meta['chunk_index']}" for meta in all_metadatas]
        
        print(f"Adding {len(ids)} chunks to the ChromaDB collection '{request.collection_name}'...")
        collection.add(embeddings=embeddings, documents=all_chunks, metadatas=all_metadatas, ids=ids)
        
        print("--- Indexing Process Finished Successfully ---")
        return {
            "message": f"Successfully indexed content for '{request.collection_name}'.",
            "total_chunks_indexed": len(all_chunks)
        }
    except Exception as e:
        print(f"--- Indexing Process Failed ---")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"An error occurred during indexing: {str(e)}")


@app.post("/chat")
def chat_with_content(request: ChatRequest):
    
    try:
        collection = chroma_client.get_collection(name=request.collection_name)
        
        collection_metadata = collection.metadata
        groq_model = "gemma2-9b-it"
        if collection_metadata and "groq_model" in collection_metadata:
            groq_model = collection_metadata["groq_model"]
        print(f"Using Groq model: {groq_model}")

    except Exception:
        raise HTTPException(status_code=404, detail=f"Content source '{request.collection_name}' not found.")

    try:
        # Specify task_type for querying
        query_embedding_list = get_embedding([request.question], task_type="RETRIEVAL_QUERY")
        if not query_embedding_list:
            raise HTTPException(status_code=500, detail="Could not embed user's question.")
        
        query_embedding = query_embedding_list[0]
        results = collection.query(query_embeddings=[query_embedding], n_results=5)
        context = "\n\n---\n\n".join(results['documents'][0])
        
        prompt = f"""
        You are a helpful assistant. Answer the user's question based ONLY on the context provided below.
        If the information is not in the context, say "I don't have enough information to answer that."

        CONTEXT:
        {context}

        USER'S QUESTION:
        {request.question}
        """

        chat_completion = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=groq_model,
            temperature=0.2
        )
        text = chat_completion.choices[0].message.content
        res = text.split("</think>")
        return {"response": res[-1]}
    except Exception as e:
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"An error occurred during chat processing: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

