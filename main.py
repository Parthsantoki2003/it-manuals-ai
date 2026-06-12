import os
from contextlib import asynccontextmanager
from typing import List, Optional
import dspy
from dspy_qdrant import QdrantRM
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from fastapi.middleware.cors import CORSMiddleware

# =====================================================================
# SYSTEM & DATA STRUCTURES (Pydantic Models)
# =====================================================================
class ChatMessage(BaseModel):
    role: str  # "user" or "bot"
    text: str

class ChatRequest(BaseModel):
    question: str
    history: List[ChatMessage] = []

class ChatResponse(BaseModel):
    corrected_question: str
    intent: str
    answer: str
    context: List[str]

# =====================================================================
# DSPY SIGNATURES
# =====================================================================
class CorrectTypo(dspy.Signature):
    """Fix any spelling mistakes or typos in the user's input. Do NOT answer the question."""
    raw_question = dspy.InputField(desc="The original input from the user")
    corrected_question = dspy.OutputField(desc="The corrected text, ready for processing")

class ClassifyIntent(dspy.Signature):
    """Classify the user's message as either casual small talk or a technical IT question."""
    question = dspy.InputField()
    intent = dspy.OutputField(desc="Must be exactly one word: 'smalltalk' or 'it_question'")

class SmallTalkResponse(dspy.Signature):
    """Respond to casual conversation, greetings, or thanks in a friendly, helpful manner."""
    history = dspy.InputField()
    question = dspy.InputField()
    answer = dspy.OutputField()

class GenerateChatAnswer(dspy.Signature):
    """
    Analyze the retrieved context and conversation history to answer the question.
    
    CRITICAL FORMATTING RULES:
    1. Do NOT reply in a single block paragraph.
    2. Use bold Markdown headings (e.g., ### ## Section Name) to break down information.
    3. Use organized bullet points or numbered lists for steps, features, or details.
    4. Keep paragraphs short (maximum 2-3 sentences per point).
    """
    context = dspy.InputField(desc="Facts retrieved from the vector database")
    history = dspy.InputField()
    question = dspy.InputField()
    answer = dspy.OutputField()

# =====================================================================
# THE RAG BOT MODULE
# =====================================================================
class AuthenticatedRAGBot(dspy.Module):
    def __init__(self):
        super().__init__()
        self.correct_typo = dspy.Predict(CorrectTypo)
        self.classify_intent = dspy.Predict(ClassifyIntent)
        self.handle_small_talk = dspy.Predict(SmallTalkResponse)
        self.retrieve = dspy.Retrieve()
        self.generate_answer = dspy.ChainOfThought(GenerateChatAnswer)

    # --- CHANGE 'history_str' TO 'history' HERE ---
    def forward(self, question: str, history: str):
        # 1. Fix typos
        clean_question = self.correct_typo(raw_question=question).corrected_question
        
        # 2. Route intent
        routing_decision = self.classify_intent(question=clean_question).intent.strip().lower()
        
        # Path A: Small Talk
        if 'smalltalk' in routing_decision or 'greet' in routing_decision:
            # --- UPDATE TO USE 'history' ---
            casual_reply = self.handle_small_talk(history=history, question=clean_question)
            return dspy.Prediction(
                context=[], 
                answer=casual_reply.answer, 
                intent="smalltalk", 
                corrected_question=clean_question
            )
            
        # Path B: IT Question (RAG)
        retrieval_results = self.retrieve(clean_question)
        context = []
        if retrieval_results and hasattr(retrieval_results, 'passages'):
            for p in retrieval_results.passages:
                if isinstance(p, str): 
                    context.append(p)
                elif hasattr(p, 'page_content'): 
                    context.append(p.page_content)
                elif hasattr(p, 'long_text'): 
                    context.append(p.long_text)
                    
        if not context:
            context = ["No relevant context found in the database."]
            
        # --- UPDATE TO USE 'history' ---
        prediction = self.generate_answer(context=context, history=history, question=clean_question)
        return dspy.Prediction(
            context=context, 
            answer=prediction.answer, 
            intent="it_question", 
            corrected_question=clean_question
        )

# =====================================================================
# FASTAPI LIFESPAN STATE MANAGEMENT
# =====================================================================

bot_instance = None

def get_bot():
    """Loads the heavy AI models ONLY when the first message is received."""
    global bot_instance
    if bot_instance is not None:
        return bot_instance
        
    print("First request detected: Downloading and Loading AI Models...")
    
    # 1. Setup Language Model
    lm = dspy.LM("groq/llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))
    
    # 2. Setup Embedding Model
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    def custom_vectorizer(queries):
        if isinstance(queries, str): queries = [queries]
        return embedder.encode(queries).tolist()
        
    # 3. Setup Qdrant Connection
    qdrant_client = QdrantClient(
        url=os.getenv("QDRANT_URL"), 
        api_key=os.getenv("QDRANT_API_KEY")
    )
    
    # 4. Configure Retriever
    retriever_model = QdrantRM(
        qdrant_collection_name="it_manuals",
        qdrant_client=qdrant_client,
        k=3,
        document_field="page_content",
        vectorizer=custom_vectorizer
    )
    
    dspy.configure(lm=lm, rm=retriever_model)
    bot_instance = AuthenticatedRAGBot()
    
    print("AI Engine is locked and loaded!")
    return bot_instance

app = FastAPI(title="IT Manuals RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, replace "*" with your actual frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================================================
# API ENDPOINTS
# =====================================================================
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatRequest):
    # Call the loader. It will pause to load models on the first run, 
    # but run instantly on all future messages.
    bot = get_bot() 
    
    try:
        # Convert incoming list of history objects to a single string for DSPy
        formatted_history_list = []
        for msg in payload.history:
            role_label = "User" if msg.role.lower() == "user" else "Bot"
            formatted_history_list.append(f"{role_label}: {msg.text}")
        
        # Pull only the last 6 entries to keep context window light
        history_string = "\n".join(formatted_history_list[-6:])
        
        # Execute the DSPy pipeline using the 'bot' we just loaded
        response = bot(question=payload.question, history=history_string)
        
        return ChatResponse(
            corrected_question=response.corrected_question,
            intent=response.intent,
            answer=response.answer,
            context=response.context
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal Processing Error: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "engine_ready": bot_instance is not None}
