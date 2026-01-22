import os
import re
import logging
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import json
from pathlib import Path

from dotenv import load_dotenv
# Load .env from the same folder as this file (works in dev + server)
load_dotenv(Path(__file__).parent / ".env")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from groq import Groq
from sentence_transformers import SentenceTransformer, util

from pymongo import MongoClient, ASCENDING, errors as mongo_errors
import torch


# ===================== LOGGING =====================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("memori-mongo")


# ===================== CONFIG =====================

MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "smart_memory_db")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not MONGO_URI:
    raise RuntimeError("MONGO_URI environment variable is not set")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY environment variable is not set")

# Limit per user; keeps performance + cost under control
MAX_SEMANTIC_PER_USER = int(os.getenv("MAX_SEMANTIC_PER_USER", "200"))

# LLM model to use on Groq
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")


# ===================== GLOBALS =====================

# IMPORTANT: root_path must match Nginx location `/memory-chat/`
app = FastAPI(
    title="Smart Memory LLM API (MongoDB + AI Facts + History + Chat)",
    root_path="/memory-chat",        # <-- crucial for Nginx `/memory-chat/`
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# CORS (adjust for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # TODO: In production, replace "*" with your domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

groq_client = Groq(api_key=GROQ_API_KEY)

# Mongo-related globals (initialized on startup)
mongo_client: Optional[MongoClient] = None
db = None
facts_col = None
semantic_col = None
conversations_col = None

# Load embedding model once
logger.info("Loading SentenceTransformer model...")
# Force CPU for safety (change to auto-detect GPU if you want)
embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
logger.info("SentenceTransformer model loaded.")


# ===================== Pydantic Models =====================

class ChatRequest(BaseModel):
    user_id: str
    message: str


class ChatResponse(BaseModel):
    reply: str
    used_facts: Dict[str, str]
    used_semantic: List[str]


class HealthResponse(BaseModel):
    status: str


# ===================== DB INITIALIZATION =====================

def init_mongo():
    global mongo_client, db, facts_col, semantic_col, conversations_col

    try:
        mongo_client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        # Trigger connection test
        mongo_client.admin.command("ping")
        logger.info("Connected to MongoDB successfully")
    except mongo_errors.ServerSelectionTimeoutError as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        raise RuntimeError("Could not connect to MongoDB")

    db_name = MONGO_DB_NAME
    logger.info(f"Using MongoDB database: {db_name}")
    db = mongo_client[db_name]

    facts_col = db["facts"]
    semantic_col = db["semantic"]
    conversations_col = db["conversations"]

    # Create indexes (idempotent, safe to call multiple times)
    logger.info("Ensuring MongoDB indexes...")

    # Facts: (user_id + key) unique so you don't get duplicate facts
    facts_col.create_index(
        [("user_id", ASCENDING), ("key", ASCENDING)],
        unique=True,
        name="user_key_unique",
    )

    # Semantic: for quick pruning / lookup
    semantic_col.create_index(
        [("user_id", ASCENDING), ("created_at", ASCENDING)],
        name="user_created_at_index",
    )

    # Conversations: for chronological retrieval per user
    conversations_col.create_index(
        [("user_id", ASCENDING), ("created_at", ASCENDING)],
        name="user_created_at_conv_index",
    )

    logger.info("Indexes ensured.")


@app.on_event("startup")
def on_startup():
    init_mongo()


@app.on_event("shutdown")
def on_shutdown():
    global mongo_client
    if mongo_client:
        mongo_client.close()
        logger.info("MongoDB connection closed.")


# ===================== FACT STORAGE (WITH HISTORY) =====================

def store_fact(user_id: str, key: str, value: str) -> None:
    """
    Upsert a single fact (user_id + key is unique).
    Keeps history of previous values inside the same document.
    """
    now = datetime.utcnow()
    key = key.strip().lower()
    value = value.strip()

    if not key or not value:
        return

    try:
        doc = facts_col.find_one({"user_id": user_id, "key": key})

        if doc is None:
            # First time this fact appears
            facts_col.insert_one(
                {
                    "user_id": user_id,
                    "key": key,
                    "value": value,
                    "is_active": True,
                    "history": [],  # list of { value, ended_at }
                    "updated_at": now,
                }
            )
        else:
            history = doc.get("history", [])
            prev_value = doc.get("value")

            # If the value actually changed, push old value into history
            if prev_value and prev_value != value:
                history.append(
                    {
                        "value": prev_value,
                        "ended_at": now,
                    }
                )

            facts_col.update_one(
                {"_id": doc["_id"]},
                {
                    "$set": {
                        "value": value,
                        "is_active": True,
                        "history": history,
                        "updated_at": now,
                    }
                },
            )
    except Exception as e:
        logger.exception("Error storing fact")
        raise HTTPException(status_code=500, detail="Error storing fact") from e


def delete_fact(user_id: str, key: str) -> None:
    """
    Instead of deleting the document, mark it inactive
    and push the current value into history.
    """
    now = datetime.utcnow()
    key = key.strip().lower()
    if not key:
        return

    try:
        doc = facts_col.find_one({"user_id": user_id, "key": key})
        if not doc:
            return

        history = doc.get("history", [])
        prev_value = doc.get("value")

        if prev_value:
            history.append(
                {
                    "value": prev_value,
                    "ended_at": now,
                }
            )

        facts_col.update_one(
            {"_id": doc["_id"]},
            {
                "$set": {
                    "value": None,
                    "is_active": False,
                    "history": history,
                    "updated_at": now,
                }
            },
        )
    except Exception as e:
        logger.exception("Error deleting fact")
        raise HTTPException(status_code=500, detail="Error deleting fact") from e


def get_all_facts(user_id: str) -> Dict[str, str]:
    """
    Return only CURRENT active facts as a simple dict.
    """
    try:
        docs = facts_col.find({"user_id": user_id, "is_active": {"$ne": False}})
        result: Dict[str, str] = {}
        for d in docs:
            v = d.get("value")
            if v is not None and v != "":
                result[d["key"]] = v
        return result
    except Exception as e:
        logger.exception("Error fetching facts")
        raise HTTPException(status_code=500, detail="Error fetching facts") from e


def get_fact_history(user_id: str) -> Dict[str, List[str]]:
    """
    Return historical values per key, e.g.:
    {
      "company": ["Infosys", "Wipro"],
      "location": ["Kerala"]
    }
    """
    try:
        docs = facts_col.find({"user_id": user_id})
        history: Dict[str, List[str]] = {}
        for d in docs:
            key = d["key"]
            hist_entries = d.get("history", []) or []
            values = [h.get("value") for h in hist_entries if h.get("value")]
            if values:
                # Deduplicate while keeping order
                seen = set()
                uniq_vals = []
                for v in values:
                    if v not in seen:
                        seen.add(v)
                        uniq_vals.append(v)
                history[key] = uniq_vals
        return history
    except Exception as e:
        logger.exception("Error fetching fact history")
        raise HTTPException(status_code=500, detail="Error fetching fact history") from e


# ===================== SEMANTIC MEMORY =====================

def store_semantic_memory(user_id: str, text: str) -> None:
    try:
        # If we know the user's profession, prefix stored text with a small
        # role tag so searches can prefer role-matching memories later.
        facts = get_all_facts(user_id)
        profession = facts.get("profession")
        prefix = f"[role:{profession}] " if profession else ""

        embedding = embedder.encode(text).tolist()

        # Store the text with the prefix but keep the embedding derived from
        # the original text so semantic similarity remains focused on content.
        stored_text = prefix + text

        semantic_col.insert_one(
            {
                "user_id": user_id,
                "text": stored_text,
                "embedding": embedding,
                "created_at": datetime.utcnow(),
            }
        )

        # Prune if user has too many semantic memories
        count = semantic_col.count_documents({"user_id": user_id})
        if count > MAX_SEMANTIC_PER_USER:
            to_delete = count - MAX_SEMANTIC_PER_USER
            logger.info(
                f"Pruning {to_delete} old semantic memories for user {user_id}"
            )

            old_docs = semantic_col.find(
                {"user_id": user_id},
                sort=[("created_at", ASCENDING)],
                limit=to_delete,
            )
            ids = [d["_id"] for d in old_docs]
            if ids:
                semantic_col.delete_many({"_id": {"$in": ids}})

    except Exception as e:
        logger.exception("Error storing semantic memory")
        raise HTTPException(
            status_code=500, detail="Error storing semantic memory"
        ) from e


def search_semantic_memory(
    user_id: str, query: str, top_k: int = 3, min_score: float = 0.35
) -> List[Tuple[str, float]]:
    """
    Brute-force cosine similarity on up to MAX_SEMANTIC_PER_USER messages.
    This is safe and fast because we prune aggressively.
    """
    try:
        docs = list(semantic_col.find({"user_id": user_id}))
        if not docs:
            return []

        # Determine user's profession so we can prefer role-matching entries
        facts = get_all_facts(user_id)
        profession = facts.get("profession")

        query_emb = embedder.encode(query, convert_to_tensor=True)

        scored: List[Tuple[str, float]] = []
        for d in docs:
            emb = torch.tensor(d["embedding"], dtype=torch.float32)
            score = util.cos_sim(query_emb, emb.unsqueeze(0))[0][0].item()

            # Small boost when the stored text contains a matching role tag
            try:
                if profession and f"[role:{profession}]" in d.get("text", ""):
                    score += 0.15
            except Exception:
                pass

            scored.append((d["text"], score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [(t, s) for t, s in scored[:top_k] if s >= min_score]

    except Exception as e:
        logger.exception("Error searching semantic memory")
        raise HTTPException(
            status_code=500, detail="Error searching semantic memory"
        ) from e


# ===================== CONVERSATION MEMORY (SHORT-TERM CONTEXT) =====================

def store_conversation_message(user_id: str, role: str, content: str) -> None:
    """
    Store a single conversation message (user/assistant/system summary).
    """
    if not content or not content.strip():
        return

    try:
        conversations_col.insert_one(
            {
                "user_id": user_id,
                "role": role,  # "user", "assistant", or "system"
                "content": content.strip(),
                "created_at": datetime.utcnow(),
            }
        )
    except Exception as e:
        logger.exception("Error storing conversation message")
        # Best-effort; don't break chat because history write failed.


def get_recent_conversation_messages(
    user_id: str, limit: int = 12
) -> List[Dict[str, str]]:
    """
    Return the last `limit` conversation messages for the user in chronological order.
    """
    try:
        docs = list(
            conversations_col.find(
                {"user_id": user_id},
                sort=[("created_at", -1)],
                limit=limit,
            )
        )
        # We fetched newest-first; reverse to oldest-first
        docs.reverse()
        return [
            {
                "role": d.get("role", "user"),
                "content": d.get("content", ""),
            }
            for d in docs
            if d.get("content")
        ]
    except Exception as e:
        logger.exception("Error fetching recent conversation messages")
        return []


def maybe_summarize_conversation(
    user_id: str,
    max_messages: int = 80,
    keep_last: int = 30,
) -> None:
    """
    If a user has more than `max_messages` conversation messages,
    summarize the oldest part and keep only a compact summary + last `keep_last` messages.
    """
    try:
        total = conversations_col.count_documents({"user_id": user_id})
        if total <= max_messages:
            return

        to_summarize_count = total - keep_last
        if to_summarize_count <= 0:
            return

        old_docs = list(
            conversations_col.find(
                {"user_id": user_id},
                sort=[("created_at", ASCENDING)],
                limit=to_summarize_count,
            )
        )

        if not old_docs:
            return

        # Build a compact transcript
        transcript_lines = []
        for d in old_docs:
            role = d.get("role", "user")
            content = d.get("content", "")
            if content:
                transcript_lines.append(f"{role}: {content}")
        transcript = "\n".join(transcript_lines)

        if not transcript.strip():
            return

        system_prompt = """
You are a conversation summarizer for a personal AI assistant.

You will receive a chronological transcript of earlier messages between a user and an assistant.
Your job is to create a concise summary that preserves:
- The important facts about the user (preferences, profile, constraints).
- Long-running tasks or projects.
- Emotional tone or relationship context if relevant.

Do NOT include exact wording or unnecessary details.
Keep the summary short but informative.

Return ONLY the summary text, no JSON, no explanations.
""".strip()

        # Use the same Groq client to summarize
        resp = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": transcript},
            ],
        )

        summary_text = (resp.choices[0].message.content or "").strip()
        if not summary_text:
            return

        # Insert the summary as a special "system" message at the current time
        conversations_col.insert_one(
            {
                "user_id": user_id,
                "role": "system",
                "content": f"SUMMARY OF EARLIER CONVERSATION:\n{summary_text}",
                "created_at": datetime.utcnow(),
            }
        )

        # Delete the old, detailed messages that we just summarized
        ids_to_delete = [d["_id"] for d in old_docs]
        if ids_to_delete:
            conversations_col.delete_many({"_id": {"$in": ids_to_delete}})

        logger.info(
            f"Summarized conversation for user {user_id}: "
            f"compressed {len(ids_to_delete)} messages into 1 summary."
        )

    except Exception as e:
        logger.exception("Error during conversation summarization")
        # Don't break chat; summarization is a best-effort optimization.


# ===================== AI FACT EXTRACTION + UPDATE LOGIC =====================

def _strip_json_fences(text: str) -> str:
    """
    Remove ```json ... ``` or ``` ... ``` fences if the LLM returns them.
    More robust than a simple split-based approach.
    """
    text = text.strip()
    # Remove leading and trailing code fences
    text = re.sub(r"^```(?:json|python)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()
    return text


def ai_extract_and_update_facts(user_id: str, user_message: str) -> None:
    """
    Use the LLM to:
      1. Look at current facts about the user.
      2. Read the new user message.
      3. Decide which facts to SET/UPDATE and which to logically DELETE.
    This gives you AI-powered update logic (job changes, moving cities, etc.).
    """

    current_facts = get_all_facts(user_id)

    system_prompt = """
You are a memory manager for a personal AI assistant.

You are given:
- A JSON object of the CURRENT known facts about the user.
- The user's NEW message.

Your job:
1. Decide which fields should be updated or added based ONLY on explicit statements in the new message.
2. Decide which fields are no longer valid (for example, when the user says they moved to a new city or changed jobs).
3. You MUST NOT invent or guess information. Only use what is clearly stated.
4. Never delete a fact unless the user directly states that it is no longer true or clearly contradicts it.

Return ONLY a JSON object with this structure (no explanations, no extra text):

{
  "set": {
    "field_name": "new value",
    "another_field": "another value"
  },
  "delete": [
    "field_to_delete",
    "another_field_to_delete"
  ]
}

Guidelines:
- If a fact clearly changes (e.g., new job, new city), put the new value in "set"
  and ALSO include the field key in "delete" so the old value is considered past history.
- If the user says they are no longer something (e.g., "I am not working at Infosys anymore"),
  add the appropriate key (e.g., "company") to "delete".
- If there are no changes, return:

{
  "set": {},
  "delete": []
}
""".strip()

    user_content = json.dumps(
        {
            "current_facts": current_facts,
            "new_message": user_message,
        },
        ensure_ascii=False,
        indent=2,
    )

    try:
        resp = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        )
        raw = resp.choices[0].message.content or ""
        raw = _strip_json_fences(raw)

        update_obj = json.loads(raw)

        to_set = update_obj.get("set", {}) or {}
        to_delete = update_obj.get("delete", []) or []

        # Normalize keys/values
        normalized_set: Dict[str, str] = {}
        for k, v in to_set.items():
            if not isinstance(k, str):
                continue
            key = k.strip().lower()
            if isinstance(v, str):
                value = v.strip()
            else:
                value = str(v)
            if key and value:
                normalized_set[key] = value

        normalized_delete: List[str] = []
        for k in to_delete:
            if isinstance(k, str):
                kk = k.strip().lower()
                if kk:
                    normalized_delete.append(kk)

        # Apply deletes first (moves current value to history & deactivates)
        for key in normalized_delete:
            delete_fact(user_id, key)

        # Apply sets (upserts, pushing previous value into history if changed)
        for key, value in normalized_set.items():
            store_fact(user_id, key, value)

        if normalized_set or normalized_delete:
            logger.info(
                f"Updated facts for user {user_id}: set={normalized_set}, delete={normalized_delete}"
            )

    except Exception as e:
        logger.exception("Error during AI fact extraction/update")
        # Don't break chat if memory update fails.


# ===================== CORE LLM LOGIC =====================

def ask_ai(user_id: str, user_input: str) -> Tuple[str, Dict[str, str], List[str]]:
    # 1. Extract + update structured facts using AI (with history handling)
    ai_extract_and_update_facts(user_id, user_input)

    # 2. Store semantic memory for this message
    store_semantic_memory(user_id, user_input)

    # 3. Load all CURRENT facts AFTER update
    facts = get_all_facts(user_id)
    history = get_fact_history(user_id)

    fact_context_lines = []
    for k, v in facts.items():
        label = k.replace("_", " ").title()
        fact_context_lines.append(f"{label}: {v}")
    fact_context = "\n".join(fact_context_lines)

    history_lines = []
    for k, vals in history.items():
        label = k.replace("_", " ").title()
        joined = ", ".join(vals)
        history_lines.append(f"{label} (past): {joined}")
    history_context = "\n".join(history_lines)

    # 4. Retrieve related semantic messages
    sem_matches = search_semantic_memory(
        user_id, user_input, top_k=3, min_score=0.35
    )
    sem_texts = [t for t, _ in sem_matches]

    # 5. Build system context
    system_context = (
        "You are a helpful AI assistant.\n"
        "You have access to remembered facts and some past messages about this user.\n"
        "Use them to personalize your answer, but never invent new facts.\n\n"
    )

    if fact_context:
        system_context += "FACTS ABOUT USER (current):\n" + fact_context + "\n\n"

    if history_context:
        system_context += (
            "FACT HISTORY ABOUT USER (previous values, may no longer be true):\n"
            + history_context
            + "\n\n"
        )

    if sem_texts:
        system_context += "RELATED PAST MESSAGES FROM THIS USER:\n"
        for t in sem_texts:
            # Remove a stored role prefix from displayed messages for clarity
            display = t
            if isinstance(display, str) and display.startswith("[role:]"):
                # split at first '] ' to keep the original message
                parts = display.split("] ", 1)
                if len(parts) == 2:
                    display = parts[1]
            system_context += f"- {display}\n"

    # Instruct the model to prefer role/context-specific interpretations when
    # ambiguous acronyms or terms are used (e.g., "CNN" -> news vs CNN (ML)).
    disambiguation_note = (
        "When a user asks about ambiguous acronyms or terms, prefer explanations "
        "relevant to the user's profession or context when available."
    )

    if system_context.strip():
        system_context = system_context + "\n" + disambiguation_note
    else:
        system_context = disambiguation_note

    # 6. Fetch recent conversation history (short-term memory)
    recent_messages = get_recent_conversation_messages(user_id, limit=12)

    # 7. Build the messages for the LLM
    messages = [{"role": "system", "content": system_context}]

    # Add stored summaries and past turns
    for msg in recent_messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if not content:
            continue

        if role not in ("user", "assistant", "system"):
            role = "user"

        messages.append(
            {
                "role": role,
                "content": content,
            }
        )

    # Finally, append the NEW user message as the latest turn
    messages.append({"role": "user", "content": user_input})

    # 8. Call Groq LLM
    try:
        resp = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
        )
    except Exception as e:
        logger.exception("Error calling Groq API for chat reply")
        raise HTTPException(
            status_code=502, detail="Error communicating with LLM provider"
        ) from e

    reply = resp.choices[0].message.content or ""

    # 9. Store this conversation turn in the conversation history
    try:
        store_conversation_message(user_id, "user", user_input)
        store_conversation_message(user_id, "assistant", reply)
        # Optionally summarize if the conversation is getting long
        maybe_summarize_conversation(user_id)
    except Exception:
        # Don't break chat because of history writing issues
        pass

    return reply, facts, sem_texts


@app.get("/health", response_model=HealthResponse)
def health_check():
    """
    Simple health check for monitoring / uptime checks.
    """
    try:
        mongo_client.admin.command("ping")
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="MongoDB unavailable")

    return HealthResponse(status="ok")


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(payload: ChatRequest) -> ChatResponse:
    """
    Main chat endpoint.
    """
    if not payload.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    reply, facts, sem_used = ask_ai(payload.user_id, payload.message)
    return ChatResponse(reply=reply, used_facts=facts, used_semantic=sem_used)


# ===================== LOCAL DEV ENTRYPOINT =====================

if __name__ == "__main__":
    import uvicorn

    # Local dev:
    #   python app.py
    # In production with PM2, you will typically run:
    #   pm2 start "uvicorn app:app --host 0.0.0.0 --port 2222 --workers 4" --name memoryChatApi
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)