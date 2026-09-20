from openai import OpenAI
from backend.config import settings

client = OpenAI(api_key=settings.OPENAI_API_KEY)

SYSTEM_PROMPT = """You are a careful assistant that answers ONLY using the provided context
excerpts from the user's uploaded document(s). Rules:
- If the answer is not clearly supported by the context, reply exactly:
  "The answer to this question was not found in the provided document(s)."
- Do not use outside knowledge. Do not speculate.
- Cite the page number(s) you used, like (p. 3), inline.
- Be concise and factual.
"""


def generate_answer(question: str, context_chunks: list[dict]) -> str:
    context_text = "\n\n".join(
        f"[Source: {c['doc_name']}, page {c['page']}]\n{c['text']}" for c in context_chunks
    ) or "No context retrieved."

    user_prompt = f"CONTEXT:\n{context_text}\n\nQUESTION:\n{question}"

    resp = client.chat.completions.create(
        model=settings.OPENAI_CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
    )
    return resp.choices[0].message.content.strip()
