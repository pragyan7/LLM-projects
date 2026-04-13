import os
import requests


OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"


def ask_openrouter(
    query: str,
    retrieved_chunks: list[tuple[str, float]],
    model: str = "nvidia/nemotron-3-super-120b-a12b:free",
) -> str:
    """
    Generate an answer using OpenRouter based only on retrieved PDF chunks.
    """
    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY not found. Please export it before running the app."
        )

    context = "\n\n".join(chunk for chunk, _ in retrieved_chunks)

    prompt = f"""
You are a helpful assistant for question answering over PDFs.

Answer the user's question using ONLY the context below.
If the answer is not present in the context, say:
"Not found in document."

Context:
{context}

Question:
{query}
""".strip()

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        # Optional but recommended by OpenRouter for app identification:
        "HTTP-Referer": "http://localhost:8501",
        "X-Title": "Chat with Your PDFs",
    }

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system", 
                "content": "You answer questions grounded in retrieved PDF context only."
            },
            {
                "role": "user", 
                "content": prompt
            },
        ],
        "temperature": 0.2,
    }

    response = requests.post(
        OPENROUTER_API_URL,
        headers=headers,
        json=payload,
       
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"OpenRouter API error {response.status_code}: {response.text}"
        )

    data = response.json()
    return data["choices"][0]["message"]["content"]