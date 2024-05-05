import uuid


def chat(app, session_id: str, user_input: str) -> str:
    return app.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": session_id}},
    )


def run_chat(app, session_id=None, answer_key=None):
    if session_id is None:
        session_id = str(uuid.uuid4())
    print("Start chatting! Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        answer = chat(app, session_id, user_input)
        if answer_key is not None:
            answer = answer[answer_key]
        print(f"Bot: {answer}")


# ------------------------------

# api key
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

from openai import OpenAI

client = OpenAI(api_key=api_key)


def simple_query_openai(
    text, model="gpt-3.5-turbo", messages=None, system="You are a helpful assistant."
):
    if messages is None:
        messages = []
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}]
        + messages
        + [{"role": "user", "content": text}],
    )
    return response.choices[0].message.content
