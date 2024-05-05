from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory


def make_chat_app(llm):
    store = {}

    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    output_parser = StrOutputParser()

    chat_system_prompt = """You're a helpful chatbot-assistant.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", chat_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    chat_chain = prompt | llm | output_parser

    return RunnableWithMessageHistory(
        chat_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        # output_messages_key="answer",
    )


def main(model="gpt-3.5-turbo-0125"):
    from langchain_community.chat_models import ChatOpenAI
    from src.lib import run_chat
    from dotenv import load_dotenv

    load_dotenv()

    common_settings = {
        "request_timeout": 60,
        "max_retries": 3,
        "max_tokens": 1024,
    }
    llm = ChatOpenAI(model=model, **common_settings)

    chat_app = make_chat_app(llm)

    run_chat(chat_app)


if __name__ == "__main__":
    main()
