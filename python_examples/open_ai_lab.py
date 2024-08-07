import os
from dotenv import load_dotenv
from openai import OpenAI


model = "gpt-3.5-turbo"


def chat_openai (user_question, client, model):
    """
    This function sends a chat message to the OpenAI API and returns the content of the response.
    It takes two parameters: the chat prompt and the model to use for the chat.
    """
    prompt = """  
    {user_question}

    Analyze the user's question and provide an answer based upon the context of the question.
    """.format(
        user_question=user_question
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You are a rude cynical assistant network engineer:",
            },
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content


# This function will load the documents, split them into chunks, embed the chunks, and store the embeddings in a database
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

user_question = "Which routers are better, Juniper or Cisco?"

response = chat_openai(user_question, client, model)

print(response)
