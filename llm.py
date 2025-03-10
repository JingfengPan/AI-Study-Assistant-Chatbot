import openai
from openai import Client
import math

openai.api_key = "sk-"

# Limits
TOKENS_MAX_LIMIT = 128000
STR_MAX_LENGTH = 1048576


def count_tokens(text: str) -> int:
    """
    Approximate token count by splitting on whitespace.
    For more accurate tokenization, consider using a tokenizer like tiktoken.
    """
    return len(text.split())


def split_text_evenly(text: str, n_chunks: int) -> list:
    """
    Splits the text evenly into n_chunks based on tokens.
    """
    tokens = text.split()
    total_tokens = len(tokens)
    chunk_size = math.ceil(total_tokens / n_chunks)
    chunks = []
    for i in range(0, total_tokens, chunk_size):
        chunk = " ".join(tokens[i:i+chunk_size])
        chunks.append(chunk)
    return chunks


def call_chat_api(prompt: str) -> str:
    """
    Calls the GPT-4-turbo API using the provided prompt and returns the output.
    """
    client = Client()
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.8
    )
    result = response.choices[0].message.content
    return result


def generate_summary(file_content: str, category: str, course_name: str) -> str:
    if category == "Reading Materials":
        task_description = (
            "Task: Generate a concise introduction and a concise conclusion from the document.\n"
            "Steps:\n"
            "1. Read the document content between triple quotes.\n"
            "2. For the **Introduction**, summarize the document's background, context, and objectives.\n"
            "3. For the **Conclusion**, summarize the final insights or recommendations presented.\n"
            "4. Output the result in the following format:\n"
            "   **Introduction**: <Your generated introduction>\n"
            "   (Leave one blank line)\n"
            "   **Conclusion**: <Your generated conclusion>\n"
            "5. Verify that both sections are present, distinct, and clear."
        )
    elif category == "Homework":
        task_description = (
            "Task: Extract key points and generate ideas from the homework document.\n"
            "Steps:\n"
            "1. Read the document content between triple quotes.\n"
            "2. Identify the main key points from the document.\n"
            "3. Generate additional ideas or suggestions relevant to the assignment.\n"
            "4. Output the result in the following format:\n"
            "   **Key Points**: <List the key points>\n"
            "   (Leave one blank line)\n"
            "   **Ideas**: <Further ideas or suggestions>\n"
            "5. Ensure the output is structured, clear, and meets the requirements."
        )
    else:
        return "Invalid category provided."

    base_prompt = (
        f"You are an AI study assistant for the course '{course_name}'.\n\n"
        f"{task_description}\n\n"
        f"Document Content:\n\"\"\"\n{{content}}\n\"\"\"\n\n"
        "Please analyze the document and provide the output in the exact format specified above. "
        "Make sure to follow each step, verify conditions, and only output the final result once all checks are passed."
    )

    token_count_val = count_tokens(file_content)
    str_length_val = len(file_content)

    if token_count_val <= STR_MAX_LENGTH and str_length_val <= STR_MAX_LENGTH:
        prompt = base_prompt.replace("{content}", file_content)
        return call_chat_api(prompt)
    elif token_count_val > TOKENS_MAX_LIMIT:
        n_chunks = math.ceil(token_count_val / TOKENS_MAX_LIMIT)
        chunks = split_text_evenly(file_content, n_chunks)
        chunk_summaries = []
        for chunk in chunks:
            prompt = base_prompt.replace("{content}", chunk)
            chunk_summary = call_chat_api(prompt)
            chunk_summaries.append(chunk_summary)
    elif str_length_val > STR_MAX_LENGTH:
        n_chunks = math.ceil(str_length_val / STR_MAX_LENGTH)
        chunks = split_text_evenly(file_content, n_chunks)
        chunk_summaries = []
        for chunk in chunks:
            prompt = base_prompt.replace("{content}", chunk)
            chunk_summary = call_chat_api(prompt)
            chunk_summaries.append(chunk_summary)
    combined_summaries = "\n".join(chunk_summaries)
    final_prompt = (
        f"You are an AI study assistant for the course '{course_name}'.\n\n"
        "Task: Combine the following chunk summaries into one final, coherent output that includes all required details.\n"
        "Chunk Summaries:\n\"\"\"\n" + combined_summaries + "\n\"\"\"\n\n"
        "Output the final result in the same structured format as specified above."
    )
    final_summary = call_chat_api(final_prompt)
    return final_summary


def generate_followup_response(context: str, course_name: str) -> str:
    prompt = (
        f"You are an AI study assistant for the course '{course_name}'.\n"
        "You have previously provided a summary of the document and answered some follow-up questions.\n"
        "Here is the conversation context:\n\"\"\"\n"
        f"{context}\n"
        "\"\"\"\n"
        "Now, please answer the new question in a clear, concise, and informative manner."
    )
    return call_chat_api(prompt)
