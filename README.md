# AI Study Assistant Chatbot

This application is designed to help students efficiently process and interact with course materials using GPT-4-turbo. The app provides an interactive UI built with Streamlit, enabling file uploads (for reading materials or homework) and follow-up chat functionality with maintained context and conversation history.

## Features

- **Home Page**  
  - Users enter the course name (e.g., "Generative AI") via a text input field.
  - A submit button moves the user to the next page.

- **Upload File Page**  
  - Provides a switchable upload area for two types of files: **Reading Materials** and **Homework**.
  - Supports file uploads for formats: **PPTX, DOCX, PDF,** and **TXT**.
  - Files can be dragged and dropped or selected from local storage.
  - Only one file per category can be uploaded at a time.
  - Upon upload, the file is immediately processed and its content is extracted and summarized.
  - Uploaded files appear in a clickable history, which can be used to revisit previous outputs and chats.

- **Chatbot Page**  
  - Users can ask follow-up questions regarding the current file.
  - The chatbot uses GPT-4-turbo via a prepared internal prompt to generate consistent output.
    - For Reading Materials: It produces a concise introduction and conclusion.
    - For Homework: It extracts key points and generates additional ideas.
  - The conversation context includes both the current file summary and any previous uploads.

- **Chat History / File Detail Page**  
  - Displays the summary and QA history for a previously uploaded file.
  - Users can ask new follow-up questions based on the historical context.
  - Visual separators are used to clearly delineate each Q&A pair.

## Innovations

- **Enhanced Learning Efficiency**:  
  The app focuses on making learning more efficient by allowing students to upload their course materials directly and interact with them using an AI chatbot. This addresses common challenges such as the inability to directly upload files in standard ChatGPT interfaces and difficulties in managing citations from extensive reading materials.

- **Context Preservation Across Files**:  
  The app maintains a global context that includes summaries from all previously uploaded files. This enables follow-up questions to leverage both the current file and historical context, providing a richer, more integrated learning experience.

- **Handling Oversized Files**:  
  To overcome the token and string length limitations of the GPT-4-turbo API, oversized files are automatically split into manageable chunks. Each chunk is processed individually, and their outputs are merged to produce a final, coherent summary.

## Installation & Usage

### Prerequisites

Make sure you have the following installed:
- [Python 3.8+](https://www.python.org/)
- [Streamlit](https://streamlit.io/)
- [PyPDF2](https://pypi.org/project/PyPDF2/)
- [python-pptx](https://pypi.org/project/python-pptx/)
- [python-docx](https://pypi.org/project/python-docx/)
- [chardet](https://pypi.org/project/chardet/)
- OpenAI Python client

You can install the required libraries using pip:

```bash
pip install streamlit PyPDF2 python-pptx python-docx chardet openai
```

### Running the App

To run the app, navigate to the project directory in your command line and use the following command:

```bash
streamlit run app.py
```

The app will open in your default web browser. Follow the on-screen instructions to enter the course name, upload files, and interact with the chatbot.

## Notes

- The app uses GPT-4-turbo as its LLM. Please ensure you have access to this model via your OpenAI API key.
- File content is extracted using dedicated libraries for each file type.
- The internal prompts are designed to produce consistent and structured outputs.
- Oversized files are automatically split into chunks to avoid exceeding API limits, and the chunk outputs are merged to form the final summary.

---

Feel free to modify or extend this README as needed to better suit your project's requirements or to add any additional details.