import streamlit as st
from dotenv import load_dotenv
import time
import openai
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.callbacks import get_openai_callback
import os
from unstructured.partition.pdf import partition_pdf
from IPython.display import display, HTML
from base64 import b64decode

import io
import base64
import numpy as np
from PIL import Image
from langchain.schema.messages import HumanMessage, SystemMessage
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

import uuid
from langchain.vectorstores import Chroma
from langchain.storage import InMemoryStore
from langchain.schema.document import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
from operator import itemgetter
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda


# Sidebar contents
with st.sidebar:
    st.title('💬 User Manual Chat App')
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot that can answer
    any questions about products based on their user manuals. 
    Please upload a user manual in the pdf format. 
    ''')
    add_vertical_space(5)
    st.write('Made by Abhinav Agarwal')

load_dotenv()


def main():
    """
    Main function to run the Streamlit app, handling file uploads and user queries.
    """

    st.header("Chat with User Manual 💬")

    # Initialize session state variables if not already set
    if 'pdf_processed' not in st.session_state:
        st.session_state.pdf_processed = False
    if 'data' not in st.session_state:
        st.session_state.data = None
    # upload a PDF file
    uploaded_file = st.file_uploader("Upload your user manual in PDF Format", type='pdf')

    if uploaded_file is not None:
        if not st.session_state.pdf_processed:
            process_pdf(uploaded_file)
            st.session_state.pdf_processed = True
            st.success("Chat Model ready. Use the query box to ask questions.")
    
    # Accept user questions/query
    query = st.text_input("Ask questions about your User Manual:")
    if query:
        handle_query(query)


def encode_image(image_path):
    """
    Encode an image file to a base64 string.
    :param image_path: Path to the image file.
    :return: Base64 encoded string of the image.
    """

    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def image_summarize(img_base64,prompt):
    """
    Summarize the content of an image using a pre-trained model.
    :param img_base64: Base64 string of the image.
    :param prompt: Prompt text to guide the model summarization.
    :return: Summary of the image content.
    """

    retries = 3
    delay = 0.276

    for i in range(retries):
        try:
            chat = ChatOpenAI(model="gpt-4-vision-preview",
                            max_tokens=1024)

            msg = chat.invoke(
                [
                    HumanMessage(
                        content=[
                            {"type": "text", "text":prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{img_base64}"
                                },
                            },
                        ]
                    )
                ]
            )
            return msg.content
        except openai.RateLimitError as e:
            print(f"Rate limit exceeded, retrying in {delay} seconds...")
            time.sleep(delay)
            delay *= 2  # Exponential backoff
        except openai.Error as e:
            print(f"An error occurred: {e}")
            time.sleep(delay)
    return None


def plt_img_base64(img_base64):
    """
    Display an image from a base64 string using IPython display.
    :param img_base64: Base64 string of the image to display.
    """
    # Create an HTML img tag with the base64 string as the source
    image_html = f'<img src="data:image/jpeg;base64,{img_base64}" />'

    # Display the image by rendering the HTML
    display(HTML(image_html))


def split_image_text_types(docs):
    """
    Split documents into images and texts based on their content.
    :param docs: List of documents to split.
    :return: Dictionary with keys 'images' and 'texts', containing the respective documents.
    """
    b64 = []
    text = []
    for doc in docs:
        try:
            b64decode(doc)
            b64.append(doc)
        except Exception as e:
            text.append(doc)
    return {
        "images": b64,
        "texts": text
    }


def index_content(texts, text_summaries, tables, table_summaries, img_base64_list, image_summaries):
    """
    Index text and image content into a retrievable format using a vector store and retriever.
    :param texts: List of text documents.
    :param text_summaries: Summaries of text documents.
    :param tables: List of table documents.
    :param table_summaries: Summaries of table documents.
    :param img_base64_list: List of base64-encoded images.
    :param image_summaries: Summaries of images.
    :return: Configured retriever object.
    """
    vectorstore = Chroma(collection_name="multi_modal_rag", embedding_function=OpenAIEmbeddings())
    store = InMemoryStore()
    id_key = "doc_id"
    retriever = MultiVectorRetriever(vectorstore=vectorstore, docstore=store, id_key=id_key)

    # Indexing texts
    doc_ids = [str(uuid.uuid4()) for _ in texts]
    summary_texts = [Document(page_content=s, metadata={id_key: id}) for s, id in zip(text_summaries, doc_ids)]
    retriever.vectorstore.add_documents(summary_texts)
    retriever.docstore.mset(list(zip(doc_ids, texts)))

    # Indexing tables
    table_ids = [str(uuid.uuid4()) for _ in tables]
    summary_tables = [Document(page_content=s, metadata={id_key: id}) for s, id in zip(table_summaries, table_ids)]
    retriever.vectorstore.add_documents(summary_tables)
    retriever.docstore.mset(list(zip(table_ids, tables)))

    # Indexing images
    img_ids = [str(uuid.uuid4()) for _ in img_base64_list]
    summary_img = [Document(page_content=s, metadata={id_key: id}) for s, id in zip(image_summaries, img_ids)]
    retriever.vectorstore.add_documents(summary_img)
    retriever.docstore.mset(list(zip(img_ids, img_base64_list)))

    # Optional: Return the retriever if you need to use it outside this function
    return retriever


def process_pdf(uploaded_file):
    """
    Process the uploaded PDF file, extract content, and update session state with the processed data.
    :param uploaded_file: Uploaded file object containing the PDF.
    """
    # Display file details
    file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type, "FileSize": uploaded_file.size}
    st.write(file_details)
    
    # Define the path to save the file (current directory)
    file_path = os.path.join(os.getcwd(), uploaded_file.name)
    
    # Save the uploaded PDF to the current directory
    if os.path.exists(file_path):
        os.remove(file_path)  # Remove the file if it exists
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"Saved File: {uploaded_file.name} to the current directory")
    
    # Define the path for the 'data' subfolder in the current working directory
    current_directory = os.getcwd()
    data_subfolder_path = os.path.join(current_directory, 'image_data')
    
    # Create the 'data' subfolder if it does not exist
    os.makedirs(data_subfolder_path, exist_ok=True)
    
    # Extract content from the PDF
    raw_pdf_elements = partition_pdf(
        filename=file_path,
        extract_images_in_pdf=True,
        infer_table_structure=True,
        extract_image_block_types=["Image", "Table"],
        extract_image_block_to_payload=False,
        chunking_strategy="by_title",
        max_characters=4000,
        new_after_n_chars=3800,
        combine_text_under_n_chars=2000,
        extract_image_block_output_dir=data_subfolder_path,
        image_output_dir_path=data_subfolder_path,
    )
    st.success("Data Extracted Successfully")

    # Categorize by type
    tables = []
    texts = []
    for element in raw_pdf_elements:
        if "unstructured.documents.elements.Table" in str(type(element)):
            tables.append(str(element))
        elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
            texts.append(str(element))

    # Summarize tables
    prompt_text = "You are an assistant tasked with summarizing tables and text from a LG Washing Machine Manual. " \
                  "Preserve important details and give a concise summary of the table or text. Table or text chunk: {element}"
    prompt = ChatPromptTemplate.from_template(prompt_text)
    model = ChatOpenAI(temperature=0, model="gpt-4")
    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

    text_summaries = texts
    table_summaries = summarize_chain.batch(tables, {"max_concurrency": 5})

    # Encode and summarize images
    img_base64_list = []
    image_summaries = []
    prompt = "Describe the following image from the LG Washing Machine User Manual in detail. Be specific about tables, " \
             "washing instructions, usage instructions, graphics, etc. There might be some logos or irrelevant images."
    #counter = 0
    for img_file in sorted(os.listdir(data_subfolder_path)):
        print(f"processing {img_file}\n")
        if img_file.endswith('.jpg'):
            img_path = os.path.join(data_subfolder_path, img_file)
            base64_image = encode_image(img_path)
            img_base64_list.append(base64_image)
            image_summaries.append(image_summarize(base64_image, prompt))
        #    counter+=1
        #if counter > 5:
        #    break

    # Store processed data in session state
    st.session_state.tables = tables
    st.session_state.texts = texts
    st.session_state.table_summaries = table_summaries
    st.session_state.img_base64_list = img_base64_list
    st.session_state.image_summaries = image_summaries

    # After all processing is done, index the content
    retriever = index_content(texts, text_summaries, tables, table_summaries, img_base64_list, image_summaries)
    
    # Store the retriever in session state for later query handling
    st.session_state.retriever = retriever
    st.success("Content indexed successfully")
    st.session_state.pdf_processed = True


def truncate_history(conversation_history, max_length=6):
    """
    Truncate the conversation history to a specified maximum length.
    :param conversation_history: List of historical conversation entries.
    :param max_length: Maximum number of entries to keep in the history.
    :return: Truncated list of conversation history.
    """
    # Keep only the last `max_length` entries of the conversation history
    if len(conversation_history) > max_length:
        return conversation_history[-max_length:]
    return conversation_history


def update_history(query, response):
    """
    Update the session state with a new entry in the conversation history.
    :param query: User query to add to the history.
    :param response: System's response to add to the history.
    """
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    st.session_state.conversation_history.append({"query": query, "response": response})
    st.session_state.conversation_history = truncate_history(st.session_state.conversation_history)


def prompt_func(dict):
    """
    Generate the prompt for the language model based on the given context.
    :param dict: Dictionary containing context data for generating the prompt.
    :return: HumanMessage object with the generated prompt.
    """
    format_texts = "\n".join(dict["context"]["texts"])
    image_content = []

    # Generate the truncated history context
    if 'conversation_history' in st.session_state:
        history_context = "\n".join([f"{item['query']}\\n{item['response']}" for item in st.session_state.conversation_history])
    else:
        history_context = ""

    # Combine history context with current text and tables for a comprehensive prompt
    combined_context = f"""
                        -- Conversation History --
                        {history_context}

                        -- Current Interaction --
                        Text and Tables Context:
                        {format_texts}
                        """

    if dict['context']['images']:  # Check if there are any images
        image_content = [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{dict['context']['images'][0]}"} }]
    
    return [
        HumanMessage(
            content=[
                {"type": "text", "text": f"""You are a customer service chat agent 
                 trained on a LG Washing Machine User Manual. Help the users by
                 answering their questions about a LG Washing Machine User Manual
                based only on the following comprehensive context, which can include conversation history, text, tables, 
                 and the image. You are allowed to ask clarifying questions:
                Question: {dict["question"]}
                Conversation history, Text and tables:
                {combined_context}
                """}
            ] + image_content
        )
    ]


def handle_query(query):
    """
    Handle user queries using the retriever and display responses.
    :param query: User query to process.
    """
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'retriever' in st.session_state and st.session_state.retriever:
        retriever = st.session_state.retriever
        docs = retriever.get_relevant_documents(query)
        docs_by_type = split_image_text_types(docs)
        image_url = ""
        if ("images" in docs_by_type) and (len(docs_by_type["images"]) >= 1):
            image_url = f"data:image/jpeg;base64,{docs_by_type['images'][0]}"

        # Set up the model pipeline
        model = ChatOpenAI(temperature=0, model="gpt-4-vision-preview", max_tokens=1024)
        chain = (
            {"context": retriever | RunnableLambda(split_image_text_types), "question": RunnablePassthrough()}
            | RunnableLambda(prompt_func)
            | model
            | StrOutputParser()
        )
        with get_openai_callback() as cb:
            response = chain.invoke(query)
            print(cb)
            print(response)

        # Display responses
        if response:
            with st.chat_message("Assistant"):
                st.write(response)  # Display the text message
                update_history(query, response)
                if image_url:
                    st.image(image_url, caption='Relevant Image')  # Display the image if available
        else:
            with st.chat_message("Assistant"):
                response_text = "No relevant information found for your query."
                st.write(response_text)
                update_history(query, response_text)
    else:
        st.error("Data has not been indexed. Please upload and process a PDF file before querying.")

if __name__ == '__main__':
    main()
