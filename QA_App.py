# export HNSWLIB_NO_NATIVE = 1
import os

import chainlit as cl
from chainlit.types import AskFileResponse
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

os.environ['OPENAI_API_KEY'] = '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'

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=1000)
embeddings = OpenAIEmbeddings()

welcome_message = '''Welcome to the Chainlit PDF QA demo. To get started:
1. Upload a PDF or text file
2. Ask a question about the file
'''


def process_file(file: AskFileResponse):
    global load
    import tempfile

    if file.type == 'text/plain':
        load = TextLoader
    elif file.type == 'application/pdf':
        load = PyPDFLoader

    with tempfile.NamedTemporaryFile() as temp:
        temp.write(file.content)
        loader = load(temp.name)
        documents = loader.load()
        docs = text_splitter.split_documents(documents)
        for i, doc in enumerate(docs):
            doc.metadata["source"] = f"source_{i}"
        return docs


def get_docsearch(file: AskFileResponse):
    docs = process_file(file)
    # save data in user session
    cl.user_session.set('docs', docs)
    # create a unique namespace
    docsearch = Chroma.from_documents(
        docs, embeddings
    )
    return docsearch


@cl.on_chat_start
async def start():
    # sending an image with the local file path
    await cl.Message(content='Welcome To a Document-based Answering System').send()

    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content=welcome_message,
            accept=['text/plain', 'application/pdf'],
            max_size_mb=100,
            timeout=180
        ).send()

    file = files[0]

    msg = cl.Message(content=f"Processing '{file.name}'...")
    await msg.send()

    docsearch = await cl.make_async(get_docsearch)(file)

    chain = RetrievalQAWithSourcesChain.from_chain_type(
        ChatOpenAI(temperature=0, streaming=True),
        chain_type='stuff',
        retriever=docsearch.as_retriever(max_tokens_limit=4097)
    )

    # let user know the system is ready
    msg.content = f"'{file.name}' processed. You can now ask questions!"
    await msg.update()

    cl.user_session.set('chain', chain)


@cl.on_message
async def main(message):
    chain = cl.user_session.get('chain')  # type:RetrievalQAWithSourcesChain

    callback = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True,
        answer_prefix_tokens=['FINAL', 'ANSWER']
    )

    callback.answer_reached = True
    response = await chain.acall(message, callbacks=[callback])

    answer = response['answer']
    sources = response['sources'].strip()
    sources_elements = []

    # get document from the user
    docs = cl.user_session.get('docs')
    metadatas = [doc.metadata for doc in docs]
    all_sources = [meta['source'] for meta in metadatas]

    if sources:
        found_sources = []

        # add sources to the message
        for source in sources.split(','):
            source_name = source.strip().replace('.', '')
            # get the index of the source
            try:
                index = all_sources.index(source_name)
            except ValueError:
                continue
            text = docs[index].page_content
            found_sources.append(source_name)
            # create the text element reference in the message
            sources_elements.append(cl.Text(content=text, name=source_name))

        if found_sources:
            answer += f"\nSources {', '.join(found_sources)}"
        else:
            answer += "\nNo sources found"

    if callback.has_streamed_final_answer:
        callback.final_stream.elements = sources_elements
        await callback.final_stream.update()
    else:
        await cl.Message(content=answer, elements=sources_elements).send()
