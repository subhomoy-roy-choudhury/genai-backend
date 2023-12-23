import os
from langchain.embeddings import GooglePalmEmbeddings
from langchain.llms import LlamaCpp, GooglePalm
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import LLMChain, MapReduceDocumentsChain, ReduceDocumentsChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.document_loaders import WebBaseLoader
from app.config import DEBUG, MODEL_PATH
from app.prompt_templates import (
    SUMMERIZATION_PROMPT_TEMPLATE,
    LEETCODE_BLOG_PROMPT_TEMPLATE,
)

# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])


# Make sure the model path is correct for your system!
llm = LlamaCpp(
    model_path=os.path.join(os.getcwd(), MODEL_PATH),
    temperature=0.75,
    max_tokens=3000,
    top_p=1,
    n_ctx=2048,
    callback_manager=callback_manager,
    verbose=True,  # Verbose is required to pass to the callback manager
)

palm_llm = GooglePalm(google_api_key="AIzaSyBMpA_LWZ0aOBdvlw3e2LTbLvblQYznuwU")
palm_llm.temperature = 0.8



def run_summerize_text(context):
    prompt = PromptTemplate.from_template(SUMMERIZATION_PROMPT_TEMPLATE)

    # Split text
    text_splitter = CharacterTextSplitter()
    texts = text_splitter.split_text(context)
    # Create multiple documents
    docs = [Document(page_content=t) for t in texts]
    # Text summarization
    chain = load_summarize_chain(llm, chain_type="map_reduce", prompt=prompt)

    return chain.run(docs)


def run_format_leetcode_data_into_blog(data):
    prompt = PromptTemplate.from_template(template=LEETCODE_BLOG_PROMPT_TEMPLATE)
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run(**data.dict())
    return response

def run_summerize_docs(url):
    # Map
    map_template = """The following is a set of documents
    {docs}
    Based on this list of docs, please identify the main themes 
    Helpful Answer:"""
    map_prompt = PromptTemplate.from_template(map_template)
    map_chain = LLMChain(llm=palm_llm, prompt=map_prompt)

    # Reduce
    reduce_template = """The following is set of summaries:
    {docs}
    Take these and distill it into a final, consolidated summary of the main themes. 
    Helpful Answer:"""
    reduce_prompt = PromptTemplate.from_template(reduce_template)

    # Run chain
    reduce_chain = LLMChain(llm=palm_llm, prompt=reduce_prompt)

    # Takes a list of documents, combines them into a single string, and passes this to an LLMChain
    combine_documents_chain = StuffDocumentsChain(
        llm_chain=reduce_chain, document_variable_name="docs"
    )

    # Combines and iteravely reduces the mapped documents
    reduce_documents_chain = ReduceDocumentsChain(
        # This is final chain that is called.
        combine_documents_chain=combine_documents_chain,
        # If documents exceed context for `StuffDocumentsChain`
        collapse_documents_chain=combine_documents_chain,
        # The maximum number of tokens to group documents into.
        token_max=4000,
    )

    # Combining documents by mapping a chain over them, then combining results
    map_reduce_chain = MapReduceDocumentsChain(
        # Map chain
        llm_chain=map_chain,
        # Reduce chain
        reduce_documents_chain=reduce_documents_chain,
        # The variable name in the llm_chain to put the documents in
        document_variable_name="docs",
        # Return the results of the map steps in the output
        return_intermediate_steps=False,
    )
    # Load Content from web
    loader = WebBaseLoader(
        'https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/')
    docs = loader.load()

    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=2000, chunk_overlap=0
    )
    split_docs = text_splitter.split_documents(docs)

    # Use Map reduce chain to summarize
    summary = map_reduce_chain.run(split_docs)
    return summary
