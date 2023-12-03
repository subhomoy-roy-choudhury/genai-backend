import os
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
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
