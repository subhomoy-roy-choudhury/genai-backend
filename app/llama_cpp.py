import os
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain

# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# Make sure the model path is correct for your system!
llm = LlamaCpp(
    model_path=os.path.join(os.getcwd(), "models/mistral-7b-instruct-v0.1.Q4_K_S.gguf"),
    temperature=0.75,
    max_tokens=3000,
    top_p=1,
    n_ctx=2048,
    callback_manager=callback_manager,
    verbose=True,  # Verbose is required to pass to the callback manager
)


def run_llama_cpp(context):
    # # person_type = "Linkedin User"
    prompt_template = """Write a concise summary within 30 words of the following:
    "{text}"
    CONCISE SUMMARY:"""
    prompt = PromptTemplate.from_template(prompt_template)

    # Split text
    text_splitter = CharacterTextSplitter()
    texts = text_splitter.split_text(context)
    # Create multiple documents
    docs = [Document(page_content=t) for t in texts]
    # Text summarization
    chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)

    return chain.run(docs)
