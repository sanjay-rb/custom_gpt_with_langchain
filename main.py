# %% [markdown]
# # Unlocking Custom Document 🗎 Conversations 💬 with Langchain 🦜 and the Hugging Face API 🤗

# %% [markdown]
# ## Note :
# - The following `10 steps` are crucial for constructing a custom chat-oriented GPT based on the chosen document.
# - The runtime of each cell will be determined by the system you are using.
# - Steps 2, 5, and 9 entail utilizing HuggingFace and free models, which can potentially lead to longer runtime.
# - You are encouraged to explore alternatives such as OpenAI or AzureOpenAI in place of HuggingFaceAPI, as they may offer enhanced performance.

# %% [markdown]
# ## Step 1: Installation of Essential Python Modules

# %%
# ! pip install PyPDF2 langchain InstructorEmbedding sentence_transformers faiss-cpu
# ! pip freeze > requirements.txt

# %% [markdown]
# ## Step 2: Setting Environment Variables in the System

# %%
from getpass import getpass

HUGGINGFACEHUB_API_TOKEN = getpass("Enter your access token generated from https://huggingface.co/settings/tokens : ")

import os

os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN

# %% [markdown]
# ## Step 3: Parsing PDF Documents

# %%
from PyPDF2 import PdfReader
pdf = PdfReader("sample.pdf")
text = ""

for page in pdf.pages:
    text += page.extract_text()

print(text)

# %% [markdown]
# ## Step 4: Dividing Text into Chunks

# %%
from langchain.text_splitter import CharacterTextSplitter

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)

chunks = text_splitter.split_text(text=text)
print(chunks)

# %% [markdown]
# ## Step 5: Converting Chunks into Embeddings & Storing them in a Vector Store
# - Utilizing the `intfloat/e5-large-v2` model from HuggingFace for Embedding & Facebook's `FAISS` for Vector Store

# %%
from langchain.embeddings import HuggingFaceEmbeddings
embedding = HuggingFaceEmbeddings(
    model_name="intfloat/e5-large-v2",
)
from langchain.vectorstores import FAISS
vectorstore = FAISS.from_texts(texts=chunks, embedding=embedding)
print(vectorstore)

# %% [markdown]
# ## Step 6: Saving the Vector Store for Future Reuse with `Pickle`

# %%
import pickle

with open("vectorstore.pkl", "wb") as pkl:
    pickle.dump(vectorstore, pkl)

# %% [markdown]
# ## Step 7: Directly Loading the Vector Store from the `vectorstore.pkl` File, Skipping Steps 3, 4, 5, and 6 😃

# %%
with open("vectorstore.pkl", "rb") as pkl:
    vectorstore = pickle.load(pkl)

# %% [markdown]
# ## Step 8: Performing Similarity Search Using the Vector Store

# %%
query = "What your policy covers?"
search_result = vectorstore.similarity_search(query=query)
print(search_result) # return 4 documents be default
search_result = vectorstore.similarity_search(query=query, k=2)
print(search_result) # return 2 documents by setting k=2

# %% [markdown]
# ## Step 9: Creating a Large Language Model (LLM) with HuggingFace's `google/flan-t5-xxl`

# %%
from langchain.llms import HuggingFaceHub

llm = HuggingFaceHub(
    repo_id="google/flan-t5-xxl",
    model_kwargs={
        "temperature": 0.5, # How innovative this model can be?  0=>None 1=>Very high innovative
    }
)
print(llm)

# %% [markdown]
# ## Optional: Creating a Custom Chat History to Set Context

# %%
chat_history = [ # (question, answer)
    (
        "How we can make our complaint?", 
        """
        Please write to:
            The Managing Director
            Arc Legal Assistance Limited
            PO Box 8921
            Colchester CO4 5YD
            Tel: 01206 615000*
            Email: customerservice@arclegal.co.uk
        """
    )
]
print(chat_history)

# %% [markdown]
# # Now, with our LLM, PDF Vector Store, and Optional Custom Chat History in Place 🎉🎉🎉

# %% [markdown]
# ## Step 10: Combining All the Information into a Unified Chain Named `Conversational Retrieval QA Chain`

# %%
from langchain.chains import ConversationalRetrievalChain

qa = ConversationalRetrievalChain.from_llm(
    llm=llm, 
    retriever=vectorstore.as_retriever(search_type = "similarity", search_kwargs = {"k":2}), 
)

print(qa)

# %% [markdown]
# ## Our Conversation Chain is Ready! Let's Engage in Some Conversations 😎😎

# %%
result = qa({"question": query, "chat_history": chat_history})
print("Human Question :", result['question'])
print("AI Answer :", result['answer'])

# %% [markdown]
# ## Let's Proceed with the Chat and Update the Chat History Accordingly

# %%
chat_history = [(query, result["answer"])]
query = "HOW TO MAKE A CLAIM?"

result = qa({"question": query, "chat_history": chat_history})
print("Human Question :", result['question'])
print("AI Answer :", result['answer'])

# %%
chat_history = [(query, result["answer"])]
query = "Under Cover 11 (Tax) What is insured?"

result = qa({"question": query, "chat_history": chat_history})
print("Human Question :", result['question'])
print("AI Answer :", result['answer'])


