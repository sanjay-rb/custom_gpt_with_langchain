# Unlocking Custom Document 🗎 Conversations 💬 with Langchain 🦜 and the Hugging Face API 🤗

## Note :
---

- The following `10 steps` are crucial for constructing a custom chat-oriented GPT based on the chosen document.
- The runtime of each cell will be determined by the system you are using.
- Steps 2, 5, and 9 entail utilizing HuggingFace and free models, which can potentially lead to longer runtime.
- You are encouraged to explore alternatives such as OpenAI or AzureOpenAI in place of HuggingFaceAPI, as they may offer enhanced performance.

## Steps :
---

1. Installation of Essential Python Modules
1. Setting Environment Variables in the System
1. Parsing PDF Documents
1. Dividing Text into Chunks
1. Converting Chunks into Embeddings & Storing them in a Vector Store

    - Utilizing the `intfloat/e5-large-v2` model from HuggingFace for Embedding & Facebook's `FAISS` for Vector Store

1. Saving the Vector Store for Future Reuse with `Pickle`
1. Directly Loading the Vector Store from the `vectorstore.pkl` File, Skipping Steps 3, 4, 5, and 6 😃

1. Performing Similarity Search Using the Vector Store
1. Creating a Large Language Model (LLM) with HuggingFace's `google/flan-t5-xxl`

    > Optional: Creating a Custom Chat History to Set Context

1. Combining All the Information into a Unified Chain Named `Conversational Retrieval QA Chain`
---

# Our Conversation Chain is Ready! Let's Engage in Some Conversations 😎😎