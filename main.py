from services import vectorEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from configs import client, prompt 
import time

vectors = None
print("Initializing vector database...\n")
vectors = vectorEmbeddings()
print("Initialized vector database. Preparing the prompt...\n")

conversation_history = []

while True:
    question = str(input("Enter your question: "))
    if question:
        document_chain = create_stuff_documents_chain(llm=client, prompt=prompt)
        retriever = vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        response = retrieval_chain.invoke({
            'input': question,
            'history': "\n".join(conversation_history)
        })
        answer = response['answer']
        print(f"Answer: {answer}\n")
        conversation_history.append(f"Question: {question}\n")
        conversation_history.append(f"Answer: {answer}\n")