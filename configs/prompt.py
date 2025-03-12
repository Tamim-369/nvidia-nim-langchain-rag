from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template(
"""
Analyze the context properly answer questions based on the context only. You should only think about the context and analyzing the context nothing else.
<context>
{context}
<context>
<conversation_history>
{history}
<conversation_history>
Questions:{input}
"""
)