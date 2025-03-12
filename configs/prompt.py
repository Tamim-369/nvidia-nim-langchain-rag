from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context and conversation history and your thought but your thought should not be strictly against the context. Please provide the most accurate response based on the question and don't just blindly say without thinking about the context and the question properly.
<context>
{context}
<context>
<conversation_history>
{history}
<conversation_history>
Questions:{input}
"""
)