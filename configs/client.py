from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA
import os
# loading environment variables
load_dotenv()
# loading nvidia nim api key
NVIDIA_NIM_API = str(os.getenv("NVIDIA_NIM_API"))
os.environ['NVIDIA_API_KEY'] = NVIDIA_NIM_API
# initializing chat model
client = ChatNVIDIA(
  model="meta/llama3-70b-instruct",
  api_key=NVIDIA_NIM_API, 
  temperature=0.2,
  top_p=0.7,
  max_tokens=1024,
)