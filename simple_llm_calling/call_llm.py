from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()

llm = ChatGoogleGenerativeAI(model = "gemini-3.5-flash-lite")

response = llm.invoke([
    ["system", "you are a helpfull assistant that answers in short one line"],
    ["human", "waht is the biggest animal in world ?"]
])
print(response.text)