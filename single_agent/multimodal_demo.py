from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver 
from langchain_core.messages import HumanMessage
import base64

load_dotenv()

with open("blood_work.png","rb") as f:
    image_b64= base64.b64encode(f.read()).decode()

image_b64[:200]
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

message = HumanMessage(content=[
    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
    {"type": "text",      "text": "This is a blood work report. Extract all test results and flag any values outside the normal range."},
])

response = llm.invoke([message])
print(response.text)

# @tool
# def diet_recommendation(message : str)->str:
#     "look at the  of blood report and give diet recommendation. "
#     a = message.get()
#     return str(a)

# llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

# agent = create_agent(
#     model=llm,
#     tools=[diet_recommendation],
#     system_prompt="You are a helpful health assistant .",
#     checkpointer=InMemorySaver() #saves memory and remembers everything
# )
# config = {"configurable": {"thread_id": "1"}}# to remember things in thread
# response2 = llm.invoke([message])
# print(response2.text)


@tool
def diet_recommendation(abnormal_findings: str) -> str:
    """Give diet recommendations based on abnormal blood test findings, e.g. 'high cholesterol, low iron'."""
    return (
        f"Findings: {abnormal_findings}. "
        "Suggest a balanced diet addressing these findings, and advise consulting a doctor."
    )

agent = create_agent(
    model=llm,
    tools=[diet_recommendation],
    system_prompt="You are a helpful health assistant. Analyze blood reports and use the diet tool for recommendations.",
    checkpointer=InMemorySaver(),
)

config = {"configurable": {"thread_id": "1"}}

# 1. Send the image to the agent
result = agent.invoke({"messages": [message]}, config=config)
print(result["messages"][-1].text)

# 2. Follow-up (memory remembers the image and results)
result = agent.invoke(
    {"messages": [{"role": "user", "content": "Give me a diet recommendation based on this report."}]},
    config=config,
)
print(result["messages"][-1].text)

