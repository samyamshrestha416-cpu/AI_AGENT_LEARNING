from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver # for memomry

load_dotenv()

PRODUCTS = {
    "wireless headphones": {"price": 79.99,  "description": "Over-ear Bluetooth, 30-hr battery, active noise cancellation."},
    "smart watch":         {"price": 199.99, "description": "Tracks heart rate and sleep. 5-day battery, water-resistant."},
    "mechanical keyboard": {"price": 129.00, "description": "Tenkeyless, Cherry MX Brown switches, per-key RGB."},
    "laptop stand":        {"price": 34.99,  "description": "Adjustable aluminium, fits 11-17 inch laptops, folds flat."},
}
REVIEWS = {
    "wireless headphones": {"reviews": 1262, "rating": 4.6},
    "smart watch":         {"reviews": 340,  "rating": 3.9},
    "mechanical keyboard": {"reviews": 67,   "rating": 4.8},
    "laptop stand":        {"reviews": 781,  "rating": 4.5},
}


@tool
def get_product(name: str) -> str:
    """Look up a product by name and return its price, rating, stock, and description."""
    p = PRODUCTS.get(name.lower())
    if not p:
        return f"Product not found. Available: {', '.join(PRODUCTS)}"
    return str(p)

@tool
def get_review(name: str) -> str:
    """Look up a product review by a product name. Return the product name, number of reviews and rating"""
    r = REVIEWS.get(name.lower())
    if not r:
        return f"Review not available for this product"
    return str(r)

# llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")


# agent = create_agent(
#     model=llm,
#     tools=[get_product],
#     system_prompt="You are a helpful product assistant for an online tech store.",
# )

# def ask(question : str):
#     result = agent.invoke({"messages" : [{"role":"user", "content": question}]})
#     print(result["messages"][-1].text)

# ask("What is price of  wireless headphones")

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

agent2 = create_agent(
    model=llm,
    tools=[get_product, get_review],
    system_prompt="You are a helpful product assistant for an online tech store.",
    checkpointer=InMemorySaver() #saves memory and remembers everything
)
config = {"configurable": {"thread_id": "1"}}# to remember things in thread

def ask2(question: str):
    result = agent2.invoke({"messages": [{"role": "user", "content": question}]}, config=config,)
    print(result["messages"][-1].text)

ask2("What is the price of wireless headphones?")
ask2("What about its rating?")   # "its" works because of memory

