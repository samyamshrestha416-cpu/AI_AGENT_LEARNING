from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()

llm = ChatGoogleGenerativeAI(model = "gemini-3.5-flash-lite")

with open("bloodwork.txt" , "r") as f:
    blood_report = f.read()

#print(blood_report[:200])#print bloodreport in less than 200 words

extraction_prompt = f"""
You are a medical data extraction assistant.

From the blood report below, extract ALL test values and classify each one as HIGH, LOW, or NORMAL
based on the reference ranges provided in the report.

Format your response as:
- Test Name: value | Status: HIGH/LOW/NORMAL | Reference: range

Blood Report:
{blood_report}
give it in json format.
"""

extraction_response = llm.invoke(extraction_prompt)
extraction_value = extraction_response.text
print("==EXTRACTED VALUES==")
print(extraction_value)

diet_prompt = f"""
You are a clinical nutritionist specializing in Indian dietary habits.

Based on the blood work analysis below, write:
1. A short health summary in 2 lines explaining the patient's condition in simple language
2. A short, practical Indian diet plan having only two sections: (1) Foods to avoid (2) Foods to eat more of.
   Do not include any other sections in diet plan.

Blood Work Analysis:
{extraction_value}
"""
diet_response=llm.invoke(diet_prompt)
print("== DIET PLAN ==")
print(diet_response.text)
