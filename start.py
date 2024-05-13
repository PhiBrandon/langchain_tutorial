from langchain_aws import ChatBedrock
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()


class ClassificationOutput(BaseModel):
    categorization: str = Field(..., description="The cateogry that the customer is.")
    explanation: str = Field(..., description="An explanation for the categorization.")


output_parser = PydanticOutputParser(pydantic_object=ClassificationOutput)

llm = ChatBedrock(model_id="anthropic.claude-3-haiku-20240307-v1:0")
prompt_template = PromptTemplate(
    template="How would you categorize this customer: {customer_information} based on {industry}. The categories are as follows: {categories}\n\nOnly output JSON.\n\n{format_instructions}",
    input_variables=["customer_information", "industry", "categories"],
    partial_variables={"format_instructions": output_parser.get_format_instructions()},
)

customer_information = "Brandon loves cats and is an avid parrot lover as well. He doesn't always love dogs, but he is willing to take care of them if needed, not a preference."
industry = "Ecommerce store that sells dog toys."
categories = str(["BEST CUSTOMER", "OKAY CUSTOMER", "WORST CUSTOMER"])
prompt = prompt_template.format(
    customer_information=customer_information, industry=industry, categories=categories
)

chat_prompt_template = ChatPromptTemplate(
    messages=[
        SystemMessage(content="You are an expert at outputting JSON."),
        HumanMessage(content=prompt),
    ]
)

chat_prompt = chat_prompt_template.format_messages(
    customer_information=customer_information, industry=industry, categories=categories
)

output = llm.invoke(chat_prompt)
parsed_output = output_parser.parse(output.content)
print(parsed_output.categorization)


