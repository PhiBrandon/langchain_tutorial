from langchain_aws import ChatBedrock
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()


class ClassificationOutput(BaseModel):
    categorization: str = Field(..., description="The cateogry that the customer is.")
    explanation: str = Field(..., description="An explanation for the categorization.")


class EmailListOutput(BaseModel):
    email_list_selection: str = Field(
        ..., description="Email list that customer should be part of."
    )
    explanation: str = Field(
        ..., description="Exmaplantion for the email list selection"
    )

llm = ChatBedrock(model_id="anthropic.claude-3-haiku-20240307-v1:0")

from langchain_core.tools import tool
@tool
def classify_customer(customer_name: str):
    """Classify the customer"""
    output_parser = PydanticOutputParser(pydantic_object=ClassificationOutput)

    prompt_template = PromptTemplate(
        template="How would you categorize this customer: {customer_information} based on {industry}. The categories are as follows: {categories}\n\nOnly output JSON.\n\n{format_instructions}",
        input_variables=["customer_information", "industry", "categories"],
        partial_variables={"format_instructions": output_parser.get_format_instructions()},
    )

    customer_information = (
        "Brandon loves cats and is an avid parrot lover as well. He doesn't prefer dogs."
    )
    industry = "Ecommerce store that sells dog toys."
    categories = str(["BEST CUSTOMER", "OKAY CUSTOMER", "WORST CUSTOMER"])

    chain_1 = prompt_template | llm | output_parser
    classification_output = chain_1.invoke(
        {
            "customer_information": customer_information,
            "industry": industry,
            "categories": categories,
        }
    )
    return classification_output

@tool
def add_to_list(classification_output: dict):
    """Choose list to add customer to"""
    output_parser_2 = PydanticOutputParser(pydantic_object=EmailListOutput)
    prompt_template_2 = PromptTemplate(
        template="Based on the categories selected and previous question, put the prospect on a partciualr emailing list. {email_lists}\n\n{previous_responses}\n\n{format_instructions}",
        input_variables=["email_lists", "previous_responses"],
        partial_variables={
            "format_instructions": output_parser_2.get_format_instructions()
        },
    )
    email_lists = str(["DOG LIST", "CAT LIST", "PARROT LIST"])

    chain_2 = prompt_template_2 | llm | output_parser_2
    email_lists = chain_2.invoke(
        {
            "email_lists": email_lists,
            "previous_responses": str(classification_output),
        }
    )
