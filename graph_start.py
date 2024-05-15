from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage
from langgraph.graph import END, MessageGraph
from langchain_core.tools import tool
from langchain_core.messages.base import BaseMessage
from langgraph.prebuilt import ToolNode
from typing import Literal, List
from lecl_start import classify_customer, add_to_list


message = HumanMessage("What is the customer classification")
model = ChatBedrock(model_id="anthropic.claude-3-haiku-20240307-v1:0")
model_with_tools = model.bind_tools([classify_customer, add_to_list])

graph = MessageGraph()

graph.add_node("oracle", model_with_tools)
tool_node = ToolNode([classify_customer])
graph.add_node("classify_customer", tool_node)

graph.add_edge("classify_customer", END)
graph.set_entry_point("oracle")

from typing import Literal


def router(state: List[BaseMessage]) -> Literal["classify_customer", "__end__"]:
    tool_calls = state[-1].additional_kwargs.get("tool_calls", [])
    if len(tool_calls):
        return "classify_customer"
    else:
        return "__end__"


graph.add_conditional_edges("oracle", router)


runnable = graph.compile()

output = runnable.invoke(message)
