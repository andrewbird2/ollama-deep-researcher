import json
from typing_extensions import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import START, END, StateGraph

from assistant.configuration import Configuration, SearchAPI
from assistant.utils import deduplicate_and_format_sources, tavily_search, format_sources, perplexity_search
from assistant.state import SummaryState, SummaryStateInput, SummaryStateOutput
from assistant.prompts import query_writer_instructions, summarizer_instructions, reflection_instructions

# Function schemas for OpenAI
QUERY_SCHEMA = {
    "name": "generate_search_query",
    "description": "Generate a search query based on the research topic",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query to use"
            }
        },
        "required": ["query"]
    }
}

REFLECTION_SCHEMA = {
    "name": "generate_followup_query",
    "description": "Generate a follow-up query based on knowledge gaps",
    "parameters": {
        "type": "object",
        "properties": {
            "follow_up_query": {
                "type": "string",
                "description": "The follow-up search query"
            }
        },
        "required": ["follow_up_query"]
    }
}

# Nodes
def generate_query(state: SummaryState, config: RunnableConfig):
    """ Generate a query for web search """

    # Format the prompt
    query_writer_instructions_formatted = query_writer_instructions.format(research_topic=state.research_topic)

    # Generate a query
    configurable = Configuration.from_runnable_config(config)
    llm_json_mode = ChatOpenAI(model="o3-mini")
    result = llm_json_mode.invoke(
        [SystemMessage(content=query_writer_instructions_formatted),
        HumanMessage(content=f"Generate a query for web search:")],
        functions=[QUERY_SCHEMA]
    )
    function_call = result.additional_kwargs.get('function_call', {})
    if function_call and function_call.get('name') == 'generate_search_query':
        query_data = json.loads(function_call['arguments'])
        return {"search_query": query_data['query']}

    # Fallback if function calling fails
    return {"search_query": state.research_topic}

def web_research(state: SummaryState, config: RunnableConfig):
    """ Gather information from the web """

    # Configure
    configurable = Configuration.from_runnable_config(config)

    # Handle both cases for search_api:
    # 1. When selected in Studio UI -> returns a string (e.g. "tavily")
    # 2. When using default -> returns an Enum (e.g. SearchAPI.TAVILY)
    if isinstance(configurable.search_api, str):
        search_api = configurable.search_api
    else:
        search_api = configurable.search_api.value

    # Search the web
    if search_api == "tavily":
        try:
            print(state.search_query)
            search_results = tavily_search(state.search_query, include_raw_content=True, max_results=5)
        except Exception as e:
            print(e)
            print(state.search_query)
            raise e
        search_str = deduplicate_and_format_sources(search_results, max_tokens_per_source=1000, include_raw_content=True)
    elif search_api == "perplexity":
        search_results = perplexity_search(state.search_query, state.research_loop_count)
        search_str = deduplicate_and_format_sources(search_results, max_tokens_per_source=1000, include_raw_content=False)
    else:
        raise ValueError(f"Unsupported search API: {configurable.search_api}")

    return {"sources_gathered": [format_sources(search_results)], "research_loop_count": state.research_loop_count + 1, "web_research_results": [search_str]}

def summarize_sources(state: SummaryState, config: RunnableConfig):
    """ Summarize the gathered sources """

    # Existing summary
    existing_summary = state.running_summary

    # Most recent web research
    most_recent_web_research = state.web_research_results[-1]

    # Build the human message
    if existing_summary:
        human_message_content = (
            f"<User Input> \n {state.research_topic} \n <User Input>\n\n"
            f"<Existing Summary> \n {existing_summary} \n <Existing Summary>\n\n"
            f"<New Search Results> \n {most_recent_web_research} \n <New Search Results>"
        )
    else:
        human_message_content = (
            f"<User Input> \n {state.research_topic} \n <User Input>\n\n"
            f"<Search Results> \n {most_recent_web_research} \n <Search Results>"
        )

    # Run the LLM
    configurable = Configuration.from_runnable_config(config)
    llm = ChatOpenAI(model="o3-mini")
    result = llm.invoke(
        [SystemMessage(content=summarizer_instructions),
        HumanMessage(content=human_message_content)]
    )

    return {"running_summary": result.content}

def reflect_on_summary(state: SummaryState, config: RunnableConfig):
    """ Reflect on the summary and generate a follow-up query """

    # Generate a query
    configurable = Configuration.from_runnable_config(config)
    llm_json_mode = ChatOpenAI(model="o3-mini")
    result = llm_json_mode.invoke(
        [SystemMessage(content=reflection_instructions.format(research_topic=state.research_topic)),
        HumanMessage(content=f"Identify a knowledge gap and generate a follow-up web search query based on our existing knowledge: {state.running_summary}")],
        functions=[REFLECTION_SCHEMA]
    )
    function_call = result.additional_kwargs.get('function_call', {})
    if function_call and function_call.get('name') == 'generate_followup_query':
        query_data = json.loads(function_call['arguments'])
        # Extract just the key information and format as a concise search query
        llm_query_formatter = ChatOpenAI(model="o3-mini")
        format_result = llm_query_formatter.invoke(
            [SystemMessage(content="Convert the following query into a concise search-friendly format of no more than 10 words:"),
             HumanMessage(content=query_data['follow_up_query'])]
        )
        return {"search_query": format_result.content}

    # Fallback if function calling fails
    return {"search_query": f"Tell me more about {state.research_topic}"}

def finalize_summary(state: SummaryState):
    """ Finalize the summary """

    # Format all accumulated sources into a single bulleted list
    all_sources = "\n".join(source for source in state.sources_gathered)
    state.running_summary = f"## Summary\n\n{state.running_summary}\n\n ### Sources:\n{all_sources}"
    return {"running_summary": state.running_summary}

def route_research(state: SummaryState, config: RunnableConfig) -> Literal["finalize_summary", "web_research"]:
    """ Route the research based on the follow-up query """

    configurable = Configuration.from_runnable_config(config)
    if state.research_loop_count <= configurable.max_web_research_loops:
        return "web_research"
    else:
        return "finalize_summary"

# Add nodes and edges
builder = StateGraph(SummaryState, input=SummaryStateInput, output=SummaryStateOutput, config_schema=Configuration)
builder.add_node("generate_query", generate_query)
builder.add_node("web_research", web_research)
builder.add_node("summarize_sources", summarize_sources)
builder.add_node("reflect_on_summary", reflect_on_summary)
builder.add_node("finalize_summary", finalize_summary)

# Add edges
builder.add_edge(START, "generate_query")
builder.add_edge("generate_query", "web_research")
builder.add_edge("web_research", "summarize_sources")
builder.add_edge("summarize_sources", "reflect_on_summary")
builder.add_conditional_edges("reflect_on_summary", route_research)
builder.add_edge("finalize_summary", END)

graph = builder.compile()
