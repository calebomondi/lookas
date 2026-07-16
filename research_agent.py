import operator
import time
from pydantic import BaseModel, Field
from typing import Annotated, List, Any, Sequence
from typing_extensions import TypedDict

# from langchain_community.document_loaders import WikipediaLoader
from langchain_tavily import TavilySearch 
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, get_buffer_string
# from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langgraph.constants import Send
from langgraph.graph import END, MessagesState, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver

# import wikipediaapi

import logging
from config import Config

memory = MemorySaver()

# wiki = wikipediaapi.Wikipedia(
#     user_agent="PyrMyd (kalebmokua@gmail.com)",  # required, must have contact info
#     language="en"
# )

### Config & Logging

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s %(message)s")
logging.getLogger("pyrmyd").setLevel(logging.INFO)
logger = logging.getLogger("pyrmyd.nodes")
cfg = Config()

### LLM

GOOGLE_MODELS = [
    "gemini-3.5-flash", 
    "gemini-2.5-flash"
] 

GROQ_MODELS = [
    "llama-3.3-70b-versatile", 
    "openai/gpt-oss-120b", 
    # "openai/gpt-oss-20b"
]

OPENCODE_MODELS = [
    "deepseek-v4-pro",
    "kimi-k2.7-code",
    "glm-5.2",
]

OPENCODE_OPENAI_MODELS = {
    "glm-5.2", "glm-5.1", "kimi-k2.7-code", "kimi-k2.6",
    "deepseek-v4-pro", "deepseek-v4-flash", "mimo-v2.5", "mimo-v2.5-pro"
}

OPENCODE_ANTHROPIC_MODELS = {
    "minimax-m3", "minimax-m2.7", "qwen3.7-max", "qwen3.7-plus", "qwen3.6-plus"
}


def _llm(provider: str = "gemini", model: str | None = None):
    if provider == "groq":
        m = model or GROQ_MODELS[0]
        logger.info("Using Groq (%s)", m)

        return ChatGroq(
            model=m,
            groq_api_key=cfg.groq_api_key,
            temperature=0,
        )

    elif provider == "opencode":
        m = model or OPENCODE_MODELS[0]
        if m in OPENCODE_ANTHROPIC_MODELS:
            return ChatAnthropic(
                model=m,
                anthropic_api_key=cfg.opencode_api_key,
                base_url="https://opencode.ai/zen/go/v1",
                temperature=0,
            )
        return ChatOpenAI(
            model=m,
            api_key=cfg.opencode_api_key,
            base_url="https://opencode.ai/zen/go/v1",
            temperature=0,
        )
    
    else:
        m = model or GOOGLE_MODELS[0]
        logger.info("Using Gemini (%s)", m)

        return ChatGoogleGenerativeAI(
            model=m,
            google_api_key=cfg.google_api_key,
            temperature=0,
        )

def _invoke_structured_with_fallback(
    messages: Sequence[Any],
    schema: Any | None = None,
    providers: list[tuple[str, list[str]]] | None = None,
    log_label: str = "LLM call",
):
    """
    Invoke an LLM across a prioritized list of (provider, [models]) pairs,
    falling back on failure. If `schema` is provided, uses structured output;
    otherwise does a plain invoke.

    providers defaults to Gemini models then Groq models.
    """
    if providers is None:
        providers = [
            # ("opencode", OPENCODE_MODELS), 
            ("groq", GROQ_MODELS),
            ("gemini", GOOGLE_MODELS), 
        ]

    last_err = None
    for provider, models in providers:
        for model in models:
            try:
                llm = _llm(provider, model=model)
                if schema is not None:
                    llm = llm.with_structured_output(schema)
                result = llm.invoke(messages)
                logger.info("%s succeeded with %s/%s", log_label, provider, model)
                return result
            except Exception as e:
                last_err = e
                logger.warning(
                    "%s failed with %s/%s: %s", log_label, provider, model, e
                )
                continue

    raise RuntimeError(f"All LLM providers failed for {log_label}") from last_err

### Schema 

class Analyst(BaseModel):
    affiliation: str = Field(
        description="Primary affiliation of the analyst.",
    )
    name: str = Field(
        description="Name of the analyst."
    )
    role: str = Field(
        description="Role of the analyst in the context of the topic.",
    )
    description: str = Field(
        description="Description of the analyst focus, concerns, and motives.",
    )
    @property
    def persona(self) -> str:
        return f"Name: {self.name}\nRole: {self.role}\nAffiliation: {self.affiliation}\nDescription: {self.description}\n"

class Perspectives(BaseModel):
    analysts: List[Analyst] = Field(
        description="Comprehensive list of analysts with their roles and affiliations.",
    )

class GenerateAnalystsState(TypedDict):
    topic: str # Research topic
    max_analysts: int # Number of analysts
    human_analyst_feedback: str # Human feedback
    analysts: List[Analyst] # Analyst asking questions

class InterviewState(MessagesState):
    max_num_turns: int # Number turns of conversation
    context: Annotated[list, operator.add] # Source docs
    analyst: Analyst # Analyst asking questions
    interview: str # Interview transcript
    sections: list # Final key we duplicate in outer state for Send() API
    section_word_count: int # Per-section word limit (auto-distributed)

class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Search query for retrieval.")

class ResearchGraphState(TypedDict):
    topic: str # Research topic
    max_analysts: int # Number of analysts
    human_analyst_feedback: str # Human feedback
    analysts: List[Analyst] # Analyst asking questions
    sections: Annotated[list, operator.add] # Send() API key
    introduction: str # Introduction for the final report
    content: str # Content for the final report
    conclusion: str # Conclusion for the final report
    final_report: str # Final report
    word_count: int # Total target word count for the report

### Nodes and edges

analyst_instructions="""You are tasked with creating a set of AI analyst personas. Follow these instructions carefully:

1. First, review the research topic:
{topic}
        
2. Examine any editorial feedback that has been optionally provided to guide creation of the analysts: 
        
{human_analyst_feedback}
    
3. Determine the most interesting themes based upon documents and / or feedback above.
                    
4. Pick the top {max_analysts} themes.

5. Assign one analyst to each theme."""

def create_analysts(state: GenerateAnalystsState):
    """Create analysts"""
    topic = state["topic"]
    max_analysts = state["max_analysts"]
    human_analyst_feedback = state.get("human_analyst_feedback", "")

    system_message = analyst_instructions.format(
        topic=topic,
        human_analyst_feedback=human_analyst_feedback,
        max_analysts=max_analysts,
    )

    analysts = _invoke_structured_with_fallback(
        messages=[
            SystemMessage(content=system_message),
            HumanMessage(content="Generate the set of analysts."),
        ],
        schema=Perspectives,
        log_label="create_analysts",
    )

    logger.info("Analysts Created: %s", analysts.analysts)
    return {"analysts": analysts.analysts}

# Generate analyst question
question_instructions = """You are an analyst tasked with interviewing an expert to learn about a specific topic. 

Your goal is boil down to interesting and specific insights related to your topic.

1. Interesting: Insights that people will find surprising or non-obvious.
        
2. Specific: Insights that avoid generalities and include specific examples from the expert.

Here is your topic of focus and set of goals: {goals}
        
Begin by introducing yourself using a name that fits your persona, and then ask your question.

Continue to ask questions to drill down and refine your understanding of the topic.
        
When you are satisfied with your understanding, complete the interview with: "Thank you so much for your help!"

Remember to stay in character throughout your response, reflecting the persona and goals provided to you."""

# def generate_question(state: InterviewState):

#     """ Node to generate a question """

#     # Get state
#     analyst = state["analyst"]
#     messages = state["messages"]

#     # Generate question 
#     system_message = question_instructions.format(goals=analyst.persona)

#     # for gemini_model in GOOGLE_MODELS:
#     #     try:
#     #         # Initialize the LLM
#     #         llm = _llm("gemini", model=gemini_model)
#     #         # Generate question 
#     #         question = llm.invoke([SystemMessage(content=system_message)]+messages)
#     #         logger.info("Question Generated: %s", question.content)
#     #         # Write messages to state
#     #         return {"messages": [question]}
#     #     except Exception as e:
#     #         logger.warning("Failed to generate question with model %s: %s", gemini_model, e)
#     #         continue

#     for groq_model in GROQ_MODELS:
#         try:
#             # Initialize the LLM
#             llm = _llm("groq", model=groq_model)
#             # Generate question 
#             question = llm.invoke([SystemMessage(content=system_message)]+messages)
#             logger.info("Question Generated: %s", question.content)
#             # Write messages to state
#             return {"messages": [question]}
#         except Exception as e:
#             logger.warning("Failed to generate question with model %s: %s", groq_model, e)

#     raise RuntimeError("All LLM Providers failed.")

def generate_question(state: InterviewState):
    """Node to generate a question"""
    analyst = state["analyst"]
    messages = state["messages"]

    system_message = question_instructions.format(goals=analyst.persona)

    question = _invoke_structured_with_fallback(
        messages=[SystemMessage(content=system_message)] + messages,
        log_label="generate_question",
    )

    logger.info("Question Generated: %s", question.content)
    return {"messages": [question]}

# Search query writing
search_instructions = SystemMessage(content=f"""You will be given a conversation between an analyst and an expert. 

Your goal is to generate a well-structured query for use in retrieval and / or web-search related to the conversation.
        
First, analyze the full conversation.

Pay particular attention to the final question posed by the analyst.

Convert this final question into a well-structured web search query                                  
""")

def _invoke_tavily_with_retry(
    query: str,
    max_results: int = 3,
    max_retries: int = 3,
    backoff_seconds: float = 1.0,
    log_label: str = "Tavily search",
):
    """
    Run a Tavily search with retry on transient failures.
    Retries max_retries times with linear backoff, then raises.
    """
    tavily_search = TavilySearch(max_results=max_results)

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            data = tavily_search.invoke({"query": query})
            search_docs = data.get("results", data)
            logger.info("%s succeeded on attempt %d", log_label, attempt)
            return search_docs
        except Exception as e:
            last_err = e
            logger.warning(
                "%s failed on attempt %d/%d: %s", log_label, attempt, max_retries, e
            )
            if attempt < max_retries:
                time.sleep(backoff_seconds * attempt)  # linear backoff
            continue

    raise RuntimeError(f"{log_label} failed after {max_retries} attempts") from last_err

def search_web(state: InterviewState):
    """Retrieve docs from web search"""

    search_query = _invoke_structured_with_fallback(
        messages=[search_instructions] + state["messages"],
        schema=SearchQuery,
        log_label="search_web:search_query",
    )
    logger.info("Search query Tavily: %s", search_query.search_query)

    search_docs = _invoke_tavily_with_retry(
        query=search_query.search_query,
        log_label="search_web:tavily",
    )
    logger.info("Search docs Tavily: %s", search_docs)

    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document href="{doc["url"]}"/>\n{doc["content"]}\n</Document>'
            for doc in search_docs
        ]
    )

    return {"context": [formatted_search_docs]}

# Generate expert answer
answer_instructions = """You are an expert being interviewed by an analyst.

Here is analyst area of focus: {goals}. 
        
You goal is to answer a question posed by the interviewer.

To answer question, use this context:
        
{context}

When answering questions, follow these guidelines:
        
1. Use only the information provided in the context. 
        
2. Do not introduce external information or make assumptions beyond what is explicitly stated in the context.

3. The context contain sources at the topic of each individual document.

4. Include these sources your answer next to any relevant statements. For example, for source # 1 use [1]. 

5. List your sources in order at the bottom of your answer. [1] Source 1, [2] Source 2, etc
        
6. If the source is: <Document source="assistant/docs/llama3_1.pdf" page="7"/>' then just list: 
        
[1] assistant/docs/llama3_1.pdf, page 7 
        
And skip the addition of the brackets as well as the Document source preamble in your citation."""

def generate_answer(state: InterviewState):
    """Node to answer a question"""

    analyst = state["analyst"]
    messages = state["messages"]
    context = state["context"]

    system_message = answer_instructions.format(goals=analyst.persona, context=context)

    answer = _invoke_structured_with_fallback(
        messages=[SystemMessage(content=system_message)] + messages,
        log_label="generate_answer",
    )
    logger.info("Answer Generated: %s", answer.content)

    # Name the message as coming from the expert
    answer.name = "expert"

    return {"messages": [answer]}

def save_interview(state: InterviewState):
    
    """ Save interviews """

    # Get messages
    messages = state["messages"]
    
    # Convert interview to a string
    interview = get_buffer_string(messages)
    
    # Save to interviews key
    return {"interview": interview}

def route_messages(state: InterviewState, 
                   name: str = "expert"):

    """ Route between question and answer """
    
    # Get messages
    messages = state["messages"]
    max_num_turns = state.get('max_num_turns',2)

    # Check the number of expert answers 
    num_responses = len(
        [m for m in messages if isinstance(m, AIMessage) and m.name == name]
    )

    # End if expert has answered more than the max turns
    if num_responses >= max_num_turns:
        return 'save_interview'

    # This router is run after each question - answer pair 
    # Get the last question asked to check if it signals the end of discussion
    last_question = messages[-2]
    
    if "Thank you so much for your help" in last_question.content:
        return 'save_interview'
    return "ask_question"

# Write a summary (section of the final report) of the interview
section_writer_instructions = """You are an expert technical writer. 
            
Your task is to create a short, easily digestible section of a report based on a set of source documents.

1. Analyze the content of the source documents: 
- The name of each source document is at the start of the document, with the <Document tag.
        
2. Create a report structure using markdown formatting:
- Use ## for the section title
- Use ### for sub-section headers
        
3. Write the report following this structure:
a. Title (## header)
b. Summary (### header)
c. Sources (### header)

4. Make your title engaging based upon the focus area of the analyst: 
{focus}

5. For the summary section:
- Set up summary with general background / context related to the focus area of the analyst
- Emphasize what is novel, interesting, or surprising about insights gathered from the interview
- Create a numbered list of source documents, as you use them
- Do not mention the names of interviewers or experts
- Aim for approximately {word_count} words maximum
- Use numbered sources in your report (e.g., [1], [2]) based on information from source documents
        
6. In the Sources section:
- Include all sources used in your report
- Provide full links to relevant websites or specific document paths
- Separate each source by a newline. Use two spaces at the end of each line to create a newline in Markdown.
- It will look like:

### Sources
[1] Link or Document name
[2] Link or Document name

7. Be sure to combine sources. For example this is not correct:

[3] https://ai.meta.com/blog/meta-llama-3-1/
[4] https://ai.meta.com/blog/meta-llama-3-1/

There should be no redundant sources. It should simply be:

[3] https://ai.meta.com/blog/meta-llama-3-1/
        
8. Final review:
- Ensure the report follows the required structure
- Include no preamble before the title of the report
- Check that all guidelines have been followed"""

def write_section(state: InterviewState):
    """Node to write a section"""

    # interview = state["interview"]
    context = state["context"]
    analyst = state["analyst"]

    system_message = section_writer_instructions.format(
        focus=analyst.description,
        word_count=state.get("section_word_count", 400),
    )

    section = _invoke_structured_with_fallback(
        messages=[
            SystemMessage(content=system_message),
            HumanMessage(content=f"Use this source to write your section: {context}"),
        ],
        providers=[("groq", GROQ_MODELS)],
        log_label="write_section",
    )

    return {"sections": [section.content]}

def interviewBuilder():
    # Add nodes and edges 
    interview_builder = StateGraph(InterviewState)
    interview_builder.add_node("ask_question", generate_question)
    interview_builder.add_node("search_web", search_web)
    # interview_builder.add_node("search_wikipedia", search_wikipedia)
    interview_builder.add_node("answer_question", generate_answer)
    interview_builder.add_node("save_interview", save_interview)
    interview_builder.add_node("write_section", write_section)

    # Flow
    interview_builder.add_edge(START, "ask_question")
    interview_builder.add_edge("ask_question", "search_web")
    # interview_builder.add_edge("ask_question", "search_wikipedia")
    interview_builder.add_edge("search_web", "answer_question")
    # interview_builder.add_edge("search_wikipedia", "answer_question")
    interview_builder.add_conditional_edges("answer_question", route_messages,['ask_question','save_interview'])
    interview_builder.add_edge("save_interview", "write_section")
    interview_builder.add_edge("write_section", END)

    # Compile
    return interview_builder.compile()

def initiate_all_interviews(state: ResearchGraphState):

    """ Kick off interviews in parallel via Send() API, auto-distributing word count """

    topic = state["topic"]
    total = state.get("word_count", 1000)
    n = state.get("max_analysts", 3)
    section_limit = max(100, int(total * 0.6 / n))

    return [
        Send(
            "conduct_interview", 
            {
                "analyst": analyst,
                "section_word_count": section_limit,
                "messages": [
                    HumanMessage(
                        content=f"So you said you were writing an article on {topic}?"
                    )
                ]
            }
        ) for analyst in state["analysts"]
    ]

# Write a report based on the interviews
report_writer_instructions = """You are a technical writer creating a report on this overall topic: 

{topic}
    
You have a team of analysts. Each analyst has done two things: 

1. They conducted an interview with an expert on a specific sub-topic.
2. They write up their finding into a memo.

Your task: 

1. You will be given a collection of memos from your analysts.
2. Think carefully about the insights from each memo.
3. Consolidate these into a crisp overall summary that ties together the central ideas from all of the memos. 
4. Summarize the central points in each memo into a cohesive single narrative.

To format your report:
 
1. Use markdown formatting. 
2. Include no pre-amble for the report.
3. Use no sub-heading. 
4. Start your report with a single title header: ## Insights
5. Do not mention any analyst names in your report.
6. Preserve any citations in the memos, which will be annotated in brackets, for example [1] or [2].
7. Create a final, consolidated list of sources and add to a Sources section with the `## Sources` header.
8. List your sources in order and do not repeat. ## Always include sources in the report
9. Target around {word_count} words maximum for the report body (excluding sources).

[1] Source 1
[2] Source 2

Here are the memos from your analysts to build your report from: 

{context}"""

def write_report(state: ResearchGraphState):
    """Node to write the final report body"""

    sections = state["sections"]
    topic = state["topic"]

    # Concat all sections together
    formatted_str_sections = "\n\n".join([f"{section}" for section in sections])

    word_count = state.get("word_count", 1000)
    section_target = max(100, int(word_count * 0.6))

    system_message = report_writer_instructions.format(
        topic=topic, 
        context=formatted_str_sections, 
        word_count=section_target
    )

    report = _invoke_structured_with_fallback(
        messages=[
            SystemMessage(content=system_message),
            HumanMessage(content="Write a report based upon these memos."),
        ],
        log_label="write_report",
    )

    return {"content": report.content}

# Write the introduction or conclusion
intro_conclusion_instructions = """You are a technical writer finishing a report on {topic}

You will be given all of the sections of the report.

You job is to write a crisp and compelling introduction or conclusion section.

The user will instruct you whether to write the introduction or conclusion.

Include no pre-amble for either section.

Target around {word_count} words, crisply previewing (for introduction) or recapping (for conclusion) all of the sections of the report.

Use markdown formatting. 

For your introduction, create a compelling title and use the # header for the title.

For your introduction, use ## Introduction as the section header. 

For your conclusion, use ## Conclusion as the section header.

Here are the sections to reflect on for writing: {formatted_str_sections}"""

def _write_intro_or_conclusion(state: ResearchGraphState, kind: str):
    """kind is 'introduction' or 'conclusion'"""
    sections = state["sections"]
    topic = state["topic"]

    formatted_str_sections = "\n\n".join([f"{section}" for section in sections])

    word_count = state.get("word_count", 1000)
    limit = max(50, int(word_count * 0.2))
    instructions = intro_conclusion_instructions.format(
        topic=topic, formatted_str_sections=formatted_str_sections, word_count=limit
    )

    result = _invoke_structured_with_fallback(
        messages=[instructions, HumanMessage(content=f"Write the report {kind}")],
        log_label=f"write_{kind}",
    )

    return {kind: result.content}

def write_introduction(state: ResearchGraphState):
    return _write_intro_or_conclusion(state, "introduction")

def write_conclusion(state: ResearchGraphState):
    return _write_intro_or_conclusion(state, "conclusion")

def finalize_report(state: ResearchGraphState):

    """ The is the "reduce" step where we gather all the sections, combine them, and reflect on them to write the intro/conclusion """

    # Save full final report
    content = state["content"]
    if content.startswith("## Insights"):
        content = content.strip("## Insights")
    if "## Sources" in content:
        try:
            content, sources = content.split("\n## Sources\n")
        except:
            sources = None
    else:
        sources = None

    final_report = state["introduction"] + "\n\n---\n\n" + content + "\n\n---\n\n" + state["conclusion"]
    if sources is not None:
        final_report += "\n\n## Sources\n" + sources
    return {"final_report": final_report}

def build_agent():
    """ Build the research agent """
    # Add nodes and edges 
    builder = StateGraph(ResearchGraphState)
    builder.add_node("create_analysts", create_analysts)
    builder.add_node("conduct_interview", interviewBuilder())
    builder.add_node("write_report",write_report)
    builder.add_node("write_introduction",write_introduction)
    builder.add_node("write_conclusion",write_conclusion)
    builder.add_node("finalize_report",finalize_report)

    # Logic
    builder.add_edge(START, "create_analysts")
    builder.add_conditional_edges("create_analysts", initiate_all_interviews, ["create_analysts", "conduct_interview"])
    builder.add_edge("conduct_interview", "write_report")
    builder.add_edge("conduct_interview", "write_introduction")
    builder.add_edge("conduct_interview", "write_conclusion")
    builder.add_edge(["write_conclusion", "write_report", "write_introduction"], "finalize_report")
    builder.add_edge("finalize_report", END)

    # Compile
    return builder.compile(checkpointer=memory)

