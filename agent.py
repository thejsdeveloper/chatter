"""Core agent logic for Chatter: a CSV FAQ agent.

Kept free of Streamlit so it can be imported and tested on its own.
"""

import warnings

import pandas as pd
from langchain_openai import ChatOpenAI

# langchain-experimental warns that it is being sunset; it still works and is
# the data frame agent the brief asks for.
warnings.filterwarnings(
    "ignore", category=DeprecationWarning, module="langchain_experimental"
)
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

# Both providers speak the OpenAI API, so ChatOpenAI works for each; only the
# base URL, model id and key env var differ.
PROVIDERS = {
    "OpenRouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "model": "openai/gpt-4o-mini",
        "env_key": "OPENROUTER_API_KEY",
    },
    "OpenAI": {
        "base_url": None,
        "model": "gpt-4o-mini",
        "env_key": "OPENAI_API_KEY",
    },
}
DEFAULT_PROVIDER = "OpenRouter"
NOT_FOUND_MESSAGE = "I could not find this information in the uploaded files."

SYSTEM_PROMPT_TEMPLATE = """You are Chatter, a support assistant that answers questions ONLY from \
the CSV files a support agent has uploaded. Each file is loaded as a pandas DataFrame:

{table_list}

Rules:
1. Decide which DataFrame(s) are relevant from the file names and columns, then use the \
python tool to search them. Match keywords case-insensitively across the text columns \
(for example Question, Topic, Feature, Category) and read the matching rows in full.
2. Answer ONLY with information found in those rows. Never use general world knowledge, \
never guess, and never invent values.
3. For text questions, return the key text from the answer-style column (for example \
Answer, Policy Text, Detail Text or Description). Keep the exact numbers, dates and limits.
4. For numeric questions (totals, averages, counts), compute the result with pandas \
and report the number. Columns may store numbers as text or mix units (for example \
"495" and "3%"): convert with pd.to_numeric(..., errors="coerce"), never add values with \
different units together, and say which rows you included.
5. If no row answers the question, reply with exactly: "{not_found}"
6. Write the final answer in clear, friendly English that a support agent can paste into \
an email or chat. Do not mention DataFrames, pandas, code or column names.
"""


def build_system_prompt(dataframes: dict[str, pd.DataFrame]) -> str:
    """Describe each loaded file (df1, df2, ...) so the model can pick the right table."""
    table_list = "\n".join(
        f"- df{i} = {name} ({len(df)} rows; columns: {', '.join(map(str, df.columns))})"
        for i, (name, df) in enumerate(dataframes.items(), start=1)
    )

    return SYSTEM_PROMPT_TEMPLATE.format(
        table_list=table_list, not_found=NOT_FOUND_MESSAGE
    )


def build_agent(
    dataframes: dict[str, pd.DataFrame],
    api_key: str,
    provider: str = DEFAULT_PROVIDER,
    model: str | None = None,
    verbose: bool = True,
):
    """Create a LangChain pandas agent over all uploaded DataFrames."""
    if not dataframes:
        raise ValueError("Upload at least one CSV file first.")

    cfg = PROVIDERS[provider]
    llm = ChatOpenAI(
        model=model or cfg["model"],
        temperature=0,
        api_key=api_key,
        base_url=cfg["base_url"],
    )

    # The library runs str.format() on the prefix, so escape literal braces.
    prefix = build_system_prompt(dataframes).replace("{", "{{").replace("}", "}}")

    return create_pandas_dataframe_agent(
        llm,
        list(dataframes.values()),
        agent_type="openai-tools",
        prefix=prefix,
        verbose=verbose,
        allow_dangerous_code=True,  # the agent runs pandas code locally
        return_intermediate_steps=True,
        max_iterations=10,
    )


def ask(agent, question: str) -> tuple[str, list]:
    """Run one question through the agent. Returns (answer, intermediate_steps)."""
    result = agent.invoke({"input": question})
    return result["output"], result.get("intermediate_steps", [])
