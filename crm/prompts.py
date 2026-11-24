"""Protocol v0 prompts: CRMArena-Pro's ReAct prompts with SOQL/SOSL swapped for SQLite SQL and search.

  messages = prompts.v0_messages(task, schema)   # [system, user]
"""
# Adapted from CRMArena (Salesforce AI Research), crm_sandbox/agents/prompts.py, CC BY-NC 4.0.
# Changes: SQL and a search action instead of SOQL/SOSL; examples rewritten to match.
from __future__ import annotations

SYSTEM = "Salesforce instance"

SCHEMA_STRING = """\
The objects available in the Salesforce instance are:
{object_names}

Each object is a SQLite table with the same name. Quote table names in double quotes (for example "Case", "Order"). Timestamps are ISO-8601 text.

## The fields available for the objects along with their descriptions and dependencies are:
{object_fields}
"""

SYSTEM_METADATA = """\
# Additional task context
{system_metadata}
"""

RULE_STRING = """\
Invalid output format! Use the following format: <thought> your thought </thought> and <execute> a valid SQLite SQL query </execute> or <search> keywords </search> or <respond> response to user </respond>
"""

_ROLE = {
    False: "an internal user (i.e., an employee of the same company)",
    True: "a human user (i.e., a customer)",
}

_REACT = """\
You are an expert in Salesforce and you have access to a {system}. You are interacting with the system and {role}.

# Instructions
- You will be provided a question, the system description, and relevant task context.
- Think step by step and interact with the {system} to build SQL queries (SQLite dialect) or text searches as appropriate, to help you answer the question.
- The search action runs a keyword search over names and long text fields (knowledge articles, emails, call and chat transcripts, case descriptions) and returns matching record Ids with snippets.
- Your generation should always be a Thought followed by an Action command and NOTHING ELSE. Generate only one Thought and one Action command.
- DO NOT generate ANY system observation, you will receive this based on your Action command.
- If multiple records are found matching the requirements mentioned, just return a comma-separated string.
- If no record is found matching the requirements mentioned, just return 'None'.

Here is a description of how to use these commands:
## Thought
- A single line of reasoning to process the context and inform the decision making. Do not include any extra lines.
- Format: <thought> your thought </thought>

## Action
- Can be 'execute', 'search' or 'respond'.
- execute, to run one SQLite SQL query on the {system} and return the observation.
- search, to run a keyword search on the {system} and return the matching records.
- respond, to return the final answer of the task to the user.
- Format: <execute> a valid SQLite SQL query </execute> or <search> keywords </search> or <respond> response to user </respond>

# Guidelines
- Always start with a Thought and then proceed with an Action. Generate only one Thought and one Action command at a time.
- Execute SQL queries and searches to understand the {system} that will help you find the answer to the question.
- When you are confident about the answer, submit it with <respond>.
- Always end with a <respond> action containing ONLY the answer, NO full sentences or any explanation.

# Example 1
Question: What is the total number of opportunities?
Output:
<thought> I need to find the total number of opportunities in the system. </thought>
<execute> SELECT COUNT(*) FROM Opportunity </execute>
     (If the observation from the {system} is [{{'COUNT(*)': 100}}], your next step can be)
<thought> I have found the total number of opportunities. </thought>
<respond> 100 </respond> OR <respond> The total number of opportunities is 100 </respond>

# Example 2
Question: Look for the name Joe Smith in the name field of a lead and return the name and phone number.
Output:
<thought> I need to search for the name Joe Smith and find the matching lead. </thought>
<search> Joe Smith </search>
    (If the observation from the {system} is [{{'object': 'Lead', 'Id': '00QXX0000001AAA', 'field': 'Name', 'snippet': '[Joe] [Smith]'}}], your next step can be)
<thought> I found the lead. I need its name and phone number. </thought>
<execute> SELECT FirstName, LastName, Phone FROM Lead WHERE Id = '00QXX0000001AAA' </execute>
    (If the observation from the {system} is [{{'FirstName': 'Joe', 'LastName': 'Smith', 'Phone': '1234567890'}}], your next step can be)
<thought> I have found the name Joe Smith and the phone number. </thought>
<respond> Joe Smith, 1234567890 </respond> OR <respond> The name is Joe Smith and the phone number is 1234567890 </respond>

# Example 3
Question: Find all contacts with the last name Smith and return their names.
Output:
<thought> I need to find all contacts where the last name is Smith and return their names. </thought>
<execute> SELECT FirstName, LastName FROM Contact WHERE LastName = 'Smith' </execute>
    (If the observation from the {system} is [{{'FirstName': 'John', 'LastName': 'Smith'}}, {{'FirstName': 'Jane', 'LastName': 'Smith'}}], your next step can be)
<thought> I have found the names of all contacts where the last name is Smith. I need to return their names as a comma-separated string. </thought>
<respond> John Smith, Jane Smith </respond>


# {system} description
{system_description}
"""


def schema_text(schema_objects: list[dict]) -> str:
    """The schema block, built the way CRMArena's ChatAgent._build_schema does."""
    fields = {o["object"]: "\n".join(f"  - {k}: {v}" for k, v in o["fields"].items() if v is not None)
              for o in schema_objects}
    return SCHEMA_STRING.format(object_names=", ".join(fields),
                                object_fields="\n".join(f"{obj}\n{f}" for obj, f in fields.items()))


def v0_system(task, schema_objects: list[dict], aided: bool = True) -> str:
    """System prompt in the paper's order: role prompt with schema, then required, then optional."""
    prompt = _REACT.format(system=SYSTEM, role=_ROLE[task.external],
                           system_description=schema_text(schema_objects))
    if task.required:
        prompt += SYSTEM_METADATA.format(system_metadata=task.required)
    if aided and task.optional:
        prompt += "\n" + task.optional
    return prompt.strip()


def v0_messages(task, schema_objects: list[dict], aided: bool = True) -> list[dict]:
    return [{"role": "system", "content": v0_system(task, schema_objects, aided)},
            {"role": "user", "content": task.query.strip()}]


def observation_message(text: str) -> dict:
    return {"role": "user", "content": f"{SYSTEM} output: {text}"}


# ---- tools_v1: native tool calls, written for a working app rather than the paper's protocol ----

def _tool(name: str, description: str, props: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {
        "name": name, "description": description, "strict": True,
        "parameters": {"type": "object", "properties": props, "required": required, "additionalProperties": False}}}


TOOLS = [
    _tool("run_sql", "Run one read-only SQLite query on the CRM database and get the rows back.",
          {"query": {"type": "string", "description": "One SQLite SELECT statement."}}, ["query"]),
    _tool("search", "Keyword search over names and long text fields (knowledge articles, emails, call and chat "
          "transcripts, case descriptions). Returns matching record Ids with snippets.",
          {"query": {"type": "string", "description": "Keywords to search for."}}, ["query"]),
    _tool("respond", "Give the final answer to the user. Call this once, when you are confident.",
          {"answer": {"type": "string", "description": "ONLY the answer: an Id, a name, a month, a state code, a "
                      "comma-separated list, or None if nothing matches. No explanation."}}, ["answer"]),
]

TOOLS_NUDGE = "Use one of the tools: run_sql, search, or respond with the final answer."

_TOOLS_SYSTEM = """\
You are an expert Salesforce analyst with access to a {system}. You are helping {role}.

# How to work
- Use the tools to look things up. Do not guess values you can query.
- run_sql runs one SQLite query. Quote table names in double quotes (for example "Case", "Order").
- Timestamps are ISO-8601 text; use julianday() or date() for date arithmetic.
- Relative dates ("past 3 months", "last quarter", "Winter 2023") are relative to the task's "Today's date" below, never the real current date. Use the domain definitions below for quarters and seasons.
- search finds records by keywords in names and long text (knowledge articles, emails, transcripts).
- Follow any policy given in the task context exactly; it defines how metrics are computed.
- When you are confident, call respond with ONLY the answer (an Id, a name, a month, a state code, a comma-separated list, or None). No sentences.

# {system} description
{system_description}
"""


def v1_system(task, schema_objects: list[dict], aided: bool = True) -> str:
    """System prompt for tools_v1: same schema and task context as v0, tool instructions instead of tags."""
    prompt = _TOOLS_SYSTEM.format(system=SYSTEM, role=_ROLE[task.external],
                                  system_description=schema_text(schema_objects))
    if task.required:
        prompt += SYSTEM_METADATA.format(system_metadata=task.required)
    if aided and task.optional:
        prompt += "\n" + task.optional
    return prompt.strip()


def v1_messages(task, schema_objects: list[dict], aided: bool = True) -> list[dict]:
    return [{"role": "system", "content": v1_system(task, schema_objects, aided)},
            {"role": "user", "content": task.query.strip()}]


# ---- actions_v1 / actions_sql_v1: standard CRM actions instead of raw SQL (tools from crm.actions) ----

ACTIONS_NUDGE = "Use one of the tools, or call respond with the final answer."

_ACTIONS_SYSTEM = """\
You are an expert Salesforce analyst with access to a {system}. You are helping {role}.

# How to work
- Use the actions to look things up. Do not guess values you can look up.
- find_by_name turns a name (account, contact, product, user, opportunity...) into a record Id.
- get_record reads every field of one record by its Id.
- query_records lists records of one object that match filters, with the fields you choose.
- summarize_records counts, totals or averages records, grouped by a field (or by month(Field)); use it for "how many", "most", "highest average", "total" questions. avg_days_between gives the average days from one date field to another.
- search_text finds records by keywords in long text (knowledge articles, emails, call and chat transcripts, case descriptions).
- Fields can follow one reference to the parent record: RefField.ParentField, for example AccountId.ShippingState on Case. The references are listed below.
- A date-only filter value (YYYY-MM-DD) compares by day, so CreatedDate <= 2024-01-31 includes the whole of January 31.
- Relative dates ("past 3 months", "last quarter", "Winter 2023") are relative to the task's "Today's date" below, never the real current date. Use the domain definitions below for quarters and seasons.
- Follow any policy given in the task context exactly; it defines how metrics are computed.
- When you are confident, call respond with ONLY the answer (an Id, a name, a month, a state code, a comma-separated list, or None). No sentences.

# {system} description
{system_description}
## References (field -> object it points to; read the parent's fields with RefField.ParentField)
{references}
"""


def v2_system(task, schema_objects: list[dict], org: str, aided: bool = True) -> str:
    """System prompt for the action protocols: action instructions, schema, references, then task context."""
    from crm.actions import references_text  # local: crm.actions imports this module
    prompt = _ACTIONS_SYSTEM.format(system=SYSTEM, role=_ROLE[task.external],
                                    system_description=schema_text(schema_objects),
                                    references=references_text(org))
    if task.required:
        prompt += SYSTEM_METADATA.format(system_metadata=task.required)
    if aided and task.optional:
        prompt += "\n" + task.optional
    return prompt.strip()


def v2_messages(task, schema_objects: list[dict], org: str, aided: bool = True) -> list[dict]:
    return [{"role": "system", "content": v2_system(task, schema_objects, org, aided)},
            {"role": "user", "content": task.query.strip()}]
