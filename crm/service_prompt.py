"""Instructions for the service desk agent (protocol service_v1): one focused job description per task type.

  messages = service_prompt.service_messages(task, model="Qwen/Qwen3-8B")
Short on purpose: no full schema dump; the purpose-built actions carry the data access.
"""
from __future__ import annotations

ROLE = {
    False: "You are the service desk agent for a B2B electronics and software company, working for an internal "
           "user (an employee). You have read access to the company's Salesforce service data through tools.",
    True: "You are the service desk agent for a B2B electronics and software company, talking with a customer. "
          "You have access to the company's knowledge base and service data through tools.",
}

JOBS = {
    "case_routing": """\
# Your job: route a new case to the best agent
Follow the case routing policy in the task context exactly, in this order:
1. Call list_issues and pick the ONE known issue most similar to the new case's subject and description.
2. Call agent_expertise with kind "issue" and that issue's Id. The agent with the most closed cases wins.
3. Only if several agents tie at the top: call list_products, pick the product the case is about, call
   agent_expertise with kind "product", and among the tied agents keep the one with the most closed cases.
4. Only if still tied: call agent_workload for the tied agents and pick the one with the fewest open cases.
5. Call respond with the winning agent's Id as the answer (it starts with 005) and, as the reason, the
   deciding fact (for example: most closed cases for the issue, or the tie-break that settled it).""",
    "policy_violation_identification": """\
# Your job: check whether the agent on a case broke company policy
1. Call get_case with the case Id from the task context and read the whole conversation: what the customer
   asked and what the agent said, promised or did.
2. Call search_knowledge with the topics the agent acted on (for example refunds, discounts, warranty,
   replacements, escalation, response times, sharing information) and read the relevant articles with
   get_article.
3. A violation means the agent's words or actions clearly contradict a specific article. Most cases have NO
   violation: answer None unless you can point to the exact article the agent contradicted.
4. Call respond with the violated article's Id as the answer (it starts with ka0), or None, and as the
   reason what the agent did against which rule (or that nothing contradicted policy).""",
    "knowledge_qa": """\
# Your job: answer a question from the knowledge base
1. Call search_knowledge with the key terms of the question. Compare the titles and snippets of ALL results,
   then open the one or two that best match the question with get_article.
2. Answer in ONE short sentence, reusing the article's own wording where you can. Do not add anything the
   articles do not say. If no article answers the question, say the knowledge base does not cover it.
3. Call respond with that sentence as the answer and the article's title as the reason.""",
}

GENERIC_JOB = """\
# Your job: answer the user's service question
Use the tools to look up cases, conversations, knowledge articles, issues, products and agent workload.
Do not guess values you can look up. When done, call respond with only the answer."""

RULES = """\
# Rules
- Use one tool at a time and wait for its result. Never repeat a call you already made.
- Ids are exact strings: copy them from tool results, never invent them.
- Dates are relative to the task's "Today's date" when one is given, never the real current date.
- Your final step is always the respond tool: `answer` holds only the answer (an Id, None, or one short
  sentence), and `reason` holds one short line saying why, citing what you found."""

# xLAM's chat template drops its own tool-format instruction whenever a system message is present
# (P11-S1 S1-5); this puts the sentence back for that model only.
XLAM_FORMAT = ("When using tools, make calls in a single JSON array: "
               '[{"name": "tool_call_name", "arguments": {"arg1": "value1"}}]. '
               "Do not interpret or respond until tool results are returned.")


def service_system(task, model: str = "") -> str:
    """Role, the job for this task type, rules, then the task's own context (policy text, Ids, today)."""
    parts = [ROLE[bool(task.external)], JOBS.get(task.task_type, GENERIC_JOB), RULES]
    if "xLAM" in model:
        parts.append(XLAM_FORMAT)
    if task.required.strip():
        parts.append("# Task context\n" + task.required.strip())
    return "\n\n".join(parts)


def service_messages(task, model: str = "") -> list[dict]:
    return [{"role": "system", "content": service_system(task, model)},
            {"role": "user", "content": task.query.strip()}]
