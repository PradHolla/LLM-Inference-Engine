"""Prompt text for the chat agent: system prompt, planner and summary instructions, search block.

  from app import prompts; prompts.system_prompt("2026-09-23")
"""
from __future__ import annotations

SYSTEM_PROMPT = """You are a helpful assistant running on a self-hosted Qwen3-8B.

Answer the question directly. Default to 150-300 words; go longer only when the user asks
for depth or the task genuinely needs it (code, step-by-step work). Use a list or table only
when it makes the answer easier to read, not by habit. Do not add a closing summary that
repeats the answer.

The conversation comes first. When search results are provided, use them only where they
are relevant to what the user is asking now, and cite them inline as [1], [2] matching their
numbers. If the results do not cover the question, say so plainly and answer from general
knowledge, marked as unverified. Never invent figures, tickers, dates or names.

Today's date is {date}."""

PLANNER_INSTRUCTION = """Before answering the user's latest message above, decide how to handle it. Reply with JSON only, in exactly this form:
{"search": true or false, "queries": ["..."], "think": true or false}

search: true when the answer depends on current or specific facts (prices, news, releases, people, figures, anything dated), or the user asks to look something up. false for chit-chat, rewording, maths, code, opinions, or follow-ups the conversation already answers.
queries: when search is true, one to three web search queries. Each must stand on its own: name the entities and resolve words like "it", "those" or "that" from the conversation. Split a multi-part question into up to three queries. Use [] when search is false.
think: true for multi-step reasoning, calculation, comparison or planning. false for greetings, simple facts and formatting."""

SUMMARY_INSTRUCTION = """Summarise the conversation above, including any earlier summary, for your own later reference. Keep every name, number, date, decision, and the user's stated preferences and open questions. Write plain prose of at most 200 words. Reply with the summary only."""


def system_prompt(date: str) -> str:
    """Static text with the date on the last line, so a new day invalidates only what follows."""
    return SYSTEM_PROMPT.format(date=date)


def summary_message(text: str) -> dict:
    return {"role": "system", "content": f"Summary of the earlier conversation:\n{text}"}


def _query_label(queries: list[str]) -> str:
    return "; ".join(f'"{query}"' for query in queries)


def search_block(queries: list[str], rendered: str) -> str:
    """Label gateway.search.render_block output with the queries that produced it."""
    return f"Web search for {_query_label(queries)}.\n{rendered}"


def empty_search(queries: list[str]) -> str:
    return (f"Web search for {_query_label(queries)} found no usable sources. Answer from "
            "general knowledge and say plainly that you could not verify it.")
