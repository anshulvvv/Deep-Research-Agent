import re
import json
from gemini_client import GeminiClient
from models import Plan
from config import logger

class PlanGenerator:
    """Generates a flat research plan from a user query via function-calling."""
    def __init__(self):
        self.client = GeminiClient()

    async def generate(self, query: str) -> Plan:
        plan_schema = Plan.schema_json(indent=2)
        prompt = f"""
You are a research planner. Given a single-line query, output _only_ a JSON object
give at max 5 tasks please,
that matches this exact JSON schema (no markdown, no backticks, no extra keys):

{plan_schema}

Here’s the query:

{json.dumps({"query": query})}
""".strip()

        raw = self.client.chat(prompt)
        clean = raw.strip()
        match = re.search(r"\{.*\}", clean, re.DOTALL)
        if not match:
            logger.error("Could not find JSON object in Gemini response:\n%s", raw)
            raise ValueError("Invalid JSON from Gemini for plan generation")

        json_blob = match.group(0)
        plan = Plan.parse_raw(json_blob)
        logger.info("Generated plan with %d tasks", len(plan.tasks))
        return plan
