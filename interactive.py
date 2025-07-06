import re
import json
import asyncio
from typing import Optional, List, Dict, Any
from models import Plan, Task
from plan_generator import PlanGenerator
from config import logger

def print_plan(plan: Plan) -> None:
    print("\nCurrent Research Plan Tasks:")
    for task in plan.tasks:
        print(f"{task.id}. {task.description}")

def keep_history(plan: Plan, instruction: Optional[str], history: List[Dict[str, Any]]) -> None:
    snapshot = [t.dict() for t in plan.tasks]
    history.append({
        'instruction': instruction or 'initial generation',
        'tasks': snapshot
    })

async def interactive_plan_loop(query: str) -> Plan:
    pg = PlanGenerator()
    original_query = query
    user_history: List[str] = []
    llm_history: List[Dict[str, Any]] = []

    plan = await pg.generate(original_query)
    keep_history(plan, None, llm_history)

    while True:
        print_plan(plan)
        resp = input("\nAre these tasks appropriate? (y/n): ").strip().lower()
        if resp == 'y':
            break

        instruction = input("Describe what you'd like to change: ").strip()
        if not instruction:
            print("No instruction provided, please try again.")
            continue
        user_history.append(instruction)

        tasks_context = json.dumps(
            [{"index": idx+1, **t.dict()} for idx, t in enumerate(plan.tasks)],
            indent=2
        )
        user_context = "\n".join(f"{i+1}. {instr}" for i, instr in enumerate(user_history))
        llm_context = json.dumps(llm_history, indent=2)

        refine_prompt = (
            f"You are refining a research plan for query: '{original_query}'.\n"
            f"User refinement history:\n{user_context}\n"
            f"LLM action history:\n{llm_context}\n"
            f"Current tasks:\n{tasks_context}\n"
            f"Apply the latest instruction: '{instruction}'.\n"
            "Output only a JSON object matching the Plan schema with updated tasks."
        )

        try:
            raw = pg.client.chat(refine_prompt)
            clean = raw.strip()
            match = re.search(r"\{.*\}", clean, re.DOTALL)
            if not match:
                raise ValueError("No JSON found in LLM response")
            plan = Plan.parse_raw(match.group(0))
            logger.info("Refined plan with %d tasks", len(plan.tasks))
            keep_history(plan, instruction, llm_history)
        except Exception as e:
            logger.error("Refinement failed: %s", e)
            print("Automatic refinement failed; please edit manually.")
            action = input("(e)dit a task or (a)dd a new task? ").strip().lower()
            if action == 'e':
                try:
                    tid = int(input("Task ID to edit: ").strip())
                    desc = input("New description: ").strip()
                    for t in plan.tasks:
                        if t.id == tid:
                            t.description = desc
                            print(f"Task {tid} updated.")
                            break
                    else:
                        print(f"No task with ID {tid}.")
                except ValueError:
                    print("Invalid ID.")
            elif action == 'a':
                desc = input("Description for new task: ").strip()
                new_id = max((t.id for t in plan.tasks), default=0) + 1
                plan.tasks.append(Task(id=new_id, description=desc))
                print(f"Task {new_id} added.")
            else:
                print("Invalid option.")

    return plan

async def f2() -> Optional[Plan]:
    query = input("Enter your research query: ").strip()
    try:
        return await interactive_plan_loop(query)
    except Exception as e:
        logger.error("Error during plan loop: %s", e)
        print("An error occurred. Please try again later.")
        return None
