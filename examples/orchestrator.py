"""A LADDER orchestrator: it decides what the suite's low-level policy should do next, in words.

    export VLM_BASE_URL=<OpenAI-compatible endpoint> VLM_API_KEY=<key> VLM_MODEL=<model>
    scripts/eval.sh --suite simplerenv_long --orchestrator examples/orchestrator.py:PlanThenSwitch

The low-level policy is the Bridge policy on the WidowX suites and pi0.5 on RoboLab. It is good at
one short instruction ("put the carrot on the plate") and bad at long ones, so an orchestrator turns
the task into subgoals and hands them over one at a time.

One object is made per episode, and instruction() is called every control step with what a real
robot would see. Return the instruction the policy should act on now.
"""

import base64
import io
import json
import os

import openai
import PIL.Image

client = openai.OpenAI(base_url=os.environ["VLM_BASE_URL"], api_key=os.environ["VLM_API_KEY"])
MODEL = os.environ.get("VLM_MODEL", "gemini-3.7-flash")

PLAN_PROMPT = """You see a robot's workspace. Split the task into the pick-and-place steps it needs,
one object per step, in order, each as a short instruction like "put the carrot on the plate".
Task: {task}
Answer with a JSON list of strings and nothing else."""


def ask(prompt: str, image) -> str:
    """One question about one image to the vision-language model."""
    buffer = io.BytesIO()
    PIL.Image.fromarray(image).save(buffer, format="JPEG")
    url = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()
    reply = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": url}},
                ],
            }
        ],
    )
    return reply.choices[0].message.content or ""


def parse_json(answer: str):
    """The JSON in a model's answer, which may come wrapped in a ```json fence."""
    return json.loads(answer.strip().removeprefix("```json").strip("`"))


class PlanThenSwitch:
    """Plan once from the first image, then move to the next subgoal each time the gripper opens
    after holding something (one pick and place done)."""

    def __init__(self) -> None:
        self.plan = None  # the subgoals; LADDER records them with the episode
        self.current = 0
        self.holding = False

    def instruction(self, obs: dict) -> str:
        if self.plan is None:
            image = next(iter(obs["images"].values()))
            try:
                self.plan = [
                    str(step) for step in parse_json(ask(PLAN_PROMPT.format(task=obs["instruction"]), image))
                ]
            except (ValueError, TypeError):
                self.plan = []
            self.plan = self.plan or [obs["instruction"]]  # no usable plan: the task as it is

        closed = obs["gripper"] < 0.5
        if self.holding and not closed:
            self.current = min(self.current + 1, len(self.plan) - 1)
        self.holding = closed
        return self.plan[self.current]
