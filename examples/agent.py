"""A LADDER agent: a program that drives the arm itself, with no low-level policy.

    export VLM_BASE_URL=<OpenAI-compatible endpoint> VLM_API_KEY=<key> VLM_MODEL=<model>
    scripts/eval.sh --suite simplerenv_long --agent examples/agent.py:PointAndPlace

One object is made per episode and its run(arm) is called once. Write it as straight-line code:
every arm call blocks until the arm has done it, and the simulator waits while the program thinks.
The episode ends when its step budget is spent or the task is done, wherever the program is.

The arm, in metres in the robot base frame:
    arm.task                        the instruction
    arm.observation                 images, eef_pose, gripper, depth (see examples/README.md)
    arm.move_to([x, y, z])          move the fingertips there in a straight line, hand pointing down
    arm.close() / arm.open()        close (True if something is held) / open the gripper
    arm.locate(camera, x, y)        the 3D point seen at a pixel of that camera, from its depth
    arm.step(eef_pose, gripper)     one raw control step, if you want to command poses yourself
"""

import base64
import io
import json
import os

import numpy as np
import openai
import PIL.Image

client = openai.OpenAI(base_url=os.environ["VLM_BASE_URL"], api_key=os.environ["VLM_API_KEY"])
MODEL = os.environ.get("VLM_MODEL", "gemini-3.7-flash")

POINT_PROMPT = """You see a robot's workspace in an image {width} pixels wide and {height} high.
Task: {task}
List the pick-and-place steps it needs, one object per step, in order. For each, give the pixel
on the object to pick and the pixel on the place to put it, as [x, y] with x to the right and y down.
Answer with JSON only: a list of objects, each with the keys "pick" and "place"."""

ABOVE = np.array([0.0, 0.0, 0.08])  # approach and retreat height, metres


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


class PointAndPlace:
    """Ask a VLM once where each object and its destination are, then pick and place them in turn."""

    def run(self, arm) -> None:
        camera = next(iter(arm.observation["depth"]))  # the camera that has depth
        image = arm.observation["images"][camera]
        answer = ask(POINT_PROMPT.format(width=image.shape[1], height=image.shape[0], task=arm.task), image)
        try:
            steps = parse_json(answer)
        except ValueError:
            return  # no usable answer: the arm stays where it is

        for step in steps:
            pick, place = arm.locate(camera, *step["pick"]), arm.locate(camera, *step["place"])
            if pick is None or place is None:
                continue  # no depth at that pixel
            arm.open()
            arm.move_to(pick + ABOVE)
            arm.move_to(pick)
            arm.close()
            arm.move_to(pick + ABOVE)
            arm.move_to(place + ABOVE)
            arm.open()
