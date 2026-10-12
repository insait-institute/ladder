PLAN = (
    "You are planning for a {robot} working on a tabletop, seen from its camera.\n"
    "Task: {task}\n\n"
    "{attempted}"
    "Break the task into the smallest goals that each move ONE object to ONE destination. Name "
    "every object by its colour and kind as it LOOKS in the image (\"the yellow lemon\", \"the blue "
    "bin\"), never by a role or a bare noun, because those names are what will be pointed at in the "
    "image later and several objects may be of the same kind.\n"
    "Only include goals the task actually asks for. An object the task does not mention must be "
    "left where it is, even when it sits among the ones that move.\n"
    "When the task states an order (\"first ... then ...\"), keep the goals in the order the task "
    "gives them, and list a goal TWICE if the task asks for the same object to be moved twice.\n"
    "Otherwise order the goals so that anything placed on an open surface, where a later reach "
    "could knock it off, comes after anything dropped into a container.\n"
    "Leave out any goal the image shows is already met, and return an empty list of goals when "
    "the whole task is done.\n"
    "Give each goal as a sentence, plus the object it moves and where that object goes, named the "
    "same way.\n"
    'Output STRICT JSON and nothing else: '
    '{{"goals": [{{"goal": "...", "object": "...", "destination": "..."}}]}}'
)

ATTEMPTED_HEADER = (
    "Already worked on, with what the executor reported. The image is the evidence of what actually "
    "happened, so ask for one of these again ONLY if the picture shows it did not take -- repeating "
    "a goal that took sends the arm back over it and knocks it over:\n{lines}\n\n"
)

RECHECK = (
    "Look again before stopping. The episode is NOT over, and stopping early throws the rest of it "
    "away: on a two-goal SimplerEnv unit 57% of the steps were spent standing still after the agent "
    "decided it was finished, while a third of the goals were still unmet. Name anything the task "
    "asks for that is not done, however small.\n"
    "Where the destination HIDES what goes into it -- a bin, a box, a pot you cannot see the bottom "
    "of -- the executor's report is the evidence and the picture is not. A goal reported done there "
    "IS done, and asking for it again sends the hand back in to take the object out.\n"
    "Where the destination is OPEN -- a plate, a cloth, a patch of table -- the picture is the "
    "evidence and the report is not. If the object is not lying on it, that goal is NOT done, "
    "whatever was reported: a released object rolls off, and a later reach knocks it off.\n"
    "Return an empty list only when every part of the task is either visible in the picture or "
    "reported done into something you cannot see inside.\n"
)

CHOOSE = (
    "You are controlling a {robot}. The image is the scene right now.\n"
    "Goal: {goal}\n"
    "The gripper is {hand} right now.\n\n"
    "Available tools:\n{catalog}\n\n"
    "{history}"
    "Choose ONE tool call that makes progress on this goal. pick_and_place is the cheap path and "
    "handles most goals in one call. When it reports something the hand could not do, do not repeat "
    "it: the tools below it are yours -- measure the thing, then take hold of it where YOU judge the "
    "jaws will close, or push it there along the surface when it is too wide to hold at all, and "
    "place it yourself. run_code drives the same API when a goal needs something none of the tools "
    "expresses.\n"
    "If a previous attempt failed, do not repeat it unchanged: name the OBJECT differently, since "
    "that name is what gets pointed at in the image and it may not match what is visible. The "
    "destination is what the goal asks for -- describe it differently if it was not found, but "
    "never put the object somewhere else.\n"
    'Output STRICT JSON and nothing else: {{"tool": "<name>", "args": {{...}}}}'
)

HISTORY_HEADER = "Already tried on this goal:\n{lines}\n\n"

BOX = (
    "Point out {name} in this image.\n"
    "Draw it TIGHTLY around {name} alone. Anything lying beside or touching it must be OUTSIDE the "
    "box, even where that means cutting the box in close.\n"
    "Then add one box for EVERY other object lying near it, each with its own label, so {name} can "
    "be told apart from them.\n"
    "A name can say WHICH one it means by where it lies or what it is beside. Box the one it points "
    "at, and box the others of the same kind as well, so they can be told apart. Never reply with "
    "an empty list because the wording is not an exact name.\n"
    "Reply with ONLY a JSON list of boxes, each [ymin, xmin, ymax, xmax] normalised to 0-1000:\n"
    '[{{"box_2d": [y0, x0, y1, x1], "label": "..."}}]\n'
    "Return an empty list [] only if nothing of that kind is in the picture at all."
)
