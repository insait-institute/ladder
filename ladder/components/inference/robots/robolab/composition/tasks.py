from ladder.components.inference.robots.robolab.composition.factory import make_task
from ladder.components.inference.robots.robolab.composition.specs import SPECS, task_class_name

# One RoboLab Task subclass per composition spec, all in this module: RoboLab discovers tasks by
# scanning a module for Task subclasses, and env_server addresses a single one as
# "<this file>::<ClassName>" (see specs.task_target). Built from SPECS rather than written out so a
# spec can never be added without its task -- which is what the generated per-spec stub files, one
# import line each, existed to guarantee.
for _spec_name in SPECS:
    globals()[task_class_name(_spec_name)] = make_task(_spec_name)
