import math
from typing import Dict, List, Optional, Tuple

_TABLE = (0.197, 0.897, -0.500, 0.500)  # xmin, xmax, ymin, ymax of the tabletop
# Bare table that must stay visible beyond an object's own footprint, even at worst-case jitter. An
# object sitting at the lip is both very hard to grasp (no room for the fingers to come round it) and
# a physics hazard -- a nudge tips it off the table and the episode is unwinnable from that moment.
_EDGE_KEEP = 0.05
_RECEP_MARGIN = 0.02  # keep free objects this far outside a receptacle footprint beyond (radius + jitter)
_PAIR_MARGIN = 0.01  # slack beyond touching between two objects at worst-case jitter
_RIM_KEEP = 0.02  # bare receptacle surface beyond an object resting ON it (transfer sources)
_OBJ_RECT = (0.33, 0.72, -0.42, 0.42)  # arm-reachable band; the front ~0.18 m sits behind the mount
_WORKSPACE_CENTER = (0.50, 0.0)
_GRID_NX, _GRID_NY = 33, 41
RECEP_ANCHOR_1 = (0.70, 0.0)  # single receptacle: deep centre
RECEP_ANCHOR_2A = (0.72, 0.26)  # split/transfer/relay: deep, pushed to the back edge to free the band
RECEP_ANCHOR_2B = (0.72, -0.26)
# How far from the base a GRASPABLE may start: grasp success drops sharply for objects starting much
# beyond ~0.63 m. Applies to start positions only; placing INTO a bin at 0.70 m is fine. 0.62 not
# 0.65 because the greedy placement is not monotone in the bound: 0.65 cannot seat ten objects at any
# ladder jitter, 0.62 seats them at 0.02.
REACH_MAX = 0.62
# Transfer and relay put graspables on a receptacle rather than the table, so the SOURCE has to obey
# the same bound: at RECEP_ANCHOR_2A the objects would rest at 0.72-0.80 m, out of comfortable reach. The
# destination stays deep.
SOURCE_ANCHOR = (0.45, 0.26)
# Jitter is DERIVED per task: each scene takes the largest value here for which a provably safe layout
# exists (nothing leaves the table, enters a receptacle or overlaps a neighbour at worst-case jitter).
JITTER_LADDER = (0.07, 0.06, 0.05, 0.04, 0.03, 0.02)

# Episode budget (s) keyed by PICK-PLACE OPERATIONS, taken from RoboLab's own medians over the 95
# benchmark tasks with a placement conditional. Forward-filled so more work never means less time; the
# tail past nine extrapolates at the ~33 s/op their large tasks settle to.
ROBOLAB_BUDGET_S = {
    1: 60,
    2: 90,
    3: 180,
    4: 180,
    5: 180,
    6: 180,
    7: 240,
    8: 270,
    9: 300,
    10: 330,
    11: 330,
    12: 330,
}


def dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _candidate_cells(
    radius: float, jitter: float, reach_max: Optional[float] = None
) -> List[Tuple[float, float]]:
    """Candidate centres for one object: grid points inside the arm-reachable band that are also far
    enough from every table edge that neither the jitter nor the object's own footprint can bring it
    within ``_EDGE_KEEP`` of the lip, and (when ``reach_max`` is given) within that distance of the
    base even at worst-case jitter. Empty when the object cannot be placed safely at all."""
    m = radius + jitter + _EDGE_KEEP
    xmin, xmax = max(_TABLE[0] + m, _OBJ_RECT[0]), min(_TABLE[1] - m, _OBJ_RECT[1])
    ymin, ymax = max(_TABLE[2] + m, _OBJ_RECT[2]), min(_TABLE[3] - m, _OBJ_RECT[3])
    if xmax <= xmin or ymax <= ymin:
        return []
    xs = [xmin + (xmax - xmin) * i / (_GRID_NX - 1) for i in range(_GRID_NX)]
    ys = [ymin + (ymax - ymin) * j / (_GRID_NY - 1) for j in range(_GRID_NY)]
    cells = [(x, y) for x in xs for y in ys]
    if reach_max is None:
        return cells
    return [c for c in cells if math.hypot(abs(c[0]) + jitter, abs(c[1]) + jitter) <= reach_max]


def _clear_of_receptacles(
    c: Tuple[float, float], radius: float, jitter: float, reserved: List[Tuple[float, float, float, float]]
) -> bool:
    """The object never overlaps a (pinned) receptacle's rectangular footprint, even at the worst jitter:
    reserved entries are (x, y, half_x, half_y). A rectangular keep-out (not a circle) is what lets
    objects sit beside a wide bin on the open table instead of being excluded from the whole back."""
    for rx, ry, hx, hy in reserved:
        if (
            abs(c[0] - rx) < hx + radius + jitter + _RECEP_MARGIN
            and abs(c[1] - ry) < hy + radius + jitter + _RECEP_MARGIN
        ):
            return False
    return True


def _scatter(
    radii: List[float],
    reserved: List[Tuple[float, float, float, float]],
    jitter: float,
    reach_max: Optional[float] = None,
) -> Optional[List[Tuple[float, float]]]:
    """Authored (x, y) per object (indexed as ``radii``) for which NO per-episode jitter can push an
    object off the table, under a pinned receptacle, or into a neighbour. Objects are placed
    largest-first (the banana needs the most room) by farthest-point selection -- start nearest the
    workspace centre, then repeatedly add the candidate maximising the minimum distance to those
    already chosen -- among cells satisfying every hard constraint for that object's own radius.

    The pairwise bound is what stops the scene from tearing itself apart: two objects can close by
    ``2 * jitter``, so authoring them merely "far apart" (maximising the minimum distance without
    asserting one) lets high-N scenes spawn interpenetrating, and the simulator resolves the overlap
    by ejecting an object, sometimes off the table.

    Returns None when the scene is over-full at this jitter, so callers can retry lower."""
    placed: Dict[int, Tuple[float, float]] = {}
    for oi in sorted(range(len(radii)), key=lambda i: -radii[i]):
        r = radii[oi]
        cand = [
            c
            for c in _candidate_cells(r, jitter, reach_max)
            if _clear_of_receptacles(c, r, jitter, reserved)
            and all(dist(c, p) >= r + radii[j] + 2 * jitter + _PAIR_MARGIN for j, p in placed.items())
        ]
        if not cand:
            return None
        if not placed:
            placed[oi] = min(cand, key=lambda c: dist(c, _WORKSPACE_CENTER))
        else:
            placed[oi] = max(cand, key=lambda c: min(dist(c, p) for p in placed.values()))
    return [placed[i] for i in range(len(radii))]


def fit_scatter(
    radii: List[float],
    reserved: List[Tuple[float, float, float, float]],
    reach_max: Optional[float] = None,
) -> Tuple[float, List[Tuple[float, float]]]:
    """The most per-episode variation this scene can safely carry: the largest jitter on the ladder
    that still admits a provably safe layout, together with that layout."""
    for jitter in JITTER_LADDER:
        positions = _scatter(radii, reserved, jitter, reach_max)
        if positions is not None:
            return jitter, positions
    raise ValueError(
        f"no safe layout for {len(radii)} objects (max radius {max(radii):.3f}) clear of "
        f"{len(reserved)} receptacle(s) within reach {reach_max} at any jitter down to "
        f"{JITTER_LADDER[-1]}"
    )


def pull_into_reach(
    anchor: Tuple[float, float], offsets: List[Tuple[float, float]], jitter: float, reach_max: Optional[float]
) -> Tuple[float, float]:
    """Slide a source receptacle straight toward the base until every object resting on it obeys
    ``reach_max`` at worst-case jitter. Objects on a receptacle are laid out by ``fit_on_receptacle``,
    which knows nothing about reach, so a hand-picked anchor would otherwise silently decide how far
    out they start."""
    if reach_max is None:
        return anchor
    scale = 1.0
    for _ in range(200):
        ax, ay = anchor[0] * scale, anchor[1] * scale
        worst = max(math.hypot(abs(ax + dx) + jitter, abs(ay + dy) + jitter) for dx, dy in offsets)
        if worst <= reach_max:
            return (ax, ay)
        scale -= 0.005
    raise ValueError(f"cannot bring {anchor} within reach {reach_max}")


def fit_on_receptacle(
    radii: List[float], footprint: Tuple[float, float]
) -> Tuple[float, List[Tuple[float, float]]]:
    """Same idea for objects that start resting ON a receptacle (the transfer sources): offsets from
    the receptacle centre, on a concentric ring, plus the largest jitter for which every object stays
    ``_RIM_KEEP`` inside the rim and no two can overlap. Randomizing these within the source, rather
    than pinning them, keeps the source intact while still giving each episode a fresh draw."""
    n = len(radii)
    for jitter in JITTER_LADDER:
        usable = min(footprint) - _RIM_KEEP - jitter - max(radii)
        if usable <= 0:
            continue
        if n == 1:
            return jitter, [(0.0, 0.0)]
        need = max(radii[i] + radii[j] for i in range(n) for j in range(n) if i != j)
        ring = (need + 2 * jitter + _PAIR_MARGIN) / (2 * math.sin(math.pi / n))
        if ring <= usable:
            return jitter, [
                (ring * math.cos(2 * math.pi * k / n), ring * math.sin(2 * math.pi * k / n)) for k in range(n)
            ]
    raise ValueError(f"cannot fit {n} objects on a {footprint} receptacle with any ladder jitter")
