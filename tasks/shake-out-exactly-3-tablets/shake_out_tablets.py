"""On the station table: an amber 150 cc pill bottle (50 mm across, 104 mm tall) with no cap, standing upright in front
of the right arm, holding 30 white tablets (10 mm across, 4 mm thick).
"""

from __future__ import annotations

import math
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from partkit import on_reset, on_step, part, write_pose  # noqa: E402
from yam_arena.task_file import TABLE_TOP  # noqa: E402

SCENE = "station"
TOP = TABLE_TOP[SCENE]

AMBER = (0.60, 0.33, 0.08)

# -- the bottle, in its own frame: origin at the centre of its base, z up its axis ---------------------------------------
# A 150 cc amber PET packer bottle with a 38-400 neck (TricorBraun's spec: 1.98 in across, 4.09 in tall, 19.5 g). The walls
# are rings of thin boxes (a 32-sided prism, within 0.1 mm of round), since a moving body collides only as convex pieces.
OUTER_R = 0.02515
HEIGHT = 0.1039
WALL = 0.003  # a real wall is about 0.5 mm; 3 mm (with CCD on the tablets) stops small, fast tablets passing through it
INNER_R = OUTER_R - WALL
MOUTH_R = 0.0155  # a 31 mm bore; with the 3 mm wall the neck is 37 mm across (a 38-400 thread is 38 mm)
NECK_H = 0.010  # a 38-400 finish is about 10 mm tall
SHOULDER_H = 0.009
BODY_TOP = HEIGHT - NECK_H - SHOULDER_H
NECK_BOTTOM = HEIGHT - NECK_H
SIDES = 32  # more facets → smaller seams between wall boxes
BOTTLE_KG = 0.0195  # the real empty bottle
BOTTLE_XY = (0.30, -0.20)
# Overlap adjacent wall boxes so PhysX cannot wedge a tablet into a seam (visual "stuck in the wall").
WALL_OVERLAP = 1.35


def inner_radius(z: float) -> float:
    """The bottle's inside radius at height z (m) in its own frame."""
    if z <= BODY_TOP:
        return INNER_R
    if z >= NECK_BOTTOM:
        return MOUTH_R
    return INNER_R + (MOUTH_R - INNER_R) * (z - BODY_TOP) / SHOULDER_H


def _ring(r0: float, z0: float, r1: float, z1: float) -> list:
    """A ring of SIDES boxes, WALL thick, whose inner faces run from radius r0 at height z0 to r1 at z1."""
    tilt = math.atan2(r1 - r0, z1 - z0)  # the wall's lean from vertical; negative leans in
    length = math.hypot(r1 - r0, z1 - z0)
    # Chord width plus WALL_OVERLAP so neighbouring facets overlap and leave no tunnel seam.
    width = WALL_OVERLAP * 2 * (max(r0, r1) + WALL) * math.tan(math.pi / SIDES)
    mid_r = (r0 + r1) / 2 + WALL / 2 * math.cos(tilt)
    mid_z = (z0 + z1) / 2 - WALL / 2 * math.sin(tilt)
    boxes = []
    for k in range(SIDES):
        a = 2 * math.pi * k / SIDES
        cy, sy, ct, st = math.cos(a / 2), math.sin(a / 2), math.cos(tilt / 2), math.sin(tilt / 2)
        rot = (cy * ct, -sy * st, cy * st, sy * ct)  # (w, x, y, z): turn to angle a about z, then lean about y
        boxes.append(((mid_r * math.cos(a), mid_r * math.sin(a), mid_z), (WALL, width, length), AMBER, rot))
    return boxes


BOTTLE_BOXES = (
    _ring(INNER_R, 0.0, INNER_R, BODY_TOP + WALL)  # body; runs on past the shoulder's foot to close the joint outside
    + _ring(INNER_R, BODY_TOP, MOUTH_R, NECK_BOTTOM)  # shoulder
    + _ring(MOUTH_R, NECK_BOTTOM, MOUTH_R, HEIGHT)  # neck
)
BOTTLE_BASE = ((0.0, 0.0, WALL / 2), OUTER_R, WALL, AMBER)

BOTTLE = part("bottle", pos=(*BOTTLE_XY, TOP + 0.0005), boxes=BOTTLE_BOXES, cylinders=(BOTTLE_BASE,), mass=BOTTLE_KG,
              color=AMBER, friction=1.2, restitution=0.0, max_depenetration=0.5)

# -- the tablets: 30 round, flat 3D-printed PLA tablets, 10 mm across and 4 mm thick (0.39 g) ------------------------------
TABLET_R, TABLET_H, TABLET_KG = 0.005, 0.004, 0.00039
WHITE = (0.95, 0.95, 0.92)
# They start lying flat inside the bottle in 5 layers of 6, 2 mm apart, each layer turned 30 degrees from the one below,
# and settle into a heap on the base as the trial starts.
LAYERS, PER_LAYER, LAYER_GAP = 5, 6, 0.002
RING_R = INNER_R - TABLET_R - 0.002  # keep a 2 mm clearance from the inner wall


def _tablet_start(i: int) -> tuple[float, float, float]:
    """Tablet i's starting centre, in the bottle's frame."""
    layer, k = divmod(i, PER_LAYER)
    a = 2 * math.pi * k / PER_LAYER + math.radians(30) * layer
    z = WALL + LAYER_GAP + TABLET_H / 2 + layer * (TABLET_H + LAYER_GAP)
    return RING_R * math.cos(a), RING_R * math.sin(a), z


# Kit parts rather than plain cylinders, for PhysX settings plain objects lack: a cap on how fast an overlap is pushed
# apart (else a tablet caught in a wall is fired out), a speed cap applied on every physics substep (PhysX can fire a
# tablet that clipped a wall at 50-200 m/s; a real shake stays under ~2 m/s), and swept CCD so a 4 mm tablet cannot
# pass through a 3 mm wall or the table between substeps.
MAX_TABLET_M_S = 1.5
TABLET_FRICTION = 0.3  # PLA on PET; "min" combine so it wins over the bottle's grippy 1.2
MAX_TABLET_DEG_S = 3600.0
TABLETS = [
    part(f"tablet_{i + 1:02d}", pos=tuple(b + t for b, t in zip(BOTTLE.pos, _tablet_start(i))), boxes=(),
         cylinders=(((0.0, 0.0, 0.0), TABLET_R, TABLET_H, WHITE),), mass=TABLET_KG, color=WHITE,
         friction=TABLET_FRICTION, friction_combine="min",
         contact_offset=0.001, linear_damping=0.5, angular_damping=1.0, max_depenetration=0.3, restitution=0.0,
         max_linear_velocity=MAX_TABLET_M_S, max_angular_velocity=MAX_TABLET_DEG_S, ccd=True)
    for i in range(LAYERS * PER_LAYER)
]
OBJECTS = [BOTTLE, *TABLETS]

N_OUT = 3  # how many tablets to shake out
SCORE_OVER = "end"
# Bottle upright / on its side: angle of its axis from vertical (degrees).
UPRIGHT_DEG = 20.0
SIDE_DEG = 70.0
# Base on the table: lowest point within this of the table top (m).
ON_TABLE_MM = 0.008


# Tablets that have come to rest on the table outside the bottle this trial. They stay counted even if pushed back in
# (PharmacyBench: "any tablets that the robot manually puts back in are still counted").
_LANDED: set = set()
# A tablet's centre is at most this far above the table top when it lies on it (flat: 2 mm; on its edge: 5 mm).
LANDED_ABOVE_TABLE = TABLET_R + 0.003


def _in_bottle(bottle, pos) -> bool:
    """The tablet centre is inside the bottle's free volume (in the bottle's frame, so also when it is tipped)."""
    x, y, z = bottle.to_local(pos)
    return 0.0 < z < HEIGHT and math.hypot(x, y) < inner_radius(z)


def tablets_out(objects) -> int:
    """How many tablets have left the bottle: centre outside its inside volume now, or landed on the table earlier in
    the trial (put back or not). A tablet that falls through the table is parked far below it and counts as out."""
    bottle = objects["bottle"]
    return sum(1 for t in TABLETS if t.name in _LANDED or not _in_bottle(bottle, objects[t.name].pos))


def _bottle_tilt_deg(bottle) -> float:
    """Angle between the bottle's axis and the vertical."""
    return math.degrees(math.acos(max(-1.0, min(1.0, float(bottle.rotation()[2, 2])))))


def _bottle_on_table(bottle) -> bool:
    """The bottle's base is on the table (lowest point near the table top) and still."""
    return float(bottle.low[2]) <= TOP + ON_TABLE_MM and bottle.resting()


def count_credit(n: int) -> float:
    """c in 0.75·c + 0.25·u: 1 if exactly N_OUT, 0.5 if off by one, else 0."""
    d = abs(n - N_OUT)
    if d == 0:
        return 1.0
    if d == 1:
        return 0.5
    return 0.0


def upright_credit(objects) -> float:
    """u: 1 upright on the table, 0.5 on its side on the table, 0 if dropped / still held. Only if ≥1 tablet out."""
    if tablets_out(objects) < 1:
        return 0.0
    bottle = objects["bottle"]
    if not _bottle_on_table(bottle):
        return 0.0
    tilt = _bottle_tilt_deg(bottle)
    if tilt <= UPRIGHT_DEG:
        return 1.0
    if tilt >= SIDE_DEG:
        return 0.5
    return 0.0  # leaning between upright and side: no upright credit


def success(objects) -> bool:
    """Exactly N_OUT tablets out and the bottle set down upright on the table; must hold for 1 s."""
    return tablets_out(objects) == N_OUT and upright_credit(objects) == 1.0


def score(objects) -> float:
    """PharmacyBench task 3: 0.75·c + 0.25·u. End state only."""
    return 0.75 * count_credit(tablets_out(objects)) + 0.25 * upright_credit(objects)


@on_step
def _track_tablets(environment) -> None:
    """Record tablets that land on the table outside the bottle, and park any that fall through the table far under
    the world (a safety net; they count as out)."""
    from yam_arena.scene import object_states

    states = object_states(environment, ["bottle", *(t.name for t in TABLETS)])
    bottle = states["bottle"]
    for t in TABLETS:
        z = float(states[t.name].pos[2])
        if -1.0 < z < TOP - 0.02:
            write_pose(environment, t.name, (0.0, 0.0, -10.0), (0.0, 0.0, 0.0, 1.0))
            _LANDED.add(t.name)
        elif z < TOP + LANDED_ABOVE_TABLE and not _in_bottle(bottle, states[t.name].pos):
            _LANDED.add(t.name)


@on_reset
def _clear_landed(environment) -> None:
    _LANDED.clear()
