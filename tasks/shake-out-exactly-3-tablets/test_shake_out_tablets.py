"""Checks of shake_out_tablets.py without the simulator. Needs yam_arena (the official stack, or just its source on
PYTHONPATH):

    python -m pytest -q test_shake_out_tablets.py
"""

import math
from pathlib import Path

import numpy as np
import pytest

import demo as D
from partkit import PartState, _box, _quat_matrix
from yam_arena import station
from yam_arena.task_file import ObjectState, load

TASK = load(Path(__file__).with_name("shake_out_tablets.py"))
M = TASK.module
MM = 0.001


def _solid(p) -> bool:
    """Whether a point (bottle frame) is inside any of the bottle's shapes."""
    p = np.asarray(p, float)
    for centre, size, _, rot in map(_box, M.BOTTLE.boxes):
        local = _quat_matrix(rot).T @ (p - np.array(centre))
        if np.all(np.abs(local) <= np.array(size) / 2):
            return True
    for centre, radius, height, _ in M.BOTTLE.cylinders:
        d = p - np.array(centre)
        if math.hypot(d[0], d[1]) <= radius and abs(d[2]) <= height / 2:
            return True
    return False


def _at(r, angle, z):
    return (r * math.cos(angle), r * math.sin(angle), z)


ANGLES = [2 * math.pi * k / (4 * M.SIDES) for k in range(4 * M.SIDES)]  # four per side, so seams are hit too
HEIGHTS = np.linspace(0.0, M.HEIGHT, 120)


def test_loads():
    assert TASK.names == ("bottle", *(f"tablet_{i:02d}" for i in range(1, 31)))
    assert all(TASK.module.OBJECTS[i].free for i in range(31))


def test_outside_matches_the_real_bottle():
    w, d, h = M.BOTTLE.size
    assert w == pytest.approx(0.0503, abs=0.6 * MM) and d == pytest.approx(0.0503, abs=0.6 * MM)
    assert h == pytest.approx(M.HEIGHT, abs=0.1 * MM)


def test_wall_is_closed_all_the_way_round():
    """1 mm into the wall, everywhere from the base to the lip, is solid: no gaps at seams or joints."""
    gaps = [(round(a, 3), round(z, 4)) for z in HEIGHTS for a in ANGLES if not _solid(_at(M.inner_radius(z) + MM, a, z))]
    assert not gaps, gaps[:10]


def test_base_is_closed():
    assert all(_solid((r * math.cos(a), r * math.sin(a), M.WALL / 2)) for r in np.linspace(0, M.INNER_R, 12)
               for a in ANGLES)


def test_inside_is_clear():
    """0.5 mm in from the inner wall, from just above the base up through the mouth, nothing is solid."""
    blocked = [(round(r, 4), round(a, 3), round(z, 4))
               for z in HEIGHTS if z > M.WALL + 0.5 * MM
               for r in np.linspace(0, M.inner_radius(z) - 0.5 * MM, 6)
               for a in ANGLES if _solid(_at(r, a, z))]
    assert not blocked, blocked[:10]


def test_mouth_is_open_above():
    assert not any(_solid(_at(r, a, M.HEIGHT + MM)) for r in np.linspace(0, M.OUTER_R, 8) for a in ANGLES)


def _local(tablet):
    return np.array(tablet.pos) - np.array(M.BOTTLE.pos)


def test_tablets_match_the_printed_ones():
    for t in M.TABLETS:
        (_, radius, height, _), = t.cylinders
        assert (radius, height) == (0.005, 0.004)
        assert t.mass == pytest.approx(math.pi * 0.005 ** 2 * 0.004 * 1240, rel=0.01)  # solid PLA


def test_tablets_start_inside_the_bottle_clear_of_the_walls():
    """Each tablet, upright as spawned, sits at least 1 mm inside the inner wall and above the base."""
    for t in M.TABLETS:
        x, y, z = _local(t)
        assert math.hypot(x, y) + M.TABLET_R <= M.INNER_R - MM, t.name
        assert z - M.TABLET_H / 2 >= M.WALL + MM, t.name
        assert z + M.TABLET_H / 2 < M.BODY_TOP, t.name


def test_tablets_start_apart():
    """No two tablets overlap: either their discs are apart or their layers are."""
    for i, a in enumerate(M.TABLETS):
        for b in M.TABLETS[i + 1:]:
            d = _local(a) - _local(b)
            assert math.hypot(d[0], d[1]) >= 2 * M.TABLET_R + MM or abs(d[2]) >= M.TABLET_H + MM, (a.name, b.name)


# -- counting tablets out -----------------------------------------------------------------------------------------------

def _bottle_state(tilt_deg=0.0):
    """The bottle at its start, tipped ``tilt_deg`` about world -x (mouth toward +y), about its base centre."""
    t = math.radians(tilt_deg)
    return PartState(pos=np.array(M.BOTTLE.pos), quat_xyzw=np.array([-math.sin(t / 2), 0.0, 0.0, math.cos(t / 2)]),
                     lin_vel=np.zeros(3), ang_vel=np.zeros(3), low=np.zeros(3), high=np.zeros(3))


def _tablet_state(pos):
    p = np.asarray(pos, float)
    return ObjectState(pos=p, quat_xyzw=np.array([0, 0, 0, 1.0]), lin_vel=np.zeros(3), ang_vel=np.zeros(3), low=p,
                       high=p)


def _scene(bottle=None, **moved):
    """The start: every tablet where it spawns, inside the bottle, unless given as name=world position."""
    bottle = bottle or _bottle_state()
    out = {"bottle": bottle}
    for t in M.TABLETS:
        out[t.name] = _tablet_state(moved.get(t.name, bottle.to_world(_local(t))))
    return out


def test_at_the_start_no_tablet_is_out():
    assert M.tablets_out(_scene()) == 0


def test_tablets_inside_a_tipped_bottle_are_not_out():
    assert M.tablets_out(_scene(_bottle_state(tilt_deg=110))) == 0


def test_tablets_on_the_table_or_past_the_lip_are_out():
    b = _bottle_state()
    beside = (M.BOTTLE.pos[0], M.BOTTLE.pos[1] + 0.06, M.TOP + 0.002)
    above_lip = b.to_world((0.0, 0.0, M.HEIGHT + 0.003))
    assert M.tablets_out(_scene(tablet_01=beside, tablet_02=above_lip)) == 2


def test_a_tablet_in_the_neck_is_still_in():
    b = _bottle_state(tilt_deg=100)
    assert M.tablets_out(_scene(b, tablet_01=b.to_world((0.0, 0.0, M.HEIGHT - 0.003)))) == 0


def _track(monkeypatch, scene):
    """Run the task's per-step tablet tracker on a frozen scene (no simulator)."""
    import sys
    import types

    fake = types.ModuleType("yam_arena.scene")
    fake.object_states = lambda environment, names: {n: scene[n] for n in names}
    monkeypatch.setitem(sys.modules, "yam_arena.scene", fake)
    monkeypatch.setattr(M, "write_pose", lambda *a, **k: None)
    M._track_tablets(None)


def test_a_tablet_put_back_after_landing_still_counts(monkeypatch):
    monkeypatch.setattr(M, "_LANDED", set())
    beside = (M.BOTTLE.pos[0] - 0.08, M.BOTTLE.pos[1], M.TOP + 0.002)
    _track(monkeypatch, _scene(tablet_01=beside))
    assert M.tablets_out(_scene()) == 1  # back at its start, inside the bottle, but it landed on the table


def test_a_tablet_falling_past_the_lip_is_not_latched_until_it_lands(monkeypatch):
    monkeypatch.setattr(M, "_LANDED", set())
    t = _bottle_state(tilt_deg=100)
    b = PartState(pos=t.pos + np.array([0.0, 0.0, 0.12]), quat_xyzw=t.quat_xyzw, lin_vel=np.zeros(3),
                  ang_vel=np.zeros(3), low=np.zeros(3), high=np.zeros(3))  # held up at shaking height
    in_air = b.to_world((0.0, 0.0, M.HEIGHT + 0.01))
    _track(monkeypatch, _scene(b, tablet_01=in_air))
    assert M._LANDED == set()
    assert M.tablets_out(_scene(b)) == 0


# -- scoring ------------------------------------------------------------------------------------------------------------

def _on_table(tilt_deg=0.0):
    """Bottle resting on the table at ``tilt_deg``, with its lowest point at the table top."""
    b = _bottle_state(tilt_deg)
    # PartState.low is used by _bottle_on_table; put it on the table.
    return PartState(pos=b.pos, quat_xyzw=b.quat_xyzw, lin_vel=np.zeros(3), ang_vel=np.zeros(3),
                     low=np.array([b.pos[0], b.pos[1], M.TOP]), high=b.pos + 0.05)


def test_score_exact_and_upright_is_one():
    # On the table, clear of the bottle in −x (not along the pour axis, which is +y).
    beside = (M.BOTTLE.pos[0] - 0.08, M.BOTTLE.pos[1], M.TOP + 0.002)
    scene = _scene(_on_table(), **{f"tablet_{i:02d}": beside for i in (1, 2, 3)})
    assert M.tablets_out(scene) == 3
    assert M.score(scene) == pytest.approx(1.0)
    assert M.success(scene)


def test_score_off_by_one_and_on_side():
    beside = (M.BOTTLE.pos[0] - 0.08, M.BOTTLE.pos[1], M.TOP + 0.002)
    scene = _scene(_on_table(tilt_deg=90), **{f"tablet_{i:02d}": beside for i in (1, 2, 3, 4)})
    assert M.score(scene) == pytest.approx(0.75 * 0.5 + 0.25 * 0.5)


def test_score_zero_when_nothing_came_out():
    assert M.score(_scene(_on_table())) == 0.0
    assert not M.success(_scene(_on_table()))


def test_upright_credit_ignored_until_a_tablet_is_out():
    assert M.upright_credit(_scene(_on_table())) == 0.0


# -- the demo's motion --------------------------------------------------------------------------------------------------

class _FakeCtx:
    """The demo's context with a frozen scene: no tablet ever leaves, so the demo tips the bottle all the way."""

    def __init__(self):
        self.left = D.Arm("left", station.BASES["left"])
        self.right = D.Arm("right", station.BASES["right"])
        self.note = {}

    def states(self):
        return _scene()


def test_the_demo_is_within_reach_and_smooth():
    """Played against a scene where nothing comes out, the right arm tips and shakes the bottle: the IK reaches
    every pose, and no joint moves more than 12 degrees in one control step (0.05 s). Shake jerks are sharper
    than a smooth pour, so the per-step budget is a bit higher than the old 8° pour limit."""
    ctx = _FakeCtx()
    q, biggest = ctx.right.q.copy(), 0.0
    for _ in D.script(M, ctx):
        biggest = max(biggest, float(np.abs(ctx.right.q - q).max()))
        q = ctx.right.q.copy()
    assert ctx.note["out_at_turn_back"] == 0
    assert ctx.right.worst_mm < 0.5, ctx.right.worst_mm
    assert math.degrees(biggest) < 12.0, math.degrees(biggest)
