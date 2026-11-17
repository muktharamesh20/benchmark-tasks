"""Scripted demo for the shake-out task, run in the yam_arena embodiment.

    python demo.py shake_out_tablets.py demo_out [--max-steps 1200] [--no-record]

The right arm grips the bottle from the side, lifts it, tips the mouth partly toward the table, and shakes it with
short back-and-forth jerks (not a continuous pour) until exactly N_OUT tablets have left. It then snaps upright and
sets the bottle down. DEMO_DEBUG=1 prints tilt and count every half second.

DEMO_VARIANT=settle only waits 10 s, for checking that the tablets settle in the bottle and none escape.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

CONTROL_HZ = 20
GRIP_Z = 0.06                      # grip height above the bottle's base (m)
GRIP_PITCH = math.radians(20)      # the approach axis pitched down from level
GRIP_HEADING = math.radians(40)    # the approach's heading about the vertical (0: straight out along +x)
BACK_OFF = 0.08                    # how far back along the approach the fingers open before closing in
# Close the pads to 44 mm, 3 mm inside each side of the 50.3 mm bottle (about 15 N a finger, at the finger drive's
# 5000 N/m) rather than fully shut: the full 50 N squeeze sets the light bottle chattering, which throws the tablets
# through its walls.
GRIP_SHUT = 0.022 / 0.04695
SHAKE_TCP = np.array([0.30, -0.15, 0.16])  # grip point while shaking (z above the table)
TILT_AXIS = np.array([-1.0, 0.0, 0.0])     # tipping about -x turns the mouth toward +y
# Shake, don't pour: hold a tip and jerk ±SHAKE_AMP_DEG, deepening the base tip slowly only if nothing comes out.
# The shoulder narrows the 44 mm inside to the 31 mm mouth, so tablets only clear it once the mouth points a little
# below level; nothing came out at up to 93 deg in the first run with the real-size bottle.
BASE_DEG = 95.0
SHAKE_AMP_DEG = 12.0
SHAKE_S = 0.18          # half-period of one shake jerk
MAX_SHAKES = 40
DEEPEN_EVERY = 3        # shakes with no new tablet → tip base 4° deeper
DEEPEN_DEG, MAX_BASE_DEG = 4.0, 115.0
# Turning back upright: fast enough that no more tablets slide out, slow enough not to fling the ones in the neck.
# (0.35 s to 40 deg threw 2 extra tablets out in an earlier run.)
TURN_BACK_S = 0.7
LAUNCH_M_S = 1.0        # log any tablet moving faster than this
CHECK_DEG = 50.0        # tip back to this (mouth up) after each round to settle the count
MAX_ROUNDS = 4


def _rot(axis, angle) -> np.ndarray:
    from yam_arena.urdf import axis_angle_matrix

    return axis_angle_matrix(np.asarray(axis, float), angle)


def grip_rotation() -> np.ndarray:
    """The TCP rotation of the side grip: fingers closing horizontally, approach pitched down GRIP_PITCH."""
    from yam_arena.demo import grip_rotation as down_grip

    return _rot([0, 0, 1], GRIP_HEADING) @ _rot([0, 1, 0], -(math.pi / 2 - GRIP_PITCH)) @ down_grip(0.0)


class Arm:
    """One arm: smooth Cartesian moves (IK on the URDF), each starting where the last command left it."""

    def __init__(self, name: str, base_pos):
        from yam_arena import ik
        from yam_arena.constants import READY

        self.name, self.base = name, np.asarray(base_pos, float)
        self.q = np.array(READY, float)
        self.grip = 1.0
        self.pos, self.rot = ik.tcp_pose(self.q, self.base)
        self.cmd = np.r_[self.q, self.grip]
        self.worst_mm = 0.0

    def _solve(self, pos, rot):
        from yam_arena import ik

        q, pe, _ = ik.solve(pos, ik.matrix_to_quat_xyzw(rot), self.q, base_pos=self.base, iters=60)
        self.q, self.pos, self.rot = q, np.asarray(pos, float), rot
        self.worst_mm = max(self.worst_mm, pe * 1000)
        self.cmd = np.r_[q, self.grip]

    def move(self, pos, rot, duration: float, grip: float | None = None):
        """Generator: go to ``pos``/``rot`` (world TCP); ``grip`` 1 open .. 0 closed (None keeps it)."""
        from yam_arena import ik

        p0, r0, g0 = self.pos.copy(), self.rot.copy(), self.grip
        g1 = g0 if grip is None else grip
        n = max(1, round(duration * CONTROL_HZ))
        for k in range(1, n + 1):
            s = ik.smoothstep(k / n)
            self.grip = g0 + s * (g1 - g0)
            self._solve(p0 + s * (np.asarray(pos) - p0), ik.slerp_matrix(r0, rot, s))
            yield

    def set_grip(self, grip: float, duration: float = 0.8):
        yield from self.move(self.pos, self.rot, duration, grip)

    def aim(self, pos, rot):
        """Track a pose this step (for closed-loop moves)."""
        self._solve(pos, rot)


def wait(seconds: float):
    for _ in range(max(1, round(seconds * CONTROL_HZ))):
        yield


def _fallen_through(task, states) -> int:
    """Tablets parked under the world (sim failure) — must stay 0 for a clean demo."""
    return sum(1 for t in task.TABLETS if float(states[t.name].pos[2]) < task.TOP - 0.02)


def script(task, ctx):
    """The demo as a generator: one yield per control step."""
    T, A = task, ctx.right
    ctx.note["tilt_deg"] = 0.0
    ctx.note["target"] = task.N_OUT
    ctx.phase = "settle"

    # 0. Let the tablets settle.
    yield from wait(1.5 if os.environ.get("DEMO_VARIANT") != "settle" else 10.0)
    if os.environ.get("DEMO_VARIANT") == "settle":
        return

    # 1. Grip the bottle from the side and lift it.
    ctx.phase = "grip"
    G = grip_rotation()
    approach = G[:, 2]
    grasp = np.asarray(ctx.states()["bottle"].pos, float) + [0.0, 0.0, GRIP_Z]
    yield from A.move(grasp - BACK_OFF * approach + [0, 0, 0.03], G, 2.5, grip=1.0)
    yield from A.move(grasp - BACK_OFF * approach, G, 1.0)
    yield from A.move(grasp, G, 1.5)
    yield from A.set_grip(GRIP_SHUT, 1.0)
    yield from wait(0.3)
    yield from A.move(grasp + [0, 0, 0.08], G, 1.5)
    hold = SHAKE_TCP + [0.0, 0.0, T.TOP]
    yield from A.move(hold, G, 1.5)

    def out_count():
        return T.tablets_out(ctx.states())

    def committed():
        """Tablets out, plus those already in the neck (past the shoulder), which will follow if we keep tipping."""
        st = ctx.states()
        bottle = st["bottle"]
        in_neck = 0
        for t in T.TABLETS:
            x, y, z = bottle.to_local(st[t.name].pos)
            if T.NECK_BOTTOM < z < T.HEIGHT and math.hypot(x, y) < T.inner_radius(z):
                in_neck += 1
        return T.tablets_out(st) + in_neck

    def enough():
        return committed() >= ctx.note["target"]

    def tilt_to(deg, duration, stop=enough):
        """Turn the bottle to ``deg`` about TILT_AXIS; returns True if ``stop()`` cut it short."""
        from yam_arena import ik

        start, n = ctx.note["tilt_deg"], max(1, round(duration * CONTROL_HZ))
        for k in range(1, n + 1):
            ctx.note["tilt_deg"] = start + ik.smoothstep(k / n) * (deg - start)
            # Small lateral jerk while shaking so it reads as a shake, not a static pour.
            jolt = np.array([0.0, 0.008 * math.sin(math.radians(ctx.note["tilt_deg"] * 3)), 0.0])
            A.aim(hold + jolt, _rot(TILT_AXIS, math.radians(ctx.note["tilt_deg"])) @ G)
            yield
            if stop():
                return True
        return False

    # 2. Tip, then shake with short jerks until N_OUT tablets are out or in the neck; tip back mouth-up so any in the
    # neck slide back in, check the count, and go again for the rest if short.
    never = lambda: False  # noqa: E731
    base = BASE_DEG
    shakes, rounds = 0, 0
    while out_count() < T.N_OUT and shakes < MAX_SHAKES and rounds < MAX_ROUNDS:
        rounds += 1
        # One more tablet usually trails out as the bottle turns back, so the first round stops one short.
        ctx.note["target"] = T.N_OUT - 1 if out_count() < T.N_OUT - 1 else T.N_OUT
        ctx.phase = "tip"
        yield from tilt_to(base, 2.5 if rounds == 1 else 1.0, never)
        ctx.phase = "shake"
        reached, stall = enough(), 0
        while not reached and shakes < MAX_SHAKES:
            shakes += 1
            before = committed()
            # One shake cycle: dip, return past base, settle — cut short the moment enough are committed.
            reached = (yield from tilt_to(base + SHAKE_AMP_DEG, SHAKE_S))
            if not reached:
                reached = yield from tilt_to(base - SHAKE_AMP_DEG * 0.5, SHAKE_S)
            if not reached:
                reached = yield from tilt_to(base, SHAKE_S * 0.6)
            if committed() == before:
                stall += 1
                if stall >= DEEPEN_EVERY:
                    base = min(MAX_BASE_DEG, base + DEEPEN_DEG)
                    stall = 0
                    yield from tilt_to(base, 0.5, never)
            else:
                stall = 0
        # Mouth up, so tablets in the neck slide back; the count is then settled.
        ctx.phase = "check"
        yield from tilt_to(CHECK_DEG, TURN_BACK_S, never)
        yield from wait(0.8)
        base = max(BASE_DEG, base - DEEPEN_DEG)  # resume a little shallower: the heap is now nearer the mouth
    ctx.note["shakes"] = shakes
    ctx.note["rounds"] = rounds
    ctx.note["base_deg"] = base
    ctx.note["out_at_turn_back"] = out_count()
    ctx.note["fallen_through"] = _fallen_through(T, ctx.states())

    # 3. Snap upright immediately so no extra tablets slide out, then set down.
    ctx.phase = "turn_back"
    yield from tilt_to(15.0, 1.0, never)
    yield from tilt_to(0.0, 1.0, never)
    yield from wait(0.5)
    ctx.phase = "set_down"
    yield from A.move(grasp + [0, 0, 0.03], G, 2.0)
    yield from A.move(grasp + [0, 0, 0.002], G, 1.0)
    yield from A.set_grip(1.0, 0.8)
    yield from A.move(grasp - BACK_OFF * approach, G, 1.0)
    yield from A.move(grasp - BACK_OFF * approach + [0, 0, 0.06], G, 1.0)


class Ctx:
    def __init__(self, emb):
        from yam_arena import station

        self.emb = emb
        self.left = Arm("left", station.BASES["left"])
        self.right = Arm("right", station.BASES["right"])
        self.note: dict = {}
        self.phase = "start"

    def states(self):
        from yam_arena.scene import object_states

        return object_states(self.emb._env, self.emb.task.names)

    def action(self):
        from inspect_robots.types import Action
        from inspect_robots_yam.packing import pack

        return Action(data=pack(self.left.cmd, self.right.cmd))


def _write_mp4(frames, path: Path) -> None:
    import imageio

    imageio.mimsave(path, [np.asarray(f) for f in frames], fps=CONTROL_HZ, codec="libx264",
                    output_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])


def _bottle_tilt_deg(bottle) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, float(bottle.rotation()[2, 2])))))


def run(task_path: str, out_dir: str, max_steps: int = 1200, record: bool = True, cam_width: int = 320,
        cam_height: int = 176, hold_s: float = 2.0) -> dict:
    from inspect_robots import Scene
    from yam_arena.inspect_embodiment import yam_arena_embodiment

    task_path = str(Path(task_path).resolve())
    sys.path.insert(0, str(Path(task_path).parent))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    emb = yam_arena_embodiment(task=task_path, object_jitter=0, control_hz=CONTROL_HZ, cam_width=cam_width,
                               cam_height=cam_height)
    obs = emb.reset(Scene(id="demo", instruction="demo"))
    task = emb.task.module
    ctx = Ctx(emb)
    gen = script(task, ctx)
    frames: dict[str, list] = {}

    def snap(o):
        if record:
            for cam, img in o.images.items():
                frames.setdefault(cam, []).append(np.asarray(img))

    snap(obs)
    steps, ended, success, info, fastest = 0, None, False, {}, 0.0
    launches: list[dict] = []
    while steps < max_steps:
        try:
            next(gen)
        except StopIteration:
            ended = steps if ended is None else ended
        res = emb.step(ctx.action())
        steps += 1
        info = res.info
        snap(res.observation)
        st = ctx.states()
        for t in task.TABLETS:
            speed = float(np.linalg.norm(st[t.name].lin_vel))
            fastest = max(fastest, speed)
            if speed > LAUNCH_M_S and len(launches) < 200:
                local = [round(float(c) * 1000, 1) for c in st["bottle"].to_local(st[t.name].pos)]
                launches.append({"step": steps, "phase": ctx.phase, "tablet": t.name, "m_s": round(speed, 2),
                                 "in_bottle_mm": local, "bottle_tilt_deg": round(_bottle_tilt_deg(st["bottle"]), 1)})
                if os.environ.get("DEMO_DEBUG"):
                    print("LAUNCH", launches[-1], flush=True)
        if steps % 100 == 0 or (os.environ.get("DEMO_DEBUG") and steps % 10 == 0):
            print(f"step {steps} score {info.get('score')} tilt_cmd {ctx.note.get('tilt_deg', 0.0):.1f} "
                  f"bottle_tilt {_bottle_tilt_deg(st['bottle']):.1f} out {task.tablets_out(st)} "
                  f"fallen {_fallen_through(task, st)} fastest_tablet_m_s {fastest:.2f} "
                  f"lowest_tablet_z {min(st[t.name].pos[2] for t in task.TABLETS):.3f}",
                  flush=True)
        if res.terminated:
            success = res.termination_reason == "success"
            break
        if ended is not None and steps - ended >= hold_s * CONTROL_HZ:
            break
    st = ctx.states()
    scored = float(task.score(st)) if hasattr(task, "score") else float(info.get("score", 0.0) or 0.0)
    score = 1.0 if success or (hasattr(task, "success") and task.success(st)) else scored
    result = {"task_score": score, "success": bool(success or (hasattr(task, "success") and task.success(st))),
              "steps": steps, "variant": os.environ.get("DEMO_VARIANT", ""),
              "tablets_out": task.tablets_out(st), "fallen_through": _fallen_through(task, st),
              "bottle_tilt_deg": round(_bottle_tilt_deg(st["bottle"]), 1),
              "fastest_tablet_m_s": round(fastest, 3), "launch_count": len(launches), **ctx.note,
              "worst_ik_mm": {"left": round(ctx.left.worst_mm, 2), "right": round(ctx.right.worst_mm, 2)}}
    if record:
        import imageio

        for cam, fs in frames.items():
            _write_mp4(fs, out / f"demo_{cam}.mp4")
        top = next((c for c in frames if "top" in c), next(iter(frames), None))
        if top:
            imageio.imwrite(out / "setup.png", frames[top][0])
    (out / "result.json").write_text(json.dumps(result, indent=2))
    (out / "launches.json").write_text(json.dumps(launches, indent=1))
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task")
    ap.add_argument("out")
    ap.add_argument("--max-steps", type=int, default=1200)
    ap.add_argument("--no-record", action="store_true")
    a = ap.parse_args()
    run(a.task, a.out, a.max_steps, record=not a.no_record)
