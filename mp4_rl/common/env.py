"""The base environment of the Mini Pupper v4's tasks in mjlab, composed on the simulation scene of ``mp4_sim``.

One policy step is 20 ms (50 Hz) of twenty 1 ms physics steps. The scene (``mp4_sim.envs.scene.scene_cfg``: the robot
with its fitted servo model and measured masses, its sensors, the terrain, the plant's randomisation) comes from the
task's options through ``scene_options``; on top of it this module puts the servo action (``mdp.ServoAction``: the
decoder and the 200 Hz STATE sampling the observations read), the observations of ``common/policy.py`` with their noise
and its ramp, the command delay, the commands, the terminations and the curricula.

``TaskOptions`` is the base environment on a plane (``make_env_cfg``); ``TerrainOptions`` adds the terrain and the
walking robot's plant, for evaluating the walker (``make_terrain_env_cfg``):

- **terrain:** mjlab's generator with the sub-terrains of ``mp4_sim.envs.terrain``, a spawn per difficulty row and
  family.
- **commands:** one fixed (vx, vy, wz) per env, written by the evaluation tools, observed whole (45 floats, or stacked
  frames for a policy with history).
- **robot:** shins and feet collide with the base, a stiffer foot contact, the base's mass drawn per env; the legs and
  feet (``legs``, ``foot``, ``foot_radius_m``: ``mp4_sim.robot.model.Legs``).
- **actions:** the policy's action scale and band; with ``limits`` the measured joint limits in the model and the
  decoder (``common/bounds.py``); ``stand_offset_abd_deg`` shifts the abductions' stand outward.
- **terminations:** a tilt beyond ``tilt_limit``, a body touching the ground, the time-out, leaving the terrain.

Observation noise is uniform with per-element half-widths. The ramp starts from a clean floor and ends at a profile
measured on the robot (``mp4_sim.envs.noise``: the sensor noise plus the broadband excess of the logs over the
simulation in motion).

Events: the task's own first events (the noise install, ``reset_base``, ``reset_joints``, ``command_delay``), then the
scene's plant events, then the task's other events, then the scene's late events (the attachment); the random stream
depends on this order (``mp4_sim.envs.scene.ordered_events``). A task that adds events after building calls
``late_events_last``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity import mdp as vel_mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.viewer import ViewerConfig
from mp4_sim import robots
from mp4_sim.actuators import servo_model
from mp4_sim.envs import terrain as terrain_mod
from mp4_sim.envs.noise import noise_profile
from mp4_sim.envs.scene import (  # noqa: F401  (re-exported for the tasks and tools)
    BORDER,
    SINGLE_ROW,
    Scene,
    SceneOptions,
    foot_heights_centre,
    foot_site_ids,
    ordered_events,
    scene_cfg,
)
from mp4_sim.robot.model import FOOT_RADIUS_M, foot_site_clearance, legs_of

from mp4_rl.common import mdp

_foot_site_ids = foot_site_ids

DECIMATION = 20
STEPS_PER_ITER = 24         # env steps per PPO iteration
CHAIN_MS = (8.0, 19.0)      # the sensing-to-command delay's range
NOISE_START = {"ang_vel": 0.02, "gravity": 0.01, "pos": 0.002, "vel": 0.05}     # rad/s, unit vector, rad, rad/s
NOISE_HOLD_ITERS, NOISE_RAMP_ITERS = 150, 300
LATE_EVENTS = ("attachment", "com_height")      # the scene's late events (``Scene.late_events``)


@dataclass
class TaskOptions:
    num_envs: int = 4096
    noise_start: dict = field(default_factory=lambda: dict(NOISE_START))
    noise_end: dict | None = None           # None: the robot's measured noise (mp4_sim.envs.noise, ``robot``)
    noise_ramp: bool = True                 # False: the end amplitudes from the start (evaluation under noise)
    noisy: bool = True                      # False: clean observations (evaluation, play)
    randomise: bool = True                  # friction, center of mass, delay drawn per env
    chain_ms: tuple = CHAIN_MS              # the sensing-to-command delay's range (ms), one draw per episode
    action_scale: tuple = (0.25, 0.50, 0.50) * 4
    episode_s: float = 20.0
    visuals: bool = False                   # keep the meshes (rendering only)
    actuator: str = servo_model.NAME        # the servo model: "md01" or "md300" (servo_model.ACTUATORS)
    supply_v_range: tuple | None = None     # per-env servo supply voltage, uniform (V); None: the model's 12 V
    supply_drop_ohm_range: tuple | None = None   # per-env pack + wiring resistance (Ohm), the bus sagging with the draw
    r_scale_range: tuple | None = None      # per-env winding resistance scale, uniform (heat); None: the fitted R

    def __post_init__(self):
        if self.noise_end is None:
            self.noise_end = noise_profile(robot=getattr(self, "robot", None))


@dataclass
class TerrainOptions(TaskOptions):
    terrain: str = "rough"                  # "rough", "flat", "square_walls", "stairs_built", "families"
    rows: int = 10                          # difficulty levels
    families: tuple | None = None           # "families": the columns (terrain.family_columns)
    difficulty_max: float = 1.0             # the last row's upper difficulty
    relief_pitch: float | None = None       # "families": the reliefs' heightfield pitch (m; None: the sub-terrains')
    tile: float = terrain_mod.TILE_SIZE
    terrain_seed: int = 0
    max_init_level: int = 2
    band_deg: tuple | None = None           # the decoder's band about the stand, 12 values (servo order; deg); None:
                                            # the walking band 20/60/60 (policy.BAND_DEG)
    history: int = 0                        # actor observation history (0: 45 floats)
    masses_json: str | None = "sku2_masses.json"        # the weighed robot (the robot's measured data)
    self_collision: bool = True             # shins and feet collide with the base
    friction_range: tuple = (0.5, 1.3)
    # the feet's contact (timeconst, dampratio). MuJoCo's (0.02, 1) lets a walking foot sink 3.3 mm (p99); (0.01, 1)
    # 1.7 mm, (0.005, 1) 0.6 mm
    foot_solref: tuple | None = (0.01, 1.0)
    fixed_command: tuple = (0.0, 0.0, 0.0)  # (vx, 0, wz) for every env
    level: int | None = 0                   # every env on this difficulty row
    stair_layout: tuple | None = None       # "stairs_built": ((riser, tread), ...) in m, a measured staircase
    stair_width: float | None = None        # "stairs_built": the treads' width (m; None: the whole tile)
    stair_landing: float | None = None      # "stairs_built": the top tread's length (m; None: to the tile's edge)
    tilt_limit: float = 0.7                 # rad: the body's tilt from vertical that ends an episode as a fall
    limits: bool = False                    # the measured joint limits (mp4_sim.limits), in the model and the decoder
    widen_limits_deg: float | None = None   # hip and knee targets may go this far past their track ends (bounds.py)
    stand_offset_abd_deg: float = 0.0       # the abductions' stand shifted outward; the clamps stay in the calibrated
                                            # stand's coordinates
    legs: str = "sku_2"                     # the leg set (mp4_sim.robot.model.Legs; 59.75 mm shin)
    foot: str = "sphere"                    # the feet: "sphere" (foot_radius_m) or "mesh" (the model's)
    foot_radius_m: float = FOOT_RADIUS_M    # the foot sphere's radius
    robot: str = robots.DEFAULT             # "sku2" or "sku1" (mp4_sim.robots)
    com_height_mm: tuple | None = None      # (LO, HI): the whole robot's CoM height above the floor at the stand, drawn
                                            # per robot (mp4_sim.envs.com_height); None: off
    leg_leg_contacts: bool = False          # the get-up's self-collision on this plant (mp4_sim.envs.contact: leg <->
                                            # leg, leg <-> hip can, thigh <-> base unless limits); the get-up has its own
    display_collision: bool = False         # the SKU1's display as a collision box on the base

    def __post_init__(self):
        super().__post_init__()
        # the SKU2's default masses name on another robot stands for that robot's own weighed masses
        if self.robot != robots.DEFAULT and self.masses_json == "sku2_masses.json":
            self.masses_json = robots.get(self.robot).masses_json


BASE_MASS_RANGE = (0.9, 1.1)                # the terrain robot's base mass scale, drawn per env
# The options both a task and the scene name the same way: taken over as they are.
_SHARED = ("num_envs", "visuals", "actuator", "supply_v_range", "supply_drop_ohm_range", "r_scale_range", "randomise",
           "robot", "legs", "foot", "foot_radius_m")
_TERRAIN = ("masses_json", "self_collision", "foot_solref", "limits", "terrain", "rows", "families", "difficulty_max",
            "relief_pitch", "tile", "terrain_seed", "stair_layout", "stair_width", "stair_landing", "friction_range")
# options passed to the scene only when set (their defaults are the scene's): the CoM-height draw, the display box
_TERRAIN_IF_SET = ("com_height_mm", "display_collision")


def scene_options(opt, **over) -> SceneOptions:
    """The scene of a task's options (the options adapter): ``TaskOptions`` on a plane; ``TerrainOptions`` (and the
    tasks built on it) on its terrain with the walking robot's plant (masses, shin and foot contacts, the feet's
    friction range, the base's mass drawn); ``over``: the task's own scene fields (sub-terrains, sensors, contact edits,
    the attachment)."""
    names = {f.name for f in fields(SceneOptions)}
    d = {k: getattr(opt, k) for k in _SHARED if hasattr(opt, k)}
    if isinstance(opt, TerrainOptions):
        d.update({k: getattr(opt, k) for k in _TERRAIN})
        d["max_init_level"] = opt.max_init_level if opt.level is None else 0
        d["base_mass_range"] = BASE_MASS_RANGE
        d.update({k: tuple(v) if isinstance(v, list) else v for k in _TERRAIN_IF_SET if (v := getattr(opt, k, None))})
        if getattr(opt, "leg_leg_contacts", False):
            if "collisions" in over:
                raise ValueError("leg_leg_contacts is the walker's; the get-up task sets its own contacts")
            d["collisions"] = True              # thigh <-> base too unless limits (mp4_sim.envs.scene)
    d.update(over)
    bad = set(d) - names
    if bad:
        raise ValueError(f"not scene options: {sorted(bad)}")
    return SceneOptions(**d)


def robot_kw(opt) -> dict:
    """``{"robot": name}`` for a robot other than SKU2 (the servo action's calibration), else nothing."""
    r = getattr(opt, "robot", None)
    return {} if r in (None, robots.DEFAULT) else {"robot": robots.get(r).name}


def install_noise(env, env_ids, start: dict, end: dict, ramp: bool):
    if not hasattr(env, "obs_noise"):
        env.obs_noise = mdp.NoiseTable(env, start, end)
        if not ramp:
            env.obs_noise.amp = dict(env.obs_noise.end)


def _obs_terms(noisy: bool) -> dict:
    return {
        "ang_vel": ObservationTermCfg(func=mdp.obs_ang_vel, params={"noisy": noisy}),
        "gravity": ObservationTermCfg(func=mdp.obs_gravity, params={"noisy": noisy}),
        "command": ObservationTermCfg(func=mdp.obs_command),
        "pos": ObservationTermCfg(func=mdp.obs_pos, params={"noisy": noisy}),
        "vel": ObservationTermCfg(func=mdp.obs_vel, params={"noisy": noisy}),
        "prev_action": ObservationTermCfg(func=mdp.obs_prev_action),
    }


def late_events_last(cfg) -> None:
    """Move the scene's late events (``LATE_EVENTS``) to the end of ``cfg.events``, after events a task added."""
    for k in LATE_EVENTS:
        if k in cfg.events:
            cfg.events[k] = cfg.events.pop(k)


def env_and_scene(opt: TaskOptions | None = None, **over) -> tuple[ManagerBasedRlEnvCfg, Scene]:
    """The base environment of ``opt`` on its scene (``scene_options(opt, **over)``), and the scene."""
    opt = opt or TaskOptions()
    sc = scene_cfg(scene_options(opt, **over))
    model = sc.servo
    critic = _obs_terms(False)
    critic.update({
        "lin_vel": ObservationTermCfg(func=mdp.obs_true_lin_vel),
        "height": ObservationTermCfg(func=mdp.obs_base_height),
        "contact": ObservationTermCfg(func=mdp.obs_foot_contact),
    })
    observations = {
        "actor": ObservationGroupCfg(terms=_obs_terms(opt.noisy), concatenate_terms=True, enable_corruption=False),
        "critic": ObservationGroupCfg(terms=critic, concatenate_terms=True, enable_corruption=False),
    }
    first = {
        "noise": EventTermCfg(func=install_noise, mode="startup",
                              params={"start": opt.noise_start, "end": opt.noise_end, "ramp": opt.noise_ramp}),
        "reset_base": EventTermCfg(func=vel_mdp.reset_root_state_uniform, mode="reset",
                                   params={"pose_range": {}, "velocity_range": {}}),
        "reset_joints": EventTermCfg(func=vel_mdp.reset_joints_by_offset, mode="reset",
                                     params={"position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0),
                                             "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))}),
        "command_delay": EventTermCfg(func=mdp.reset_command_delay, mode="reset",
                                      params={"latency_steps": model.latency_steps,
                                              "chain_ms": opt.chain_ms if opt.randomise else
                                              (sum(opt.chain_ms) / 2,) * 2}),
    }
    terminations = {
        "time_out": TerminationTermCfg(func=vel_mdp.time_out, time_out=True),
        "tilt": TerminationTermCfg(func=mdp.tilt_beyond, params={"limit_rad": 0.7}),
        "body_contact": TerminationTermCfg(func=mdp.body_contact),
    }
    curriculum = {}
    if opt.noise_ramp and opt.noisy:
        curriculum["noise_ramp"] = CurriculumTermCfg(
            func=mdp.ramp_noise, params={"hold_iters": NOISE_HOLD_ITERS, "ramp_iters": NOISE_RAMP_ITERS})
    commands = {"twist": UniformVelocityCommandCfg(
        entity_name="robot", resampling_time_range=(1.0e6, 1.0e6), heading_command=False,
        rel_standing_envs=0.0, rel_heading_envs=0.0, debug_vis=False,
        ranges=UniformVelocityCommandCfg.Ranges(lin_vel_x=(0.0, 0.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0)))}
    cfg = ManagerBasedRlEnvCfg(
        scene=sc.scene,
        observations=observations,
        actions={"servo": mdp.ServoActionCfg(scale=tuple(opt.action_scale), **robot_kw(opt))},
        commands=commands,
        events=ordered_events(sc, first),
        rewards={},
        terminations=terminations,
        curriculum=curriculum,
        viewer=ViewerConfig(origin_type=ViewerConfig.OriginType.ASSET_BODY, entity_name="robot", body_name="base",
                            distance=0.6, elevation=-10.0, azimuth=90.0),
        sim=SimulationCfg(njmax=300, nconmax=None,
                          mujoco=MujocoCfg(timestep=servo_model.PHYSICS_DT, iterations=10, ls_iterations=20)),
        decimation=DECIMATION,
        episode_length_s=opt.episode_s,
    )
    return cfg, sc


def make_env_cfg(opt: TaskOptions | None = None, **over) -> ManagerBasedRlEnvCfg:
    """The base environment (module docstring) of ``opt``; ``over``: scene fields (``scene_options``)."""
    return env_and_scene(opt, **over)[0]


# ------------------------------------------------------------------------------------------------ on terrain
def foot_r(opt) -> float:
    """A foot site's height above flat ground at a foot contact (``mp4_sim.robot.model.foot_site_clearance``)."""
    return foot_site_clearance(legs_of(opt))


def out_of_bounds(env, half_x: float, half_y: float, margin: float = 0.3):
    p = env.scene["robot"].data.root_link_pos_w
    return (p[:, 0].abs() > half_x - margin) | (p[:, 1].abs() > half_y - margin)


def sub_terrains(opt: TerrainOptions) -> dict:
    """The evaluation terrain's columns (``mp4_sim.envs.scene.sub_terrains``)."""
    from mp4_sim.envs.scene import sub_terrains as scene_sub_terrains
    return scene_sub_terrains(scene_options(opt))


def servo_limits(servo_cfg, opt, mode: str) -> None:
    """The decoder's clamps (``opt.limits``: ``bounds.decoder_bounds(mode)``) and stand offset on a ServoActionCfg."""
    if opt.limits:
        from mp4_rl.common.bounds import decoder_bounds
        servo_cfg.bounds = decoder_bounds(mode, widen_limits_deg=opt.widen_limits_deg,
                                          robot=getattr(opt, "robot", None)).to_meta()
    servo_cfg.stand_offset_abd_deg = float(opt.stand_offset_abd_deg)


def terrain_env_and_scene(opt: TerrainOptions | None = None, **over) -> tuple[ManagerBasedRlEnvCfg, Scene]:
    """The walking evaluation's environment (module docstring) of ``opt``, and its scene."""
    opt = opt or TerrainOptions()
    cfg, sc = env_and_scene(opt, **over)
    if opt.history:
        cfg.observations["actor"].history_length = opt.history

    fixed = opt.fixed_command
    cfg.commands = {"twist": UniformVelocityCommandCfg(
        entity_name="robot", resampling_time_range=(1.0e6, 1.0e6), heading_command=False,
        heading_control_stiffness=2.0, rel_standing_envs=0.0, rel_heading_envs=0.0, debug_vis=False,
        ranges=UniformVelocityCommandCfg.Ranges(lin_vel_x=(fixed[0], fixed[0]), lin_vel_y=(0.0, 0.0),
                                                ang_vel_z=(fixed[2], fixed[2])))}

    cfg.terminations["out_of_bounds"] = TerminationTermCfg(func=out_of_bounds, time_out=True,
                                                           params={"half_x": sc.half_x, "half_y": sc.half_y})
    cfg.terminations["tilt"].params["limit_rad"] = opt.tilt_limit
    if opt.band_deg is not None:
        band = tuple(float(b) for b in opt.band_deg)
        assert len(band) == 12 and all(b > 0 for b in band), band
        cfg.actions["servo"].band_deg = band
    servo_limits(cfg.actions["servo"], opt, "walk")
    cfg.sim.njmax = 600
    cfg.sim.nconmax = 64
    cfg.episode_length_s = opt.episode_s
    return cfg, sc


def make_terrain_env_cfg(opt: TerrainOptions | None = None, **over) -> ManagerBasedRlEnvCfg:
    """The walking evaluation's environment (module docstring); ``over``: scene fields (e.g. ``attachment``)."""
    return terrain_env_and_scene(opt, **over)[0]


def describe(opt: TaskOptions) -> dict:
    """What a run's saved config must name: the options, the servo model and the sources' hashes."""
    from mp4_sim import paths as sim_paths
    from mp4_sim.robot.model import legs_meta, sources

    from mp4_rl.common import bounds as limits
    model = servo_model.for_options(opt)
    legs = legs_of(opt)
    out = {"options": {k: (list(v) if isinstance(v, tuple) else asdict(v) if is_dataclass(v) else v)
                       for k, v in opt.__dict__.items()},
           "servo_model": model.name, "servo_params": model.params, "latency_steps": list(model.latency_steps),
           "decimation": DECIMATION, "physics_dt": servo_model.PHYSICS_DT,
           "sources": sources(legs, getattr(opt, "masses_json", None))}
    out["legs"] = legs_meta(legs)
    if getattr(opt, "com_height_mm", None) is not None:
        from mp4_sim.envs.com_height import describe as com_describe
        out["com_height"] = com_describe(scene_options(opt))
    if getattr(opt, "limits", False):
        r = robots.get(legs.robot)
        out["limits"] = limits.describe(True, getattr(opt, "kind", "walk"), getattr(opt, "stand_offset_abd_deg", 0.0),
                                        robot=r)
        out["sources"][sim_paths.name(r.limits_json)] = limits.LIMITS_SHA256 if r is robots.SKU2 else \
            sim_paths.sha256(r.limits_json)
    return out
