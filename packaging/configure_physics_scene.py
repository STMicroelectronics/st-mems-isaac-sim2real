# ******************************************************************************
# File Name          : configure_physics_scene.py
# Description        : Create or update the current Isaac Sim PhysicsScene to a
#                      deterministic steps-per-second setting for Sim2Real IMU
#                      validation.
# ******************************************************************************
# @attention
#
# Copyright (c) 2026 STMicroelectronics.
# All rights reserved.
#
# This software is licensed under terms that can be found in the LICENSE file
# in the root directory of this software component.
# If no LICENSE file comes with this software, it is provided AS-IS.
#
# ******************************************************************************

import argparse

import omni.usd
from pxr import PhysxSchema, Sdf, UsdPhysics


DEFAULT_STEPS_PER_SECOND = 208.0
DEFAULT_PHYSICS_SCENE_PATH = "/World/physicsScene"


def _find_or_create_physics_scene(stage):
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Scene):
            return prim
    return UsdPhysics.Scene.Define(stage, Sdf.Path(DEFAULT_PHYSICS_SCENE_PATH)).GetPrim()


def configure_physics_scene(steps_per_second: float = DEFAULT_STEPS_PER_SECOND):
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("No stage is currently open.")

    physics_scene_prim = _find_or_create_physics_scene(stage)
    physx_api = PhysxSchema.PhysxSceneAPI.Apply(physics_scene_prim)
    physx_api.GetTimeStepsPerSecondAttr().Set(float(steps_per_second))

    print(f"PhysicsScene: {physics_scene_prim.GetPath()}")
    print(f"Steps/sec: {float(steps_per_second):.3f}")
    print(f"dt: {1.0 / float(steps_per_second):.6f}s")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Create or update the current Isaac Sim PhysicsScene for Sim2Real IMU validation."
    )
    parser.add_argument(
        "--steps-per-second",
        type=float,
        default=DEFAULT_STEPS_PER_SECOND,
        help=f"Physics Scene rate to apply (default: {DEFAULT_STEPS_PER_SECOND:.0f}).",
    )
    args = parser.parse_args(argv)

    configure_physics_scene(args.steps_per_second)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
