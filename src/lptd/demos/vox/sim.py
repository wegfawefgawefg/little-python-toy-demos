from __future__ import annotations

import math
import random

from . import config, state


def update_entities(dt: float) -> None:
    for ents in state.chunk_entities.values():
        for entity in ents:
            if entity["type"] != "bunny":
                continue

            entity["hop_timer"] = (entity["hop_timer"] + dt) % entity["hop_cycle"]

            vx, _, vz = entity["vel"]
            vx += random.uniform(-0.1, 0.1)
            vz += random.uniform(-0.1, 0.1)
            speed_val = math.sqrt(vx**2 + vz**2)
            max_speed = 1.0
            if speed_val > max_speed:
                factor_speed = max_speed / speed_val
                vx *= factor_speed
                vz *= factor_speed
            entity["vel"] = (vx, 0.0, vz)
            entity["pos"][0] += vx * dt * 2
            entity["pos"][2] += vz * dt * 2

