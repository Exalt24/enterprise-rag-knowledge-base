"""The Hatchet client, built from the environment.

HATCHET_CLIENT_TOKEN is required. For the self-hosted hatchet-lite container that this repo's docker-compose.hatchet.yml
starts, also set HATCHET_CLIENT_TLS_STRATEGY=none (the container serves gRPC without TLS). The SDK reads both, and
HATCHET_CLIENT_HOST_PORT when the token's own address is not the one to use. See docs/ingestion.md.
"""
import os
from functools import lru_cache

from hatchet_sdk import Hatchet


def is_configured() -> bool:
    return bool(os.environ.get("HATCHET_CLIENT_TOKEN"))


@lru_cache(maxsize=1)
def get_hatchet() -> Hatchet:
    return Hatchet()
