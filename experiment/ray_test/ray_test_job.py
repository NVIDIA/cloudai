import sys
import ray

ray.init(address="auto")

print(f"Available resources: {ray.available_resources()}", flush=True)
