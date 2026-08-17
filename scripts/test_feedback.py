"""Send test feedback via Redis pub/sub."""

import json
import os
import redis
from dotenv import load_dotenv

load_dotenv()

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_CHANNEL = "feedback_events"

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# Send 3 test feedback events (unique each time to avoid duplicate detection)
import time
ts = int(time.time() * 1000)
for i in range(3):
    feedback = {
        "prompt": f"How do I optimize Python code {i}? Timestamp: {ts}",
        "chosen": f"Use profiling tools and NumPy for numerical operations. Run {i}.",
        "rejected": f"Just run it faster. Run {i}.",
        "user_id": f"user_{i}_{ts}",
        "timestamp": "2026-08-17",
        "feedback": "helpful",
    }
    r.publish(REDIS_CHANNEL, json.dumps(feedback))
    print(f"✓ Sent feedback #{i + 1}")

print(f"\n✓ Sent 3 test feedback events to Redis")
print(f"  Channel: {REDIS_CHANNEL}")
print(f"  Host: {REDIS_HOST}:{REDIS_PORT}")
