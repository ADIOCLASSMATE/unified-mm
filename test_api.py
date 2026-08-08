import os

from anthropic import Anthropic

client = Anthropic(
    base_url="https://apicz.boyuerichdata.com/", api_key=os.getenv("SII_API_KEY")
)

message1 = client.messages.create(
    model="deepseek-v4-pro",
    max_tokens=3200,
    thinking={"type": "enabled", "budget_tokens": 1600},
    messages=[{"role": "user", "content": "自我介绍一下"}],
)

print(message1)

message2 = client.messages.create(
    model="glm-5.2",
    max_tokens=3200,
    thinking={"type": "enabled", "budget_tokens": 1600},
    messages=[{"role": "user", "content": "自我介绍一下"}],
)

print(message2)

message3 = client.messages.create(
    model="kimi-k3",
    max_tokens=3200,
    thinking={"type": "enabled", "budget_tokens": 1600},
    messages=[{"role": "user", "content": "自我介绍一下"}],
)

print(message3)
