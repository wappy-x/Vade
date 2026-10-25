#!/usr/bin/env python3
"""Purpose: use Vade's OpenAI-compatible API from std-only Python: streaming, tool calling, prompt caching across turns.

Run:    python3 -I examples/openai_client.py           (all three demos)
        python3 -I examples/openai_client.py stream    (or tools, or cache)
Needs:  a running engine (vade launch on the Spark, or vade up from a controller)
        and Python 3.9 or newer. Nothing to install.
        VADE_API_URL picks another endpoint (default http://127.0.0.1:18080).
        VADE_API_KEY_FILE sends the key in that file as a bearer token.
The cache demo shows reused prompt tokens only when the engine was started with
VADE_FLASHNEXT_PREFIX_CACHE=1 (one Spark; the prefix cache is off by default).
Expected output: see "Python: streaming, tools and caching" in examples/README.md.
"""
import json
import os
import sys
import urllib.request

BASE = os.environ.get("VADE_API_URL", "http://127.0.0.1:18080").rstrip("/").removesuffix("/v1")
HEADERS = {"Content-Type": "application/json"}
if os.environ.get("VADE_API_KEY_FILE"):
    with open(os.environ["VADE_API_KEY_FILE"]) as key:
        HEADERS["Authorization"] = "Bearer " + key.read().strip()


def call(path, body=None):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(BASE + path, data, HEADERS)
    return urllib.request.urlopen(request, timeout=600)


def chat(body):
    with call("/v1/chat/completions", body) as response:
        return json.load(response)


def stream(model):
    """Print the answer as it arrives, then the usage the last chunk carries."""
    body = {"model": model, "stream": True, "max_tokens": 64, "enable_thinking": False,
            "messages": [{"role": "user", "content": "Write a haiku about a small computer that runs a large model."}]}
    usage = None
    with call("/v1/chat/completions", body) as response:
        for line in response:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[len("data: "):])
            for choice in chunk["choices"]:
                print(choice["delta"].get("content") or "", end="", flush=True)
            usage = chunk.get("usage") or usage
    print(f"\n[usage] prompt {usage['prompt_tokens']}, completion {usage['completion_tokens']}")


WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "Current weather for a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}


def tools(model):
    """The model asks for a tool, this script runs it, and the model answers with the result."""
    messages = [{"role": "user", "content": "What is the weather in Paris right now?"}]
    body = {"model": model, "tools": [WEATHER], "max_tokens": 256, "enable_thinking": False}
    reply = chat({**body, "messages": messages})["choices"][0]
    tool_call = reply["message"]["tool_calls"][0]
    arguments = json.loads(tool_call["function"]["arguments"])
    print(f"[finish_reason] {reply['finish_reason']}")
    print(f"[tool call] {tool_call['function']['name']}({json.dumps(arguments)})")
    result = {"city": arguments["city"], "temperature_c": 18, "sky": "clear"}  # your code runs the tool here
    messages += [reply["message"], {"role": "tool", "tool_call_id": tool_call["id"], "content": json.dumps(result)}]
    answer = chat({**body, "messages": messages})["choices"][0]["message"]["content"]
    print(f"[answer] {answer.strip()}")


def cache(model):
    """Two turns of one conversation; the second reads its shared prefix from the prefix cache."""
    notes = " ".join(f"Rule {i}: keep function {i} under {10 + i} lines." for i in range(1, 121))
    messages = [{"role": "system", "content": "You review code against these house rules. " + notes},
                {"role": "user", "content": "How many lines may function 7 have? Answer with the number."}]
    # prompt_cache_key asks the engine to keep this prompt for the next turn.
    body = {"model": model, "max_tokens": 16, "enable_thinking": False, "prompt_cache_key": "examples-review"}
    for turn in (1, 2):
        response = chat({**body, "messages": messages})
        usage = response["usage"]
        cached = usage["prompt_tokens_details"]["cached_tokens"]
        answer = response["choices"][0]["message"]["content"].strip()
        print(f"[turn {turn}] prompt {usage['prompt_tokens']} tokens, cached {cached}: {answer}")
        messages += [{"role": "assistant", "content": answer},
                     {"role": "user", "content": "And function 9? Answer with the number."}]


if __name__ == "__main__":
    with call("/v1/models") as response:
        model = json.load(response)["data"][0]["id"]
    print(f"[model] {model} at {BASE}")
    demos = {"stream": stream, "tools": tools, "cache": cache}
    for name in sys.argv[1:] or demos:
        print(f"== {name}")
        demos[name](model)
