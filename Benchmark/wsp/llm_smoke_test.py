"""Pre-flight check for a chat-completions agent model (e.g. vLLM serving Qwen3.8-27B).

Verifies, before an expensive agent run, that the server (1) returns a structured tool call, (2) returns reasoning in a
separate field, and (3) accepts the reasoning + tool result back and continues. Exit code 0 = ready.

Usage: uv run python llm_smoke_test.py --base-url http://127.0.0.1:8000/v1 --model qwen3.8-27b
"""
import argparse, json, sys

from openai import OpenAI

ap = argparse.ArgumentParser()
ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
ap.add_argument("--model", default="qwen3.8-27b")
a = ap.parse_args()
client = OpenAI(api_key="EMPTY", base_url=a.base_url, timeout=600)
tools = [{"type": "function", "function": {"name": "python", "description": "Run python code and return stdout.",
          "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]}}}]
extra = {"top_k": 20, "chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": True}}
msgs = [{"role": "system", "content": "You are an agent. Use the python tool for any computation."},
        {"role": "user", "content": "Use the python tool to compute 123456 * 654321, then tell me the result."}]

r = client.chat.completions.create(model=a.model, messages=msgs, tools=tools, temperature=1.0, top_p=0.95, max_tokens=4000, extra_body=extra)
m = r.choices[0].message
md = m.model_dump()
thinking = md.get("reasoning_content") or md.get("reasoning") or ""
print(f"turn 1: finish={r.choices[0].finish_reason} tool_calls={[(t.function.name, t.function.arguments) for t in (m.tool_calls or [])]} "
      f"reasoning_chars={len(thinking)}")
if not m.tool_calls:
    sys.exit("FAIL: no structured tool call (check --enable-auto-tool-choice --tool-call-parser qwen3_coder)")
try:
    json.loads(m.tool_calls[0].function.arguments)
except ValueError:
    sys.exit("FAIL: tool-call arguments are not valid JSON")
if not thinking:
    print("WARN: no separate reasoning field (check --reasoning-parser qwen3); the agent still works but loses preserved thinking")

asst = {"role": "assistant", "content": m.content or "", "tool_calls": [
    {"id": t.id, "type": "function", "function": {"name": t.function.name, "arguments": t.function.arguments}} for t in m.tool_calls]}
if thinking:
    asst["reasoning_content"] = thinking
msgs += [asst, {"role": "tool", "tool_call_id": m.tool_calls[0].id, "content": str(123456 * 654321)}]
r2 = client.chat.completions.create(model=a.model, messages=msgs, tools=tools, temperature=1.0, top_p=0.95, max_tokens=4000, extra_body=extra)
final = r2.choices[0].message.content or ""
print("turn 2:", final[:200].replace("\n", " "))
ok = "80779853376" in final.replace(",", "").replace(" ", "")
print("PASS" if ok else "WARN: final answer missing the tool result (round trip may be broken)")
sys.exit(0 if ok else 1)
