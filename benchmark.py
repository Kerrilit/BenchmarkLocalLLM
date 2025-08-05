#!/usr/bin/env python3
import requests
import time
from concurrent.futures import ThreadPoolExecutor

BASE_URL = "http://127.0.0.1:8080/v1/chat/completions"

PROMPTS = [
    "What is the capital of France?",
    "Tell me a joke about computers.",
    "List three animals that can fly.",
    "What does CPU stand for?",
    "Who painted the Mona Lisa?",
    "How many continents are there?",
    "What year did World War II end?",
    "What's the square root of 144?",
    "Name a programming language created in 1995.",
    "What is LLaMA model?"
]

def call_once(prompt: str, timeout=1500.0):
    payload = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user",   "content": prompt}
        ],
        "max_tokens": 128,
        "temperature": 0.1,
        "stream": False
    }
    t0 = time.time()
    # внутри call_once перед requests.post:
    # t_send = time.time()
    # print(f"sending at {t_send:.6f}")
    resp = requests.post(BASE_URL, json=payload, timeout=timeout)
    dt = time.time() - t0
    resp.raise_for_status()
    jr = resp.json()
    tok = jr.get("usage", {}).get("total_tokens")
    return dt, tok or 0

def run_benchmark(concurrency: int, reps: int = 3):
    print(f"\n→ Конкурентность: {concurrency} запросов")
    total_wall = 0.0
    total_reqs = 0
    total_tokens = 0
    lat_sum = 0.0

    for rep in range(reps):
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            ts = [PROMPTS[i % len(PROMPTS)] for i in range(concurrency)]
            t0 = time.time()
            futures = [ex.submit(call_once, prompt) for prompt in ts]
            results = [f.result() for f in futures]
        loop_time = time.time() - t0

        times, toks = zip(*results)
        total_wall += loop_time
        total_reqs += len(results)
        total_tokens += sum(toks)
        lat_sum += sum(times)

    avg_wall = total_wall / reps
    avg_latency = (lat_sum / total_reqs) if total_reqs else float("nan")
    tok_per_sec = total_tokens / total_wall if total_wall else float("nan")
    req_per_sec = total_reqs / total_wall if total_wall else float("nan")

    print(f"  Ответов/с (req/s): {req_per_sec:.3f}")
    print(f"  Токенов/с      (tok/s): {tok_per_sec:.1f}")
    print(f"  Средняя задержка       : {avg_latency:.2f} s (≈{avg_latency*1000:.0f} ms)")
    print(f"  Общее время wall-time  : {avg_wall:.2f} s")
    print(f"  Всего токенов           : {total_tokens}\n")

if __name__ == "__main__":
    import sys
    levels = [1, 5, 10, 50, 100]
    if len(sys.argv) == 2:
        try:
            lvl = int(sys.argv[1])
            levels = [lvl]
        except:
            pass

    for lvl in levels:
        run_benchmark(lvl)