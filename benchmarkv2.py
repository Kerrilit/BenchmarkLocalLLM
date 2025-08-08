#!/usr/bin/env python3
import requests
import time
from concurrent.futures import ThreadPoolExecutor
import sys
import json

BASE_URL = "http://127.0.0.1:8080/v1/chat/completions"
LOG_FILE = "benchmark.log"
SAMPLE_FILE = "sample_output.txt"

PROMPTS = [
    "Explain how the process of photosynthesis works and why it is essential for life on Earth.",
    "What are the main causes and consequences of climate change in the 21st century?",
    "Describe the architecture and training process of a modern large language model (LLM) like GPT-4.",
    "Discuss the impact of social media on mental health, particularly among teenagers.",
    "Compare and contrast capitalism and socialism as economic systems, including their pros and cons.",
    "What were the key events and outcomes of the Cold War, and how did it shape the modern world?",
    "How does the human brain process language, and what are some major theories in neurolinguistics?",
    "What is quantum computing, and how does it differ from classical computing in terms of principles and applications?",
    "Describe the plot, themes, and significance of George Orwell’s 1984 in the context of modern society.",
    "How do vaccines work, and why are they considered one of the most important achievements in medical science?"
]

# ----- Логгер -----
class Logger:
    def __init__(self, logfile):
        self.logfile = logfile
    def write(self, message):
        sys.__stdout__.write(message)
        with open(self.logfile, "a", encoding="utf-8") as f:
            f.write(message)
    def flush(self):
        sys.__stdout__.flush()

sys.stdout = Logger(LOG_FILE)

def call_once_stream(prompt: str, timeout=15000.0, save_sample=False):
    payload = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user",   "content": prompt}
        ],
        "max_tokens": 4096,
        "temperature": 0.1,
        "stream": True
    }

    t_start = time.time()
    first_token_time = None
    total_tokens = 0
    collected_text = ""

    with requests.post(BASE_URL, json=payload, timeout=timeout, stream=True) as r:
        r.raise_for_status()
        for chunk in r.iter_lines():
            if not chunk:
                continue
            if chunk.startswith(b"data: "):
                data = chunk[len(b"data: "):].strip()
                if data == b"[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue

                delta = obj["choices"][0]["delta"].get("content", "")
                if delta:
                    total_tokens += 1
                    collected_text += delta
                    if first_token_time is None:
                        first_token_time = time.time() - t_start

    total_time = time.time() - t_start

    if save_sample:
        with open(SAMPLE_FILE, "w", encoding="utf-8") as f:
            f.write("=== PROMPT ===\n")
            f.write(prompt.strip() + "\n\n")
            f.write("=== RESPONSE ===\n")
            f.write(collected_text.strip() + "\n")

    return total_time, total_tokens, (first_token_time or total_time)

def run_benchmark(concurrency: int, reps: int = 3):
    print(f"\n→ Конкурентность: {concurrency} запросов")
    total_wall = 0.0
    total_reqs = 0
    total_tokens = 0
    lat_sum = 0.0
    ttfb_sum = 0.0

    for rep in range(reps):
        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            ts = [PROMPTS[i % len(PROMPTS)] for i in range(concurrency)]
            t0 = time.time()
            futures = [
                ex.submit(
                    call_once_stream,
                    prompt,
                    save_sample=(concurrency == 1 and rep == 0 and i == 0)
                )
                for i, prompt in enumerate(ts)
            ]
            results = [f.result() for f in futures]
        loop_time = time.time() - t0

        times, toks, ttfb = zip(*results)
        total_wall += loop_time
        total_reqs += len(results)
        total_tokens += sum(toks)
        lat_sum += sum(times)
        ttfb_sum += sum(ttfb)

    avg_wall = total_wall / reps
    avg_latency = (lat_sum / total_reqs)
    avg_ttfb = (ttfb_sum / total_reqs)
    tok_per_sec = total_tokens / total_wall
    req_per_sec = total_reqs / total_wall

    print(f"  Ответов/с (req/s): {req_per_sec:.3f}")
    print(f"  Токенов/с      (tok/s): {tok_per_sec:.1f}")
    print(f"  Средняя задержка (полный ответ): {avg_latency:.2f} s")
    print(f"  Средний TTFB (первый токен):     {avg_ttfb:.2f} s")
    print(f"  Общее время wall-time  : {avg_wall:.2f} s")
    print(f"  Всего токенов           : {total_tokens}\n")

if __name__ == "__main__":
    levels = [1, 5, 10, 50, 100]
    if len(sys.argv) == 2:
        try:
            lvl = int(sys.argv[1])
            levels = [lvl]
        except:
            pass
    for lvl in levels:
        run_benchmark(lvl)
