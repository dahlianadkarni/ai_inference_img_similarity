#!/usr/bin/env python3
"""Quick TRT EP test - single image only."""
import tritonclient.http as httpclient
import numpy as np
import time
import urllib.request
import re
import json

URL = "142.112.39.215:5399"
METRICS = "http://142.112.39.215:5037/metrics"

def get_m():
    r = urllib.request.urlopen(METRICS).read().decode()
    m = {}
    for l in r.split("\n"):
        if l.startswith("#") or not l.strip():
            continue
        x = re.match(r'(\w+)\{model="([^"]+)".*\}\s+([\d.e+]+)', l)
        if x:
            m.setdefault(x.group(2), {})[x.group(1)] = float(x.group(3))
    return m

c = httpclient.InferenceServerClient(url=URL)
t = np.random.rand(1, 3, 224, 224).astype(np.float32)
inp = [httpclient.InferInput("image", t.shape, "FP32")]
inp[0].set_data_from_numpy(t)
out = [httpclient.InferRequestedOutput("embedding")]

results = {}

for label, model in [("ONNX_CUDA_EP", "openclip_vit_b32"), ("ONNX_TRT_EP", "openclip_vit_b32_trt")]:
    print(f"\n=== {label} ({model}) ===")
    # Quick warmup (2 calls)
    for _ in range(2):
        c.infer(model, inp, outputs=out)

    m0 = get_m()
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        c.infer(model, inp, outputs=out)
        times.append((time.perf_counter() - t0) * 1000)
    m1 = get_m()

    times = np.array(times)
    reqs = m1[model]["nv_inference_request_success"] - m0[model]["nv_inference_request_success"]
    comp = (m1[model]["nv_inference_compute_infer_duration_us"] - m0[model]["nv_inference_compute_infer_duration_us"]) / reqs / 1000
    req_d = (m1[model]["nv_inference_request_duration_us"] - m0[model]["nv_inference_request_duration_us"]) / reqs / 1000

    print(f"  Client: mean={np.mean(times):.1f}ms  min={np.min(times):.1f}ms  p50={np.median(times):.1f}ms")
    print(f"  Server: compute={comp:.1f}ms  request={req_d:.1f}ms  (n={int(reqs)})")

    results[label] = {
        "client_mean_ms": float(np.mean(times)),
        "client_min_ms": float(np.min(times)),
        "client_p50_ms": float(np.median(times)),
        "server_compute_ms": float(comp),
        "server_request_ms": float(req_d),
    }

# Summary
print(f"\n{'='*60}")
print(f"  COMPARISON")
print(f"{'='*60}")
o = results["ONNX_CUDA_EP"]
t_r = results["ONNX_TRT_EP"]
print(f"                  ONNX(CUDA)    TRT EP(FP16)   Speedup")
print(f"  Client mean:  {o['client_mean_ms']:>9.1f}ms  {t_r['client_mean_ms']:>9.1f}ms  {o['client_mean_ms']/t_r['client_mean_ms']:>7.2f}x")
print(f"  Server comp:  {o['server_compute_ms']:>9.1f}ms  {t_r['server_compute_ms']:>9.1f}ms  {o['server_compute_ms']/t_r['server_compute_ms']:>7.2f}x")
print(f"  Server req:   {o['server_request_ms']:>9.1f}ms  {t_r['server_request_ms']:>9.1f}ms  {o['server_request_ms']/t_r['server_request_ms']:>7.2f}x")

with open("benchmark_results/tensorrt_ep_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to benchmark_results/tensorrt_ep_results.json")
