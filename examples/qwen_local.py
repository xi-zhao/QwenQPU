"""Optional local-only language integration. Not run by the replay/quick core demo."""
from pathlib import Path
import argparse
import atexit
import json
import os
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--model-path", type=Path, required=True)
ap.add_argument("--output", type=Path, required=True, help="Fresh JSON output path")
ap.add_argument("--max-generated-tokens", type=int, default=1024, help="Shared across planner and final reply; this is a demo, not the evaluation")
ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
args = ap.parse_args()
if not args.model_path.is_dir() or not (args.model_path / "config.json").is_file():
    ap.error("Existing local checkpoint with config.json required; no downloads are performed")
if args.output.exists():
    ap.error("Output exists; use a fresh file")
if not 128 <= args.max_generated_tokens <= 16384:
    ap.error("Choose a bounded total generated budget in 128..16384")

root = Path(__file__).resolve().parents[1]
for name, directory in [("TMPDIR",root/"review/tmp"),("HF_HOME",root/"review/cache/huggingface"),("TORCH_HOME",root/"review/cache/torch")]:
    directory.mkdir(parents=True,exist_ok=True)
    os.environ[name] = str(directory)
trace = {"status":"loading", "calls":[], "total_generated_budget":args.max_generated_tokens,
         "host_scheduled":True, "decoder_KV_fused":False, "model_downloads":False}
def persist():
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(trace,indent=2)+"\n")
atexit.register(persist)
previous_hook = sys.excepthook
def failure_hook(kind,value,tb):
    trace.update(status="failed",error_type=kind.__name__,error=str(value))
    previous_hook(kind,value,tb)
sys.excepthook = failure_hook

import torch
from transformers import AutoTokenizer, Qwen3_5ForCausalLM
from qwenqpu.runtime import execute_packet

tokenizer = AutoTokenizer.from_pretrained(str(args.model_path), local_files_only=True, trust_remote_code=False)
language = Qwen3_5ForCausalLM.from_pretrained(str(args.model_path), local_files_only=True, trust_remote_code=False,
                                          dtype=torch.bfloat16 if args.device=="cuda" else torch.float32,
                                          device_map={"":args.device}, attn_implementation="eager").eval().requires_grad_(False)
query = "Prepare a 16-qubit GHZ state from all zeros using H and a chain of CNOTs. Put the whole preparation in circuit, prefix=[], repeat=1, shots=1024."
system = "Output only strict JSON with n_qubits,prefix,circuit,repeat,shots. Gates: H q, X q, T q, CNOT control target. n_qubits1..16; <=1000 gates; shots1..10000. No prose."
used = 0
calls = []
trace["calls"] = calls

def generate(messages):
    global used
    remaining = args.max_generated_tokens-used
    if remaining <= 0:
        raise RuntimeError("Shared generation budget exhausted; no hidden retry")
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    x = tokenizer(text, return_tensors="pt").to(args.device)
    with torch.inference_mode():
        y = language.generate(**x, do_sample=False, max_new_tokens=remaining)
    ids = y[0, x["input_ids"].shape[1]:]
    used += len(ids)
    result = tokenizer.decode(ids, skip_special_tokens=True)
    eos = language.generation_config.eos_token_id
    eos_ids = [eos] if isinstance(eos,int) else list(eos or [])
    natural = bool(len(ids) and int(ids[-1]) in eos_ids)
    calls.append({"generated_tokens":len(ids), "input_tokens":x["input_ids"].shape[1], "text":result,
                  "natural_eos":natural, "remaining_budget_after":args.max_generated_tokens-used})
    trace["total_generated_tokens"] = used
    if not natural:
        raise RuntimeError("Generation ended without native EOS; recorded as incomplete, no retry")
    return result

raw = generate([{"role":"system","content":system},{"role":"user","content":query}])
packet = json.loads(raw)
if packet["n_qubits"] != 16 or packet["shots"] != 1024:
    raise ValueError("Model did not follow requested register/shot count; no automatic repair")
counts = execute_packet(packet, seed=123, device=args.device, language=language)
reply = generate([{"role":"system","content":"Return only the observed counts JSON unchanged. This is ideal classical simulation, not physical QPU execution."},
                  {"role":"user","content":query},{"role":"assistant","content":raw},
                  {"role":"user","content":"Observed counts: "+json.dumps(counts)}])
if json.loads(reply) != counts:
    raise ValueError("Final answer changed counts; no automatic repair")
trace.update(status="completed",query=query,packet=packet,counts=counts,reply=reply,total_generated_tokens=used)
persist()
print(args.output)
