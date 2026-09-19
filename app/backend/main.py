"""FastAPI backend for the EPS Dynamic-GNN live demo.

Wraps the real simulator (`src.simulation.eps_simulator.simulate_episode`)
and the four trained model checkpoints so the frontend can: run a fresh
simulated episode for a chosen fault class, see all four models' live
predictions on it, and scrub through the dynamic-GNN's learned adjacency
A(t) / attention weights over the orbit window.

Run with (from the eps-dynamic-gnn/ project root):
    uvicorn app.backend.main:app --reload --port 8000
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.dataset import load_metadata
from src.simulation.eps_simulator import simulate_episode
from src.simulation.faults import FAULT_CLASSES
from src.train import build_model, get_device

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
CKPT_DIR = os.path.join(RESULTS_DIR, "checkpoints")
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")
MODEL_ORDER = ["cnn_baseline", "lstm_baseline", "static_gnn", "dynamic_gnn"]
DISPLAY_NAMES = {
    "cnn_baseline": "CNN",
    "lstm_baseline": "LSTM",
    "static_gnn": "Static-GNN (ACS-style)",
    "dynamic_gnn": "Dynamic-GNN (proposed)",
}

app = FastAPI(title="EPS Dynamic-GNN Live Demo")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_state = {}


def _load_everything():
    meta = load_metadata()
    device = get_device()
    mean = np.array(meta["normalization"]["mean"], dtype=np.float32)   # (N,F)
    std = np.array(meta["normalization"]["std"], dtype=np.float32)

    models = {}
    for name in MODEL_ORDER:
        ckpt_path = os.path.join(CKPT_DIR, f"{name}.pt")
        if not os.path.exists(ckpt_path):
            continue
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = build_model(ckpt["config"]["model"], ckpt["config"]["model_params"],
                             len(meta["node_names"]), len(meta["feature_names"]),
                             len(meta["fault_classes"])).to(device)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        models[name] = dict(model=model, n_params=ckpt.get("n_params"))

    _state.update(meta=meta, device=device, mean=mean, std=std, models=models)


@app.on_event("startup")
def startup():
    _load_everything()


class RunRequest(BaseModel):
    fault_type: str
    seed: int | None = None


@torch.no_grad()
def run_all_models(X: np.ndarray):
    """X: (T, N, F) raw (unnormalized) telemetry. Returns per-model predictions
    and, for dynamic_gnn, the learned adjacency/attention over time."""
    meta = _state["meta"]
    device = _state["device"]
    mean, std = _state["mean"], _state["std"]

    x_norm = (X - mean[None]) / std[None]
    x_t = torch.from_numpy(x_norm.astype(np.float32)).unsqueeze(0).to(device)  # (1,T,N,F)

    predictions = {}
    dynamic_extras = None
    for name, entry in _state["models"].items():
        model = entry["model"]
        if name in ("dynamic_gnn", "static_gnn"):
            logits, extras = model(x_t, return_interpret=True)
            if name == "dynamic_gnn":
                dynamic_extras = dict(
                    a_soft=extras["a_soft"][0].cpu().numpy(),      # (T,N,N)
                    attention=extras["attention"][0].cpu().numpy(),
                )
        else:
            logits = model(x_t)
        probs = F.softmax(logits, dim=-1)[0].cpu().numpy()
        pred_idx = int(probs.argmax())
        predictions[name] = dict(
            display_name=DISPLAY_NAMES[name],
            probs={c: float(p) for c, p in zip(meta["fault_classes"], probs)},
            pred_label=meta["fault_classes"][pred_idx],
            n_params=entry["n_params"],
        )
    return predictions, dynamic_extras


@app.get("/api/meta")
def api_meta():
    meta = _state["meta"]
    return dict(
        node_names=meta["node_names"],
        feature_names=meta["feature_names"],
        fault_classes=meta["fault_classes"],
        static_edges=meta["static_edges"],
        n_steps=meta["n_steps"],
        dt_store_sec=meta["dt_store_sec"],
        t_orbit_sec=meta["t_orbit_sec"],
        models_loaded=list(_state["models"].keys()),
    )


@app.post("/api/run")
def api_run(req: RunRequest):
    if req.fault_type not in FAULT_CLASSES:
        raise HTTPException(400, f"unknown fault_type, must be one of {FAULT_CLASSES}")

    rng = np.random.default_rng(req.seed) if req.seed is not None else np.random.default_rng()
    X, info = simulate_episode(req.fault_type, rng=rng)   # (T,N,F)
    meta = _state["meta"]
    node_names = meta["node_names"]
    feature_names = meta["feature_names"]

    telemetry = {}
    for i, node in enumerate(node_names):
        telemetry[node] = {feature_names[j]: X[:, i, j].tolist() for j in range(len(feature_names))}

    predictions, dynamic_extras = run_all_models(X)

    ev = info["fault_event"]
    response = dict(
        fault_type=req.fault_type,
        t_minutes=(info["t"] / 60.0).tolist(),
        eclipse_mask=info["eclipse_mask"].tolist(),
        fault_mask=info["fault_mask"].tolist(),
        fault_event=dict(onset_sec=ev.onset_sec, duration_sec=ev.duration_sec,
                          severity=ev.severity, mode=ev.mode),
        telemetry=telemetry,
        power_balance_residual=info["power_balance_residual"].tolist(),
        predictions=predictions,
        true_label=req.fault_type,
    )
    if dynamic_extras is not None:
        response["dynamic_graph"] = dict(
            a_soft=dynamic_extras["a_soft"].round(4).tolist(),
            attention=dynamic_extras["attention"].round(4).tolist(),
        )
    return response


@app.get("/api/performance")
def api_performance():
    path = os.path.join(RESULTS_DIR, "metrics", "performance.json")
    if not os.path.exists(path):
        raise HTTPException(404, "no performance.json found; run src.evaluate first")
    with open(path) as f:
        data = json.load(f)
    for r in data["results"]:
        r["display_name"] = DISPLAY_NAMES.get(r["model"], r["model"])
    return data


@app.get("/api/history/{model_name}")
def api_history(model_name: str):
    path = os.path.join(RESULTS_DIR, "metrics", f"{model_name}_history.json")
    if not os.path.exists(path):
        raise HTTPException(404, f"no history for {model_name}")
    with open(path) as f:
        return json.load(f)


@app.get("/api/history_all")
def api_history_all():
    out = {}
    for name in MODEL_ORDER:
        path = os.path.join(RESULTS_DIR, "metrics", f"{name}_history.json")
        if os.path.exists(path):
            with open(path) as f:
                out[name] = json.load(f)
    return out


app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
