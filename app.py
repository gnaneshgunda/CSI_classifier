"""
app.py — CSI-HAR Streamlit Demo
================================
Upload a .pcap (Room A) or .csv (Room B) capture file and get:
  1. Detected room (A or B) with confidence
  2. Predicted activity with a confidence bar
  3. Optional: K-shot few-shot result using development support data

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import io
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

# ── repo root on sys.path ────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="CSI-HAR Demo",
    page_icon="📡",
    layout="centered",
)

# ── activity emoji map ────────────────────────────────────────────────────────
ACTIVITY_EMOJI = {
    "fall":    "🚨 Fall",
    "sit":     "🪑 Sit",
    "sitdown": "⬇️  Sit Down",
    "stand":   "🧍 Stand",
    "standup": "⬆️  Stand Up",
    "walk":    "🚶 Walk",
    "empty":   "🚫 Empty",
    "lying":   "🛌 Lying",
}
ROOM_EMOJI = {"A": "🏠 Room A", "B": "🏢 Room B"}

# ── colours for confidence bar ────────────────────────────────────────────────
def _conf_colour(conf: float) -> str:
    if conf >= 0.75:
        return "#2ecc71"   # green
    if conf >= 0.45:
        return "#f39c12"   # orange
    return "#e74c3c"       # red


# ═══════════════════════════════════════════════════════════════════════════════
# Cached resource loaders — run once per session
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading config…")
def _load_cfg():
    from src.config import load_config
    return load_config("configs/experiment.yaml")


@st.cache_resource(show_spinner="Loading encoder…")
def _load_encoder(cfg):
    from run_episode import load_encoder
    return load_encoder(cfg, condition="vicreg_aug_v4", device="cpu")


@st.cache_resource(show_spinner="Loading room classifier…")
def _load_router(cfg):
    from run_episode import load_router
    router = load_router(cfg)
    # Verify the clf actually loaded — if not, try loading RoomModel directly
    if getattr(router, "clf", None) is None:
        try:
            import joblib
            from src.room_model import RoomModel, RoomFeatureExtractor
            from pathlib import Path as _Path
            payload = joblib.load(str(_Path(cfg.paths.checkpoints_dir) / "room_model.pkl"))
            if isinstance(payload, dict) and "clfs" in payload:
                rm = RoomModel.__new__(RoomModel)
                rm._clfs    = payload["clfs"]
                rm._scalers = payload["scalers"]
                rm.best_name = payload.get("best_name", list(payload["clfs"].keys())[0])
                rm._is_fit  = True
                rm.rooms    = payload.get("rooms", ["A", "B"])
                import numpy as _np
                rm.classes_ = _np.array(rm.rooms)
                norm_path = _Path(cfg.paths.derived_dir) / "norm_stats.npz"
                rm._extractor = RoomFeatureExtractor(
                    log1p=True,
                    norm_stats_path=norm_path if norm_path.exists() else None,
                )
                router.clf = rm
        except Exception:
            pass
    return router


@st.cache_resource(show_spinner="Indexing development windows…")
def _load_index(cfg):
    index_csv = Path(cfg.paths.windows_index_csv)
    if not index_csv.exists():
        return None
    return pd.read_csv(str(index_csv))


@st.cache_resource(show_spinner="Pre-computing source embeddings (one-time, ~30s)…")
def _precompute_embeddings(cfg, _encoder):
    """Extract and cache all dev embeddings + CORAL alignment for the sweep."""
    from run_episode import extract_embeddings_for_df, load_normalizer, load_router
    from src.align import EmbeddingAligner

    index_df = _load_index(cfg)
    if index_df is None:
        return None

    windows_dir = Path(cfg.paths.windows_dir)
    normalizer  = load_normalizer(cfg, index_df[index_df["split"] == "dev"])
    router      = load_router(cfg)

    def _extract(mask):
        return extract_embeddings_for_df(
            index_df[mask], windows_dir, _encoder, normalizer, router
        )

    X_A_dev, y_A_dev, cids_A_dev, _ = _extract((index_df["room"] == "A") & (index_df["split"] == "dev"))
    X_B_dev, y_B_dev, cids_B_dev, _ = _extract((index_df["room"] == "B") & (index_df["split"] == "dev"))

    # Shared-only CORAL alignment
    from src.manifest import read_manifest
    aligner = EmbeddingAligner(cfg)
    all_embs   = np.vstack([X_A_dev, X_B_dev])
    all_rooms  = ["A"] * len(X_A_dev) + ["B"] * len(X_B_dev)
    all_labels = np.concatenate([y_A_dev, y_B_dev])
    try:
        aligner.load()
        # Validate dimension
        if aligner.stats.room_mean["A"].shape[0] != X_A_dev.shape[1]:
            raise ValueError("stale stats")
    except Exception:
        aligner.fit(all_embs, all_rooms, labels=all_labels, shared_only=True)

    X_A_dev_al = aligner.transform(X_A_dev, ["A"] * len(X_A_dev))
    X_B_dev_al = aligner.transform(X_B_dev, ["B"] * len(X_B_dev))

    manifest_rows = read_manifest(Path(cfg.paths.manifest_csv))

    # ── Train per-room logistic regression classifiers ──────────────────────
    # These replicate the Task 2 probe but are saved in the cache for inference.
    # Room A classifier: trained on Room A dev embeddings → predicts Room A activities
    # Room B classifier: trained on Room B dev embeddings → predicts Room B activities
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    lr_clfs = {}
    lr_scalers = {}
    for room_id, X_dev, y_dev in [("A", X_A_dev_al, y_A_dev),
                                   ("B", X_B_dev_al, y_B_dev)]:
        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X_dev)
        clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
        clf.fit(X_sc, y_dev)
        lr_clfs[room_id]    = clf
        lr_scalers[room_id] = scaler

    return {
        "X_A_dev": X_A_dev_al, "y_A_dev": y_A_dev, "cids_A_dev": cids_A_dev,
        "X_B_dev": X_B_dev_al, "y_B_dev": y_B_dev, "cids_B_dev": cids_B_dev,
        "aligner": aligner,
        "manifest_rows": manifest_rows,
        "lr_clfs":    lr_clfs,
        "lr_scalers": lr_scalers,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Core inference helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _pcap_to_amplitude(pcap_bytes: bytes) -> Optional[np.ndarray]:
    """Parse a .pcap file from bytes → (n, 256) float32 amplitude."""
    from src.pcap_reader import read_pcap
    from src.amplitude import frames_to_amplitude

    with tempfile.NamedTemporaryFile(suffix=".pcap", delete=False) as f:
        f.write(pcap_bytes)
        tmp_path = Path(f.name)

    try:
        frames, skip_stats = read_pcap(tmp_path)
        if not frames:
            return None
        amp_raw, valid = frames_to_amplitude(frames, fftshift=True)
        amp_raw[~valid] = np.nan
        return amp_raw
    except Exception as e:
        st.error(f"PCAP parsing failed: {e}")
        return None
    finally:
        tmp_path.unlink(missing_ok=True)


def _csv_to_amplitude(csv_bytes: bytes) -> Optional[np.ndarray]:
    """Parse a Room B .csv file from bytes → (n, 256) float32 amplitude."""
    from src.room_b_loader import load_csv

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="wb") as f:
        f.write(csv_bytes)
        tmp_path = Path(f.name)

    try:
        return load_csv(tmp_path)
    except Exception as e:
        st.error(f"CSV parsing failed: {e}")
        return None
    finally:
        tmp_path.unlink(missing_ok=True)


def _clean_and_window(amp_raw: np.ndarray, cfg) -> Optional[np.ndarray]:
    """Apply quality control and extract canonical (150, 256) window."""
    from src.cleaning import CleaningConfig, clean
    from src.windowing import canonical_window

    cc = CleaningConfig(
        max_nan_fraction_per_packet=cfg.cleaning.max_nan_fraction_per_packet,
        zero_energy_threshold=0.0,
        drop_consecutive_duplicates=True,
        spike_z_threshold=6.0,
    )
    cleaned, report = clean(amp_raw, cc)

    if report.n_clean < 10:
        st.warning(f"Only {report.n_clean} clean packets — capture may be too short.")
        if report.n_clean == 0:
            return None

    window = canonical_window(cleaned, cfg.window)
    return window   # (150, 256) float32 or None


def _detect_room(window: np.ndarray, router) -> tuple[str, float, float]:
    """Return (room, p_A, p_B) using the room classifier."""
    try:
        # RoomRouter wraps a RoomModel in .clf; try that first
        clf = getattr(router, "clf", None)
        if clf is None:
            return "?", 0.5, 0.5
        probs = clf.predict_proba(window)   # [P(A), P(B)]
        p_A, p_B = float(probs[0]), float(probs[1])
        room = "A" if p_A >= p_B else "B"
        return room, p_A, p_B
    except Exception as e:
        return "?", 0.5, 0.5


def _encode_window(window: np.ndarray, encoder) -> np.ndarray:
    """Normalize and encode a (150, 256) window → (256,) embedding."""
    import torch

    w = np.log1p(np.clip(window.astype(np.float64), 0, None))
    norm_win = ((w - w.mean()) / (w.std() + 1e-8)).astype(np.float32)
    t = torch.from_numpy(norm_win).unsqueeze(0)   # (1, 150, 256)
    with torch.no_grad():
        emb = encoder.encode(t).cpu().numpy()[0]  # (256,)
    return emb


def _predict_activity_lr(
    emb: np.ndarray,
    room: str,
    precomp: dict,
) -> tuple[str, float, dict]:
    """Predict activity using per-room logistic regression (Task 2 probe approach).
    
    This is the best single-file inference path — trained on same-room dev
    embeddings, so it doesn't suffer from the cross-room transfer gap.
    Expected accuracy: ~85-97% depending on room.
    """
    aligner    = precomp["aligner"]
    clf        = precomp["lr_clfs"][room]
    scaler     = precomp["lr_scalers"][room]

    emb_aligned = aligner.transform(emb[np.newaxis], [room])[0]
    X_sc = scaler.transform(emb_aligned[np.newaxis])
    probs_arr = clf.predict_proba(X_sc)[0]
    classes   = clf.classes_

    label = str(classes[int(np.argmax(probs_arr))])
    conf  = float(np.max(probs_arr))
    class_probs = {str(c): float(p) for c, p in zip(classes, probs_arr)}
    return label, conf, class_probs


def _predict_activity_zero_shot(
    emb: np.ndarray,
    room: str,
    precomp: dict,
    cfg,
) -> tuple[str, float, dict]:
    """Predict activity using source prototypes only (K=0, no support)."""
    from src.fewshot import PrototypeHead

    aligner     = precomp["aligner"]
    source_room = "B" if room == "A" else "A"

    emb_aligned = aligner.transform(emb[np.newaxis], [room])[0]

    X_src = precomp[f"X_{source_room}_dev"]
    y_src = precomp[f"y_{source_room}_dev"]

    head = PrototypeHead(target_room=room, cfg=cfg)
    head.fit(source_embeddings=X_src, source_labels=y_src)

    probs = head.predict_proba(emb_aligned[np.newaxis])[0]
    label = head.classes_[int(np.argmax(probs))]
    conf  = float(np.max(probs))
    class_probs = {cls: float(p) for cls, p in zip(head.classes_, probs)}

    import streamlit as _st
    with _st.expander("🔍 Debug — embedding space", expanded=False):
        import pandas as _pd
        emb_raw_norm   = float(np.linalg.norm(emb))
        emb_align_norm = float(np.linalg.norm(emb_aligned))
        proto_norms    = {c: float(np.linalg.norm(v)) for c, v in head.prototypes_.items()}
        cosine_sims    = {c: float(
                            np.dot(emb_aligned / (emb_align_norm + 1e-8),
                                   v / (float(np.linalg.norm(v)) + 1e-8)))
                          for c, v in head.prototypes_.items()}
        _st.write(f"Raw emb norm: **{emb_raw_norm:.3f}** | Aligned emb norm: **{emb_align_norm:.3f}**")
        _st.write(f"Source room: **{source_room}** → Target room: **{room}**")
        _st.dataframe(_pd.DataFrame({
            "class":           list(proto_norms.keys()),
            "proto_norm":      [round(v, 3) for v in proto_norms.values()],
            "cosine_sim":      [round(cosine_sims[c], 4) for c in proto_norms],
            "probability":     [round(float(p), 4) for p in probs],
        }), hide_index=True, use_container_width=True)

    return label, conf, class_probs


def _predict_activity_kshot(
    emb: np.ndarray,
    room: str,
    K: int,
    seed: int,
    precomp: dict,
    cfg,
) -> tuple[str, float, dict]:
    """Predict activity with K-shot support from dev set."""
    from src.fewshot import PrototypeHead, sample_support

    aligner     = precomp["aligner"]
    source_room = "B" if room == "A" else "A"

    # Align query to the same space as the pre-computed embeddings
    emb_aligned = aligner.transform(emb[np.newaxis], [room])[0]

    X_src  = precomp[f"X_{source_room}_dev"]
    y_src  = precomp[f"y_{source_room}_dev"]
    X_tgt  = precomp[f"X_{room}_dev"]        # already aligned target embeddings
    y_tgt  = precomp[f"y_{room}_dev"]
    cids   = precomp[f"cids_{room}_dev"]

    manifest_rows = precomp["manifest_rows"]
    support = sample_support(manifest_rows, target_room=room, K=K, seed=seed)
    sampled_cids = set(support.capture_ids)

    mask      = np.isin(cids, list(sampled_cids))
    X_support = X_tgt[mask]
    y_support = y_tgt[mask]
    cids_sup  = cids[mask]

    head = PrototypeHead(target_room=room, cfg=cfg)
    head.fit(
        source_embeddings=X_src, source_labels=y_src,
        target_support_embeddings=X_support,
        target_support_labels=y_support,
        target_support_capture_ids=cids_sup,
    )

    probs = head.predict_proba(emb_aligned[np.newaxis])[0]
    label = head.classes_[int(np.argmax(probs))]
    conf  = float(np.max(probs))
    class_probs = {cls: float(p) for cls, p in zip(head.classes_, probs)}

    import streamlit as _st
    with _st.expander("🔍 Debug — embedding space", expanded=False):
        import pandas as _pd
        emb_raw_norm   = float(np.linalg.norm(emb))
        emb_align_norm = float(np.linalg.norm(emb_aligned))
        proto_norms    = {c: float(np.linalg.norm(v)) for c, v in head.prototypes_.items()}
        cosine_sims    = {c: float(
                            np.dot(emb_aligned / (emb_align_norm + 1e-8),
                                   v / (float(np.linalg.norm(v)) + 1e-8)))
                          for c, v in head.prototypes_.items()}
        _st.write(f"**Raw norm:** {emb_raw_norm:.3f} | **Aligned norm:** {emb_align_norm:.3f}")
        _st.write(f"**Source → Target:** {source_room} → {room} | Support captures used: {int(mask.sum())}")
        _st.dataframe(_pd.DataFrame({
            "class":       list(proto_norms.keys()),
            "proto_norm":  [round(v, 3) for v in proto_norms.values()],
            "cosine_sim":  [round(cosine_sims[c], 4) for c in proto_norms],
            "probability": [round(float(p), 4) for p in probs],
        }), hide_index=True, use_container_width=True)

    return label, conf, class_probs


# ═══════════════════════════════════════════════════════════════════════════════
# UI rendering helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _render_room_result(room: str, p_A: float, p_B: float):
    if room == "?":
        st.warning("Room detection unavailable (no room classifier loaded).")
        return

    st.subheader("Step 1 — Detected Room")
    col1, col2 = st.columns(2)
    conf = p_A if room == "A" else p_B
    col1.metric("Room", ROOM_EMOJI.get(room, room))
    col2.metric("Confidence", f"{conf:.1%}")

    # Guard against NaN/Inf before charting
    p_A = float(p_A) if np.isfinite(p_A) else 0.5
    p_B = float(p_B) if np.isfinite(p_B) else 0.5

    st.markdown("**Room probabilities**")
    prob_df = pd.DataFrame({
        "Room": ["Room A", "Room B"],
        "Probability": [round(p_A, 4), round(p_B, 4)],
    })
    st.bar_chart(prob_df.set_index("Room"))


def _render_activity_result(label: str, conf: float, class_probs: dict, title: str):
    st.subheader(title)

    col1, col2 = st.columns([1, 2])
    emoji_label = ACTIVITY_EMOJI.get(label, label)
    col1.metric("Activity", emoji_label)
    col2.metric("Confidence", f"{conf:.1%}")

    # HTML confidence bar
    colour = _conf_colour(conf)
    bar_html = f"""
    <div style="background:#eee;border-radius:8px;height:20px;width:100%;margin-bottom:12px;">
      <div style="background:{colour};width:{conf*100:.1f}%;height:100%;
                  border-radius:8px;transition:width 0.4s;"></div>
    </div>
    """
    st.markdown(bar_html, unsafe_allow_html=True)

    # Per-class probability breakdown
    with st.expander("All class probabilities"):
        prob_df = (
            pd.DataFrame({"Activity": list(class_probs.keys()),
                          "Probability": list(class_probs.values())})
            .sort_values("Probability", ascending=False)
        )
        prob_df["Activity"] = prob_df["Activity"].map(
            lambda x: ACTIVITY_EMOJI.get(x, x)
        )
        st.dataframe(prob_df.style.format({"Probability": "{:.3f}"}),
                     use_container_width=True, hide_index=True)


def _render_window_preview(window: np.ndarray):
    with st.expander("CSI window preview (first 32 subcarriers)"):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 3))
        ax.imshow(
            np.log1p(window[:, :32]).T,
            aspect="auto", origin="lower", cmap="viridis",
        )
        ax.set_xlabel("Time step")
        ax.set_ylabel("Subcarrier index")
        ax.set_title("log1p(amplitude) — first 32 subcarriers")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Main app
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # ── header ────────────────────────────────────────────────────────────────
    st.title("📡 CSI-HAR — Cross-Room Activity Recognition")
    st.markdown(
        "Upload a WiFi CSI capture file and the system will detect the room "
        "and recognise the human activity using a self-supervised encoder "
        "and prototype-based few-shot classifier."
    )
    st.divider()

    # ── load models ───────────────────────────────────────────────────────────
    try:
        cfg     = _load_cfg()
        encoder = _load_encoder(cfg)
        router  = _load_router(cfg)
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        st.info("Make sure `configs/experiment.yaml`, `checkpoints/vicreg_aug_v4.pt`, "
                "and `checkpoints/room_model.pkl` are present.")
        return

    # ── sidebar controls ──────────────────────────────────────────────────────
    with st.sidebar:
        st.header("⚙️ Settings")

        mode = st.radio(
            "Prediction mode",
            ["🎯 Same-room LR (best accuracy)", "🔀 Cross-room few-shot (Task 3)"],
            index=0,
            help=(
                "Same-room LR: logistic regression trained on dev embeddings of the detected room. "
                "Expected F1 ~85-97%. | "
                "Cross-room few-shot: prototype head using source room + K target shots. "
                "Expected F1 ~35% — this is the Task 3 benchmark setting."
            ),
        )
        use_fewshot = "Cross-room" in mode

        K = st.slider("K (shots per exclusive activity)", 1, 20, 5,
                      disabled=not use_fewshot)
        seed = st.number_input("Random seed", min_value=0, max_value=99, value=42,
                               disabled=not use_fewshot)

        st.divider()
        st.markdown("**Model**")
        st.code("vicreg_aug_v4\n+ shared-only CORAL", language=None)

        st.markdown("**Activities per room**")
        st.markdown("🏠 Room A: fall, sit, sitdown, stand, standup, walk")
        st.markdown("🏢 Room B: empty, lying, sit, stand, walk")
        st.caption("Room is detected automatically from CSI content.")

    # ── file upload ───────────────────────────────────────────────────────────
    st.subheader("Upload a capture file")
    uploaded_file = st.file_uploader(
        "Drop a CSI capture file here (.pcap or .csv)",
        type=["pcap", "csv"],
    )

    if uploaded_file is None:
        st.info("Upload a `.pcap` or `.csv` file to begin. The room will be detected automatically.")
        return

    file_bytes = uploaded_file.read()
    is_pcap    = uploaded_file.name.lower().endswith(".pcap")

    st.divider()

    # ── process capture ───────────────────────────────────────────────────────
    with st.spinner("Processing capture…"):
        t0 = time.time()

        # 1. Try pcap first, then csv
        amp_raw = None
        if is_pcap:
            amp_raw = _pcap_to_amplitude(file_bytes)
        else:
            amp_raw = _csv_to_amplitude(file_bytes)

        # If pcap parse returned nothing, try csv as fallback (and vice versa)
        if amp_raw is None or amp_raw.shape[0] == 0:
            if is_pcap:
                amp_raw = _csv_to_amplitude(file_bytes)
            else:
                amp_raw = _pcap_to_amplitude(file_bytes)

        if amp_raw is None or amp_raw.shape[0] == 0:
            st.error("Could not parse the uploaded file as either .pcap or .csv.")
            return

        st.caption(f"Parsed: {amp_raw.shape[0]} packets × {amp_raw.shape[1]} subcarriers")

        # 2. Clean + window
        window = _clean_and_window(amp_raw, cfg)
        if window is None:
            st.error("No valid CSI data after quality control.")
            return

        # 3. Room detection
        room, p_A, p_B = _detect_room(window, router)

        # 4. Encode
        emb = _encode_window(window, encoder)

        elapsed = time.time() - t0

    st.caption(f"Preprocessing completed in {elapsed:.2f}s")

    # ── display window preview ────────────────────────────────────────────────
    _render_window_preview(window)

    # ── room result ───────────────────────────────────────────────────────────
    if room == "?":
        # Room classifier unavailable — fall back to highest-confidence guess
        # from the file extension as a weak prior, then warn
        room = "A" if is_pcap else "B"
        p_A  = 1.0 if room == "A" else 0.0
        p_B  = 1.0 - p_A
        st.warning(
            "Room classifier not loaded — defaulting to "
            f"{'Room A (pcap detected)' if room == 'A' else 'Room B (csv detected)'}. "
            "Make sure `checkpoints/room_model.pkl` exists for automatic detection."
        )

    _render_room_result(room, p_A, p_B)
    st.divider()

    # ── activity prediction ───────────────────────────────────────────────────
    with st.spinner("Preparing models (first run may take ~30s)…"):
        precomp = _precompute_embeddings(cfg, encoder)

    if precomp is None:
        st.error("Dev embeddings unavailable — derived/ folder missing.")
        return

    if not use_fewshot:
        # ── Same-room LR: best accuracy, no cross-room gap ────────────────
        with st.spinner("Predicting activity…"):
            label, conf, class_probs = _predict_activity_lr(emb, room, precomp)
        _render_activity_result(label, conf, class_probs,
                                "Step 2 — Predicted Activity (same-room classifier)")
    else:
        # ── Cross-room few-shot: Task 3 benchmark mode ────────────────────
        with st.spinner(f"Running K={K} cross-room few-shot prediction…"):
            label, conf, class_probs = _predict_activity_kshot(
                emb, room, K=K, seed=int(seed), precomp=precomp, cfg=cfg
            )
        _render_activity_result(label, conf, class_probs,
                                f"Step 2 — Predicted Activity (cross-room K={K})")

    st.divider()

    # ── output download ───────────────────────────────────────────────────────
    st.subheader("Download prediction")
    pred_df = pd.DataFrame([{
        "window_id": uploaded_file.name.replace(".", "_") + "__00000",
        "room": room,
        "label": label,
        "confidence": round(conf, 6),
    }])
    csv_out = pred_df.to_csv(index=False)
    st.download_button(
        label="⬇️ Download predictions.csv",
        data=csv_out,
        file_name="predictions.csv",
        mime="text/csv",
    )

    with st.expander("Prediction details"):
        st.dataframe(pred_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
