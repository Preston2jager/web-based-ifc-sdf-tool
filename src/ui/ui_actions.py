from typing import List, Tuple
import streamlit as st

ACTIONS: List[Tuple[str, str, str]] = [
    ("init", "Init / Reset",
     "Clean up previously generated intermediate files and outputs while preserving the original IFC data. "
     "Run this step **before the first execution** or whenever you want to **reset the workspace** for a fresh start."),
    ("configs", "Editing configuration file",
     "Change all configuration here. Recommended as the first step after initialisation."),
    ("pre_all", "Data Preprocess (Export + Sampling)",
     "Perform the **full preprocessing pipeline**: export IFC data to graph/attributes and then run point sampling or voxelisation. "
     "Recommended as the first step after initialization."),
    ("pre_export", "  - Data Preprocess (Export only)",
     "Run only the **IFC → graph/attribute export** stage (demo mode). Useful for testing export configuration."),
    ("pre_sample", "  - Data Preprocess (Sampling only)",
     "Run only the **sampling** stage (demo mode), assuming exported graph data already exists."),
    ("train", "Auto-Decoder Training",
     "Run a **short demonstration training loop** of the Auto-Decoder model using preprocessed data."),
    ("incr_pre", "Incremental Preprocess",
     "Perform **incremental preprocessing** (demo mode): process only newly added or modified IFC files."),
    ("incr_train", "Incremental Training",
     "Perform **incremental training** (demo mode): continue training or fine-tuning from existing model weights."),
    ("recon", "Reconstruct Geometry",
     "Reconstruct geometry **from a selected folder**: load trained checkpoints and reconstruct all target samples."),
]


def render_actions_panel(ss):
    st.subheader("Actions")
    for aid, label, hint in ACTIONS:
        if st.button(label, key=aid, use_container_width=True):
            ss["pending_action"] = (aid, label, hint)
    st.markdown("---")
    if st.button("🧹 Clear Output", use_container_width=True):
        ss["log_lines"] = []
        ss["status"] = "Idle"
