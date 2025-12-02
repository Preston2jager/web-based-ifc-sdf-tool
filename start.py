import os
import sys
import time
import torch
import shutil
import subprocess
import contextlib
from pathlib import Path
from typing import Tuple, List
import streamlit as st
import streamlit.components.v1 as components

from src.cls_cfg_loader import SDF_config_loader
from src.cls_sdf_runner import SDF_Runner
from src.cls_ifc_exporter import SDF_exporter
from src.cls_sdf_sampler import SDF_Sampler
from src.cls_sdf_reconstructor import SDF_reconstructor

from src.ui.cls_log_writer import StreamlitLogWriter
from src.ui.ui_actions import render_actions_panel

st.set_page_config(page_title="IFC Auto-Decoder Tool", layout="wide")


class IFCAutoDecoderApp:
    def __init__(self):
        """Initialize the IFC Auto-Decoder application with default configurations."""
        self.base_dir = Path(__file__).parent
        self.ui_dir = self.base_dir / "src" / "ui"
        self.ss = st.session_state
        self.cfg = self._load_config()
        self._init_state()
        self.log_placeholder = None

    # ========== Internal API ==========
    def _init_folders(self):
        root_dir = Path(__file__).resolve().parent
        targets = [
            "Data/Training/Configs",
            "Data/Training/Raw_IFC",
            "Data/Training/Expanded_IFC",
            "Data/Training/Objects",
            "Data/Training/Converted_SDF",
            "Data/Training/Trained"
        ]
        clr_targets = [
            "Data/Training/Expanded_IFC",
            "Data/Training/Objects",
            "Data/Training/Converted_SDF"
        ]
        for target in targets:
            full_path = root_dir / target
            full_path.mkdir(parents=True, exist_ok=True)
            if target in clr_targets:
                for item in full_path.iterdir():
                    if item.is_file():
                        item.unlink()
                    elif item.is_dir():
                        shutil.rmtree(item)
        configs_dir = root_dir / "Data/Training/Configs" 
        if not any(configs_dir.iterdir()):  # 目录为空
            scr = root_dir / "src" / "template.yaml"
            tar = configs_dir / "configs.yaml"
            shutil.copy2(scr, tar)
            print("Status:✅ Template configs.yaml copied")
        print("Status:✅ Project folders and files ready")

    def _init_state(self) -> None:
        """Initialize session state with default values for UI components."""
        self.ss.setdefault(
            "hint",
            "Click a button on the left to run a demo step. Notes appear here.",
        )
        self.ss.setdefault("status", "Idle")
        self.ss.setdefault("log_lines", [])
        self.ss.setdefault("pending_action", None)
        self.ss.setdefault("recon_panel_open", False)
        self.ss.setdefault("recon_selected_folder", None)
        self.ss.setdefault("config_panel_open", False)
        self.ss.setdefault("incr_pre_panel_open", False)
        self.ss.setdefault("incr_pre_selected_folder", None)
        self.ss.setdefault("incr_train_panel_open", False)
        self.ss.setdefault("incr_train_selected_folder", None)

    def _capture_to_log(self) -> contextlib.AbstractContextManager:
        """Create a context manager that redirects stdout/stderr to the log display.
        
        Returns:
            Context manager that captures output to Streamlit log placeholder.
        """
        def flush_cb(text: str):
            if self.log_placeholder is not None:
                self.log_placeholder.code(text, language="bash")

        writer = StreamlitLogWriter(self.ss, flush_callback=flush_cb)

        @contextlib.contextmanager
        def _cm():
            old_out, old_err = sys.stdout, sys.stderr
            try:
                sys.stdout = writer
                sys.stderr = writer
                yield
            finally:
                sys.stdout = old_out
                sys.stderr = old_err

        return _cm()

    def _load_config(self) -> SDF_config_loader:
        """Load configuration from primary or fallback YAML file.
        
        Returns:
            Loaded configuration object.
            
        Raises:
            FileNotFoundError: If neither primary nor fallback config exists.
        """
        primary_path = "Data/Training/Configs/configs.yaml"
        fallback_path = "template.yaml"
        if os.path.exists(primary_path):
            cfg = SDF_config_loader(primary_path)
        elif os.path.exists(fallback_path):
            cfg = SDF_config_loader(fallback_path)
        else:
            raise FileNotFoundError("No config file found")
        return cfg

    def _display_interrupt(self) -> None:
        """Print training interruption message to console."""
        print("❌ Previous training interrupted")

    def _open_in_default_editor(self, path: str) -> None:
        """Open a file in the system's default editor.
        
        Args:
            path: Absolute or relative path to the file.
            
        Raises:
            FileNotFoundError: If the specified file doesn't exist.
            RuntimeError: If no suitable text editor is found.
        """
        path = os.path.abspath(path)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        if sys.platform.startswith("win"):
            os.startfile(path)
            return
        if sys.platform == "darwin":
            subprocess.Popen(["open", path])
            return
        editor = os.environ.get("EDITOR")
        if editor:
            subprocess.Popen([editor, path])
            return
        for candidate in ("xdg-open", "nano", "vim", "vi"):
            exe = shutil.which(candidate)
            if exe:
                subprocess.Popen([exe, path])
                return
        raise RuntimeError("No suitable text editor found (set $EDITOR or install nano/vim)")

    def _get_subfolders(self, path: str) -> Tuple[List[str], List[str]]:
        """Retrieve all subfolders in a given directory.
        
        Args:
            path: Directory path to scan for subfolders.
            
        Returns:
            Tuple of (folder_names, folder_paths).
        """
        if not os.path.exists(path):
            return [], []

        file_names = []
        subfolders = [
            os.path.join(path, name)
            for name in os.listdir(path)
            if os.path.isdir(os.path.join(path, name))
        ]
        for folder in subfolders:
            file_base = os.path.basename(folder)
            file_names.append(file_base)
        return file_names, subfolders

    def _load_css(self) -> None:
        """Load and inject custom CSS styles from UI/style.css."""
        css_path = self.ui_dir / "style.css"
        if css_path.exists():
            css = css_path.read_text(encoding="utf-8")
            st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
        else:
            st.warning("⚠️ UI/style.css not found")

    def _render_topbar(self) -> None:
        """Render the top navigation bar with current status."""
        status = self.ss.get("status", "Idle")
        html_path = self.ui_dir / "topbar.html"
        if html_path.exists():
            html = html_path.read_text(encoding="utf-8")
            html = html.replace("{{STATUS}}", status)
            st.markdown(html, unsafe_allow_html=True)
        else:
            st.warning("⚠️ UI/topbar.html not found")

    def _on_action(self, action_id: str, label: str, hint: str) -> None:
        """Handle button click actions from the left sidebar.
        
        Args:
            action_id: Unique identifier for the action.
            label: Display label for the action.
            hint: Help text to show in the notes section.
        """
        self.ss.hint = hint
        self.ss.status = f"{label} — In progress…"

        with self._capture_to_log():
            print(f"[{action_id}] {label} clicked.\n")

            if action_id == "init":
                self._init_folders()
                self.ss.status = f"{label} — Finished ✓"

            elif action_id == "configs":
                self.ss["config_panel_open"] = True
                self.ss.status = "Config Editor — Please download or upload config"
                print("[configs] Config editor panel opened.")

            elif action_id == "pre_all":
                start = time.time()
                if not hasattr(self, "cfg") or self.cfg is None:
                    print("Status:❌ Please load config first")
                    self.ss.status = "Error: No config loaded"
                    return
                exporter = SDF_exporter(self.cfg)
                exporter.export()
                end1 = time.time()
                sampler = SDF_Sampler(self.cfg)
                sampler.sample()
                end2 = time.time()
                print(f"Export Time: {end1 - start:.4f} s")
                print(f"Sampling Time: {end2 - end1:.4f} s")
                self.ss.status = f"{label} — Finished ✓"
            
            elif action_id == "pre_export":
                start = time.time()
                if not hasattr(self, "cfg") or self.cfg is None:
                    print("Status:❌ Please load config first")
                    self.ss.status = "Error: No config loaded"
                    return
                sampler = SDF_Sampler(self.cfg)
                sampler.sample()
                end1 = time.time()
                print(f"Sampling Time: {end1 - start:.4f} s")
                self.ss.status = f"{label} — Finished ✓"

            elif action_id == "pre_sample":
                start = time.time()
                if not hasattr(self, "cfg") or self.cfg is None:
                    print("Status:❌ Please load config first")
                    self.ss.status = "Error: No config loaded"
                    return
                sampler = SDF_Sampler(self.cfg)
                sampler.sample()
                end1 = time.time()
                print(f"Sampling Time: {end1 - start:.4f} s")
                self.ss.status = f"{label} — Finished ✓"

            elif action_id == "train":
                start = time.time()
                try:
                    torch.cuda.init()
                    torch.cuda.empty_cache()
                    Runner = SDF_Runner(self.cfg)
                    Runner.execute()
                    end = time.time()
                    print(f"Training Time: {end - start:.4f} s")
                    self.ss.status = f"{label} — Finished ✓"
                except KeyboardInterrupt:
                    self._display_interrupt()
                    self.ss.status = f"{label} — Interrupted ⚠️"

            elif action_id == "incr_pre":
                self.ss["incr_pre_panel_open"] = True
                self.ss.status = "Incremental Pre-processing — Please select a pretrained model"
                print("[incr_pre] Incremental pre-processing panel opened.")

            elif action_id == "incr_train":
                self.ss["incr_train_panel_open"] = True
                self.ss.status = "Fine-tune Training — Please select a pretrained model"
                print("[incr_train] Fine-tune training panel opened.")

            elif action_id == "recon":
                self.ss["recon_panel_open"] = True
                self.ss.status = "Reconstruction — Please select a folder"
                print("[recon] Reconstruction panel opened. Please select a folder to reconstruct.")

    def _render_config_panel(self) -> None:
        """Render configuration file editor panel with download/upload functionality."""
        st.markdown("---")
        st.subheader("⚙️ Edit Configuration")

        config_file = "Data/Training/Configs/configs.yaml"

        if not os.path.exists(config_file):
            st.error(f"❌ Config file not found: {config_file}")
            if st.button("❌ Close", key="btn_config_close_error"):
                self.ss["config_panel_open"] = False
                self.ss.status = "Config editor closed"
                st.rerun()
            return

        with open(config_file, 'r', encoding='utf-8') as f:
            config_content = f.read()

        st.download_button(
            label="📥 Download Config File",
            data=config_content,
            file_name="configs.yaml",
            mime="text/yaml",
            key="btn_download_config"
        )

        st.text_area(
            "Current Configuration (Read-only)",
            value=config_content,
            height=300,
            key="config_content_display",
            disabled=True
        )

        st.markdown("---")
        st.markdown("**Upload Modified Config:**")

        uploaded_file = st.file_uploader(
            "Choose a YAML file",
            type=['yaml', 'yml'],
            key="config_uploader"
        )

        col1, col2 = st.columns(2)

        with col1:
            if uploaded_file is not None:
                if st.button("✅ Save Uploaded Config", key="btn_save_config", type="primary"):
                    with self._capture_to_log():
                        try:
                            backup_file = config_file + ".backup"
                            shutil.copy(config_file, backup_file)
                            print(f"📋 Backup created: {backup_file}")

                            content = uploaded_file.read().decode('utf-8')
                            with open(config_file, 'w', encoding='utf-8') as f:
                                f.write(content)

                            print("✅ Config file updated successfully!")
                            print("📁 File: {config_file}")

                            self.cfg = self._load_config()
                            print("🔄 Configuration reloaded")

                            self.ss.status = "Config updated ✓"
                            self.ss["config_panel_open"] = False

                        except Exception as e:
                            print(f"❌ Failed to save config: {e}")
                            self.ss.status = "Config update failed ❌"

                    st.rerun()

        with col2:
            if st.button("❌ Cancel", key="btn_config_cancel"):
                self.ss["config_panel_open"] = False
                self.ss.status = "Config editor closed"
                with self._capture_to_log():
                    print("[configs] Config editor closed by user.")
                st.rerun()

    def _render_incr_pre_panel(self) -> None:
        """Render incremental preprocessing panel for selecting and processing pretrained models."""
        st.markdown("---")
        st.subheader("📦 Incremental Pre-processing")

        raw_file_names, raw_subfolders = self._get_subfolders(
            self.cfg.Pathes.Trained_SDF_folder_path
        )

        if not raw_file_names:
            st.info("📂 No pretrained models found in Trained_SDF_folder_path.")
            if st.button("❌ Close panel", key="btn_incr_pre_close_empty"):
                self.ss["incr_pre_panel_open"] = False
                self.ss.status = "Incremental pre-processing cancelled"
                st.rerun()
            return

        sorted_data = sorted(
            zip(raw_file_names, raw_subfolders),
            key=lambda x: os.path.getmtime(x[1]),
            reverse=True,
        )
        file_names, subfolders = zip(*sorted_data)

        default_idx = 0
        prev_folder = self.ss.get("incr_pre_selected_folder")
        if prev_folder in subfolders:
            default_idx = subfolders.index(prev_folder)

        st.markdown("**Select a pretrained model folder:**")
        selected_name = st.selectbox(
            "Folder name (sorted by modified time)",
            file_names,
            index=default_idx,
            key="incr_pre_selectbox",
            label_visibility="collapsed"
        )

        idx = file_names.index(selected_name)
        selected_folder = subfolders[idx]

        mtime = os.path.getmtime(selected_folder)
        mtime_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mtime))
        st.caption(f"📅 Last modified: {mtime_str}")
        st.caption(f"📁 Path: `{selected_folder}`")

        self.ss["incr_pre_selected_folder"] = selected_folder

        col1, col2 = st.columns(2)
        with col1:
            do_process = st.button("✅ Start Pre-processing", key="btn_incr_pre_start", type="primary")
        with col2:
            cancel = st.button("❌ Cancel", key="btn_incr_pre_cancel")

        if do_process:
            with self._capture_to_log():
                print(f"\n{'='*60}")
                print("[Incremental Pre-processing] Starting...")
                print(f"Selected folder: {selected_folder}")
                print(f"Folder name: {selected_name}")
                print(f"{'='*60}\n")

                start = time.time()
                try:
                    exporter = SDF_exporter(self.cfg)
                    exporter.export()
                    end1 = time.time()
                    print(f"Export Time: {end1 - start:.4f} s")

                    sampler = SDF_Sampler(self.cfg, selected_folder)
                    sampler.sample(selected_folder)
                    end2 = time.time()
                    print(f"Sampling Time: {end2 - end1:.4f} s")

                    print(f"\n{'='*60}")
                    print("✅ Incremental pre-processing completed successfully!")
                    print(f"Total time: {end2 - start:.4f} s")
                    print(f"{'='*60}\n")

                    self.ss.status = f"Incremental Pre-processing from '{selected_name}' — Finished ✓"
                except Exception as e:
                    print("\n❌ Incremental pre-processing failed with error:")
                    print(f"Error: {str(e)}")
                    self.ss.status = "Incremental Pre-processing — Failed ❌"

            self.ss["incr_pre_panel_open"] = False
            st.rerun()

        if cancel:
            self.ss["incr_pre_panel_open"] = False
            self.ss.status = "Incremental pre-processing cancelled"
            with self._capture_to_log():
                print("[Incremental Pre-processing] Cancelled by user.")
            st.rerun()

    def _render_incr_train_panel(self) -> None:
        """Render fine-tune training panel for selecting and training pretrained models."""
        st.markdown("---")
        st.subheader("🎓 Fine-tune Training")

        raw_file_names, raw_subfolders = self._get_subfolders(
            self.cfg.Pathes.Trained_SDF_folder_path
        )

        if not raw_file_names:
            st.info("📂 No pretrained models found in Trained_SDF_folder_path.")
            if st.button("❌ Close panel", key="btn_incr_train_close_empty"):
                self.ss["incr_train_panel_open"] = False
                self.ss.status = "Fine-tune training cancelled"
                st.rerun()
            return

        sorted_data = sorted(
            zip(raw_file_names, raw_subfolders),
            key=lambda x: os.path.getmtime(x[1]),
            reverse=True,
        )
        file_names, subfolders = zip(*sorted_data)

        default_idx = 0
        prev_folder = self.ss.get("incr_train_selected_folder")
        if prev_folder in subfolders:
            default_idx = subfolders.index(prev_folder)

        st.markdown("**Select a pretrained model folder:**")
        selected_name = st.selectbox(
            "Folder name (sorted by modified time)",
            file_names,
            index=default_idx,
            key="incr_train_selectbox",
            label_visibility="collapsed"
        )

        idx = file_names.index(selected_name)
        selected_folder = subfolders[idx]

        mtime = os.path.getmtime(selected_folder)
        mtime_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mtime))
        st.caption(f"📅 Last modified: {mtime_str}")
        st.caption(f"📁 Path: `{selected_folder}`")

        self.ss["incr_train_selected_folder"] = selected_folder

        col1, col2 = st.columns(2)
        with col1:
            do_train = st.button("✅ Start Training", key="btn_incr_train_start", type="primary")
        with col2:
            cancel = st.button("❌ Cancel", key="btn_incr_train_cancel")

        if do_train:
            with self._capture_to_log():
                print(f"\n{'='*60}")
                print("[Fine-tune Training] Starting...")
                print(f"Selected folder: {selected_folder}")
                print(f"Folder name: {selected_name}")
                print(f"{'='*60}\n")

                start = time.time()
                try:
                    torch.cuda.init()
                    torch.cuda.empty_cache()
                    Runner = SDF_Runner(self.cfg, selected_folder)
                    Runner.execute()
                    end = time.time()

                    print(f"\n{'='*60}")
                    print("✅ Fine-tune training completed successfully!")
                    print(f"Training Time: {end - start:.4f} s")
                    print(f"{'='*60}\n")

                    self.ss.status = f"Fine-tune Training from '{selected_name}' — Finished ✓"
                except KeyboardInterrupt:
                    self._display_interrupt()
                    self.ss.status = "Fine-tune Training — Interrupted ⚠️"
                except Exception as e:
                    print("\n❌ Fine-tune training failed with error:")
                    print(f"Error: {str(e)}")
                    self.ss.status = "Fine-tune Training — Failed ❌"

            self.ss["incr_train_panel_open"] = False
            st.rerun()

        if cancel:
            self.ss["incr_train_panel_open"] = False
            self.ss.status = "Fine-tune training cancelled"
            with self._capture_to_log():
                print("[Fine-tune Training] Cancelled by user.")
            st.rerun()

    def _render_reconstruction_panel(self) -> None:
        """Render reconstruction panel for selecting and reconstructing trained models."""
        st.markdown("---")
        st.subheader("🔨 Reconstruct Geometry")
        raw_file_names, raw_subfolders = self._get_subfolders(
            self.cfg.Pathes.Trained_SDF_folder_path
        )

        if not raw_file_names:
            st.info("📂 No trained runs found in Trained_SDF_folder_path.")
            if st.button("❌ Close panel", key="btn_recon_close_empty"):
                self.ss["recon_panel_open"] = False
                self.ss.status = "Reconstruction cancelled"
                st.rerun()
            return

        sorted_data = sorted(
            zip(raw_file_names, raw_subfolders),
            key=lambda x: os.path.getmtime(x[1]),
            reverse=True,
        )
        file_names, subfolders = zip(*sorted_data)

        default_idx = 0
        prev_folder = self.ss.get("recon_selected_folder")
        if prev_folder in subfolders:
            default_idx = subfolders.index(prev_folder)

        st.markdown("**Select a trained model folder:**")
        selected_name = st.selectbox(
            "Folder name (sorted by modified time)",
            file_names,
            index=default_idx,
            key="recon_selectbox",
            label_visibility="collapsed"
        )

        idx = file_names.index(selected_name)
        selected_folder = subfolders[idx]

        mtime = os.path.getmtime(selected_folder)
        mtime_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(mtime))
        st.caption(f"📅 Last modified: {mtime_str}")
        st.caption(f"📁 Path: `{selected_folder}`")

        self.ss["recon_selected_folder"] = selected_folder

        col1, col2 = st.columns(2)
        with col1:
            do_recon = st.button("✅ Start Reconstruction", key="btn_recon_start", type="primary")
        with col2:
            cancel = st.button("❌ Cancel", key="btn_recon_cancel")

        if do_recon:
            with self._capture_to_log():
                print(f"\n{'='*60}")
                print("[Reconstruction] Starting reconstruction...")
                print(f"Selected folder: {selected_folder}")
                print(f"Folder name: {selected_name}")
                print(f"{'='*60}\n")

                start = time.time()
                try:
                    Reconstructor = SDF_reconstructor()
                    Reconstructor.reconstruct_all(self.cfg, selected_folder)
                    end = time.time()
                    print(f"\n{'='*60}")
                    print("✅ Reconstruction completed successfully!")
                    print(f"Total time: {end - start:.4f} s")
                    print(f"{'='*60}\n")

                    self.ss.status = f"Reconstruction from '{selected_name}' — Finished ✓"
                except Exception as e:
                    print("\n❌ Reconstruction failed with error:")
                    print(f"Error: {str(e)}")
                    self.ss.status = "Reconstruction — Failed ❌"

            self.ss["recon_panel_open"] = False
            st.rerun()

        if cancel:
            self.ss["recon_panel_open"] = False
            self.ss.status = "Reconstruction cancelled"
            with self._capture_to_log():
                print("[Reconstruction] Cancelled by user.")
            st.rerun()

    def _render_right(self) -> None:
        """Render the right panel with 3D OBJ viewer component."""
        st.subheader("3D Viewer (OBJ • world coords • Z-up • rotate only)")
        st.caption("Tip: Click **Choose folder** to load local .obj files (client-side; no upload).")

        html_path = self.ui_dir / "viewer.html"
        if html_path.exists():
            html = html_path.read_text(encoding="utf-8")
            components.html(html, height=900)
        else:
            st.error("❌ UI/viewer.html not found")

    # ========== Public API ==========

    def run(self) -> None:
        """Run the main application loop and render all UI components."""
        self._load_css()
        self._render_topbar()

        left_col, mid_col, right_col = st.columns([0.6, 1.6, 1.2], gap="large")

        with left_col:
            render_actions_panel(self.ss)

        with mid_col:
            st.subheader("📝 Notes")
            st.write(self.ss.hint)

            st.subheader("📊 Status")
            st.markdown(
                f"""
                <div class="status-card">
                  <span class="status-dot"></span>{self.ss.status}
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.subheader("📄 Output")
            if self.log_placeholder is None:
                self.log_placeholder = st.empty()

            self.log_placeholder.code(
                "".join(self.ss.get("log_lines", [])) or "(No output yet)",
                language="bash",
            )

            if self.ss.get("config_panel_open", False):
                self._render_config_panel()

            if self.ss.get("incr_pre_panel_open", False):
                self._render_incr_pre_panel()

            if self.ss.get("incr_train_panel_open", False):
                self._render_incr_train_panel()

            if self.ss.get("recon_panel_open", False):
                self._render_reconstruction_panel()

        with right_col:
            self._render_right()

        pending = self.ss.get("pending_action")
        if pending is not None:
            aid, label, hint = pending
            self.ss["pending_action"] = None
            self._on_action(aid, label, hint)
            st.rerun()


def main():
    """Entry point for the Streamlit application."""
    app = IFCAutoDecoderApp()
    app.run()


if __name__ == "__main__":
    if os.environ.get("RUNNING_UNDER_STREAMLIT") != "1":
        env = dict(os.environ)
        env["RUNNING_UNDER_STREAMLIT"] = "1"
        os.execvpe(
            sys.executable,
            [sys.executable, "-m", "streamlit", "run", __file__],
            env,
        )
    else:
        main()