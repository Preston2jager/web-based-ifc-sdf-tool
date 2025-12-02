"""
SDF-OBJ Viewer
==============

在指定 `root_dir/reconstruct` 目录下递归查找 .obj，
根据文件名中的 IFC 类别（如 `ifcwindow`）自动着色后可视化。
"""

import os
import glob
import random
import open3d as o3d
from typing import List, Tuple, Dict



# ────────────────────────────────────────────────────────────
# IFC → 颜色映射（0-1 浮点 RGB）
# 未映射的类别将随机生成一组近中灰色
# ────────────────────────────────────────────────────────────
IFC_COLOR_MAP: Dict[str, List[float]] = {
    "IFCWALL":   [0.88, 0.25, 0.25],  # 暖红
    "IFCFLOOR":  [0.80, 0.70, 0.50],  # 米黄
    "IFCWINDOW": [0.25, 0.55, 0.88],  # 天蓝
    "IFCDOOR":   [0.65, 0.35, 0.10],  # 棕色
    "IFCCOLUMN": [0.55, 0.55, 0.55],  # 中灰
    "IFCSLAB":   [0.65, 0.65, 0.65],  # 中灰
    "IFCBEAM":   [0.75, 0.75, 0.75],  # 中灰
}


class SDF_objviewer:
    def __init__(self, root_dir: str, recursive: bool = False):
        """
        Parameters
        ----------
        root_dir : str
            工程根目录；实际查找的是 ``root_dir/reconstruct``。
        recursive : bool, default False
            是否递归查找子文件夹中的 .obj。
        """
        self.root_dir  = os.path.join(os.path.abspath(root_dir), "reconstruct")
        self.recursive = recursive
        self.meshes: List[o3d.geometry.TriangleMesh] = []

    # ========== Public API ==========

    def view_all(self, window_size: Tuple[int, int] = (1280, 720)) -> None:
        """加载并显示所有 OBJ"""
        if not self.meshes:          # 首次调用才加载
            self._load_meshes()
        if not self.meshes:
            print(f"[SDF_objviewer] No OBJ files found in {self.root_dir}")
            return

        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        o3d.visualization.draw_geometries(
            self.meshes + [axis],
            window_name=f"OBJ Folder Viewer – {os.path.basename(self.root_dir)}",
            width=window_size[0],
            height=window_size[1],
            mesh_show_back_face=True,
        )

    # ========== Private Helpers ==========

    @staticmethod
    def _random_gray(base: float = 0.5, variation: float = 0.1) -> List[float]:
        g = max(0.0, min(1.0, random.uniform(base - variation, base + variation)))
        return [g, g, g]

    @staticmethod
    def _get_ifc_class(path: str) -> str:
        """
        解析形如 ``20250731_1_ifcwindow_guid_scaled.obj`` 的文件名，
        返回首个以 'ifc' 开头的 token（大写）。
        """
        stem = os.path.splitext(os.path.basename(path))[0]
        for token in stem.split("_"):
            if token.lower().startswith("ifc"):
                return token.upper()
        return ""   # 未识别

    def _load_meshes(self) -> None:
        """批量读取 .obj 并按 IFC 类别着色"""
        pattern   = "**/*.obj" if self.recursive else "*.obj"
        obj_paths = glob.glob(os.path.join(self.root_dir, pattern),
                              recursive=self.recursive)

        for path in sorted(obj_paths):
            mesh = o3d.io.read_triangle_mesh(path, enable_post_processing=True)
            if mesh.is_empty():
                print(f"[warn] Empty mesh skipped: {path}")
                continue

            mesh.compute_vertex_normals()

            ifc_cls = self._get_ifc_class(path)
            color   = IFC_COLOR_MAP.get(ifc_cls, self._random_gray())
            mesh.paint_uniform_color(color)

            self.meshes.append(mesh)

        print(f"[info] Loaded {len(self.meshes)} mesh(es) from '{self.root_dir}'")


# ────────────────────────────────────────────────────────────
# Quick test
# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    viewer = SDF_objviewer(root_dir=".", recursive=True)
    viewer.view_all()
