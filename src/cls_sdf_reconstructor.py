
import os
import time
import torch
import numpy as np
import trimesh
from typing import Tuple, List
from collections import deque
from skimage.measure import marching_cubes
from tqdm import tqdm

from . import cls_sdf_model as sdf_model

class _Node:
    """Octree node for adaptive mesh reconstruction.
    
    Represents a cubic region in 3D space with position, size, and depth information.
    """
    __slots__ = ("ix", "iy", "iz", "size", "depth")
    
    def __init__(self, ix: int, iy: int, iz: int, size: int, depth: int):
        """Initialize octree node.
        
        Args:
            ix: X-axis grid index.
            iy: Y-axis grid index.
            iz: Z-axis grid index.
            size: Node size in grid units.
            depth: Depth level in octree (0 is root).
        """
        self.ix, self.iy, self.iz = ix, iy, iz
        self.size = size
        self.depth = depth


class SDF_reconstructor:
    """Reconstruct 3D meshes from trained SDF models using octree sampling.
    
    Implements adaptive octree subdivision to efficiently sample the SDF field
    and extract mesh surfaces using marching cubes algorithm.
    """

    # ========== Internal API ==========

    @staticmethod
    @torch.no_grad()
    def _eval_model(
        latent: torch.Tensor,
        model: torch.nn.Module,
        pts: torch.Tensor,
        device: torch.device,
        batch_sz: int = 196608
    ) -> torch.Tensor:
        """Evaluate SDF model at given points in batches.
        
        Args:
            latent: Latent code tensor of shape (1, latent_size).
            model: Trained SDF model.
            pts: Query points of shape (N, 3).
            device: Device for computation.
            batch_sz: Batch size for evaluation.
            
        Returns:
            SDF values at query points of shape (N,).
        """
        vals = []
        for i in range(0, pts.shape[0], batch_sz):
            chunk = pts[i : i + batch_sz]
            z = latent.expand(chunk.shape[0], -1)
            x = torch.cat([z, chunk], dim=1)
            sdf = model(x).squeeze(-1)
            vals.append(sdf.cpu())
        return torch.cat(vals)

    @torch.no_grad()
    def _reconstruct_octree(
        self,
        latent: torch.Tensor,
        model: torch.nn.Module,
        bbox: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
        *,
        init_res_root: int = 256,
        first_res: int = 8,
        top_k: int = 256,
        R_leaf: int = 16,
        max_depth: int = 5,
        device: torch.device = torch.device("cuda"),
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Reconstruct mesh using adaptive octree subdivision.
        
        Performs hierarchical spatial subdivision to identify regions containing
        the zero-level surface, then applies marching cubes to extract geometry.
        
        Args:
            latent: Latent code tensor of shape (1, latent_size).
            model: Trained SDF model.
            bbox: Bounding box as ((x_min, x_max), (y_min, y_max), (z_min, z_max)).
            init_res_root: Root grid resolution.
            first_res: First level coarse resolution for initial filtering.
            top_k: Number of top scoring blocks to keep in first level.
            R_leaf: Sampling resolution at leaf nodes.
            max_depth: Maximum octree depth.
            device: Device for computation.
            
        Returns:
            Tuple of (vertices, faces) as numpy arrays.
        """
        (xmin, xmax), (ymin, ymax), (zmin, zmax) = bbox
        max_len = max(xmax - xmin, ymax - ymin, zmax - zmin)
        step = max_len / init_res_root

        blk_size = init_res_root // first_res
        assert init_res_root % first_res == 0

        nodes_lvl1, scores = [], []
        for ix in range(first_res):
            for iy in range(first_res):
                for iz in range(first_res):
                    node = _Node(
                        ix * blk_size, iy * blk_size, iz * blk_size,
                        size=blk_size, depth=1
                    )

                    corner_pts = torch.tensor([
                        [xmin + (node.ix + dx) * step,
                         ymin + (node.iy + dy) * step,
                         zmin + (node.iz + dz) * step]
                        for dx in (0, blk_size)
                        for dy in (0, blk_size)
                        for dz in (0, blk_size)
                    ], device=device, dtype=torch.float32)

                    in_batch = torch.cat([latent.expand(8, -1), corner_pts], dim=1)
                    sdf_vals = model(in_batch).squeeze(-1)

                    if torch.prod(torch.sign(sdf_vals)) <= 0:
                        score = 0.0
                    else:
                        score = torch.min(torch.abs(sdf_vals)).item()

                    nodes_lvl1.append(node)
                    scores.append(score)

        keep_idx = np.argsort(scores)[:top_k]
        queue = deque([nodes_lvl1[i] for i in keep_idx])

        vertices: List[np.ndarray] = []
        faces: List[np.ndarray] = []
        vert_offset = 0

        while queue:
            node = queue.popleft()
            size = node.size
            half = size // 2

            corner_pts = torch.tensor([
                [xmin + (node.ix + dx) * step,
                 ymin + (node.iy + dy) * step,
                 zmin + (node.iz + dz) * step]
                for dx in (0, size)
                for dy in (0, size)
                for dz in (0, size)
            ], device=device, dtype=torch.float32)

            sdf_vals = model(
                torch.cat([latent.expand(8, -1), corner_pts], dim=1)
            ).squeeze(-1)
            iso_level = step * blk_size

            if torch.max(sdf_vals) > +iso_level and torch.min(sdf_vals) > +iso_level:
                continue
            if torch.max(sdf_vals) < -iso_level and torch.min(sdf_vals) < -iso_level:
                continue

            if node.depth < max_depth:
                for dx in (0, half):
                    for dy in (0, half):
                        for dz in (0, half):
                            queue.append(
                                _Node(
                                    node.ix + dx, node.iy + dy, node.iz + dz,
                                    half, node.depth + 1
                                )
                            )
                continue

            xs = torch.linspace(
                xmin + node.ix * step, 
                xmin + (node.ix + size) * step, 
                R_leaf + 1, 
                device=device
            )
            ys = torch.linspace(
                ymin + node.iy * step, 
                ymin + (node.iy + size) * step, 
                R_leaf + 1, 
                device=device
            )
            zs = torch.linspace(
                zmin + node.iz * step, 
                zmin + (node.iz + size) * step, 
                R_leaf + 1, 
                device=device
            )
            grid = torch.stack(
                torch.meshgrid(xs, ys, zs, indexing="ij"), 
                dim=-1
            ).reshape(-1, 3)

            vals = model(
                torch.cat([latent.expand(grid.shape[0], -1), grid], dim=1)
            ).squeeze(-1)
            sdf = vals.view(R_leaf + 1, R_leaf + 1, R_leaf + 1).cpu().numpy()

            spacing = (
                (xs[1] - xs[0]).item(),
                (ys[1] - ys[0]).item(),
                (zs[1] - zs[0]).item()
            )

            try:
                vs, fs, _, _ = marching_cubes(sdf, level=0.0, spacing=spacing)
            except ValueError:
                continue

            vs += np.array([
                xmin + node.ix * step,
                ymin + node.iy * step,
                zmin + node.iz * step
            ], dtype=np.float32)

            faces.append(fs + vert_offset)
            vertices.append(vs)
            vert_offset += vs.shape[0]

        if not vertices:
            return np.empty((0, 3), np.float32), np.empty((0, 3), np.int32)

        verts = np.concatenate(vertices, axis=0).astype(np.float32)
        facs = np.concatenate(faces, axis=0).astype(np.int32)
        return verts, facs

    def _reconstruct_object(
        self,
        target_folder: str,
        latent: torch.Tensor,
        obj_id_str: str,
        model: torch.nn.Module,
        cfg,
        device: torch.device,
    ) -> str:
        """Reconstruct a single object and save to OBJ file.
        
        Args:
            target_folder: Base folder for saving results.
            latent: Latent code tensor for the object.
            obj_id_str: Object identifier string for filename.
            model: Trained SDF model.
            cfg: Configuration object.
            device: Device for computation.
            
        Returns:
            Status string: "success", "empty", or "failed".
        """
        global_box = getattr(
            cfg.Reconstruct, 
            "global_box",
            ((-2.048, 2.048), (-2.048, 2.048), (-1.024, 3.072))
        )
        if global_box is None:
            raise ValueError(
                "No region_box or global_box provided in cfg.Reconstruct"
            )

        verts, faces = self._reconstruct_octree(
            latent, model, global_box, device=device
        )

        if verts.shape[0] == 0 or faces.shape[0] == 0:
            return "empty"

        recon_path = os.path.join(target_folder, "reconstruct")
        os.makedirs(recon_path, exist_ok=True)

        mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
        output_path = os.path.join(recon_path, f"{obj_id_str}.obj")
        mesh.export(output_path)

        return "success"

    # ========== Public API ==========

    @torch.no_grad()
    def reconstruct_all(self, cfg, target_folder: str) -> None:
        """Reconstruct all objects from a trained model.
        
        Loads trained model and latent codes, then reconstructs meshes for
        all objects in the training set. Provides detailed progress tracking
        and statistics.
        
        Args:
            cfg: Configuration object containing model parameters.
            target_folder: Folder containing trained model weights and results.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        weights_path = os.path.join(target_folder, "weights.pt")

        model = sdf_model.SDFModel(
            num_layers=cfg.Train.Num_layers,
            skip_connections=cfg.Train.Skip_connections,
            latent_size=cfg.Train.Latent_size,
            inner_dim=cfg.Train.Inner_dim,
        ).to(device)
        model.load_state_dict(
            torch.load(weights_path, map_location=device, weights_only=False)
        )
        model.eval()

        str2int_path = os.path.join(target_folder, "idx_str2int_dict.npy")
        results_path = os.path.join(target_folder, "results.npy")

        print(f"\n{'='*60}")
        print(f"📂 Loading data from: {target_folder}")
        print(f"   str2int path: {str2int_path}")
        print(f"   results path: {results_path}")

        str2int = np.load(str2int_path, allow_pickle=True).item()
        results = np.load(results_path, allow_pickle=True).item()
        latent_all = torch.tensor(results["best_latent_codes"], device=device)

        total_objects_in_dict = len(str2int)
        total_latent_codes = latent_all.shape[0]

        print(f"\n{'='*60}")
        print(f"📊 Data Summary:")
        print(f"   Objects in str2int dict: {total_objects_in_dict}")
        print(f"   Total latent codes:      {total_latent_codes}")
        print(f"   Latent codes shape:      {latent_all.shape}")
        print(f"   Device:                  {device}")

        if total_objects_in_dict != total_latent_codes:
            print(f"\n⚠️  WARNING: Mismatch detected!")
            print(f"   str2int has {total_objects_in_dict} objects")
            print(f"   but there are {total_latent_codes} latent codes")
            print(
                f"   Only the {total_objects_in_dict} objects in str2int "
                f"will be reconstructed!"
            )
            print(f"\n💡 Possible causes:")
            print(f"   1. The idx_str2int_dict.npy file is incomplete")
            print(f"   2. Training was interrupted or not all objects were saved")
            print(f"   3. The files are from different training runs")

        print(f"{'='*60}\n")

        success_count = 0
        fail_count = 0
        empty_count = 0

        for obj_id_str, obj_idx in tqdm(str2int.items(), desc="Reconstructing objects"):
            if obj_idx >= total_latent_codes:
                print(
                    f"❌ {obj_id_str}: Index {obj_idx} out of range "
                    f"(max: {total_latent_codes-1})"
                )
                fail_count += 1
                continue

            latent = latent_all[obj_idx].unsqueeze(0)
            start = time.time()

            try:
                result = self._reconstruct_object(
                    target_folder, latent, obj_id_str, model, cfg, device
                )
                end = time.time()

                if result == "success":
                    success_count += 1
                    print(f"✅ {obj_id_str}: {end - start:.3f}s")
                elif result == "empty":
                    empty_count += 1
                    print(f"⚠️  {obj_id_str}: Empty mesh (no geometry)")
                else:
                    fail_count += 1
                    print(f"❌ {obj_id_str}: Failed")

            except Exception as e:
                fail_count += 1
                print(f"❌ {obj_id_str}: Exception - {str(e)}")

        print(f"\n{'='*60}")
        print(f"🎯 Final Results:")
        print(f"   ✅ Success: {success_count}/{total_objects_in_dict}")
        print(f"   ⚠️  Empty:   {empty_count}/{total_objects_in_dict}")
        print(f"   ❌ Failed:  {fail_count}/{total_objects_in_dict}")

        if total_objects_in_dict != total_latent_codes:
            print(
                f"\n⚠️  Note: {total_latent_codes - total_objects_in_dict} "
                f"latent codes were not reconstructed"
            )
            print(f"   because they are missing from idx_str2int_dict.npy")

        print(f"{'='*60}\n")