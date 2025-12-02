
import os
import glob
import trimesh
import numpy as np
import point_cloud_utils as pcu
from trimesh import repair
from tqdm import tqdm
from typing import Optional, Tuple, Dict, Any

def clean_mesh(
    mesh: trimesh.Trimesh,
    min_face_area: float = 1e-8,
    min_total_area: float = 1e-6
) -> Optional[trimesh.Trimesh]:
    """Clean and repair mesh by removing degenerate geometry.
    
    Performs comprehensive mesh cleaning including removal of NaN/Inf values,
    degenerate faces, duplicate faces, and faces below minimum area threshold.
    Also attempts to fill holes and fix normals.
    
    Args:
        mesh: Input mesh to clean.
        min_face_area: Minimum area threshold for individual faces.
        min_total_area: Minimum total mesh area threshold.
        
    Returns:
        Cleaned mesh, or None if mesh becomes invalid after cleaning.
    """
    if mesh.vertices.shape[0] == 0 or mesh.faces.shape[0] == 0:
        return None

    mesh.remove_infinite_values()
    mesh.remove_unreferenced_vertices()
    mesh.remove_degenerate_faces()
    mesh.remove_duplicate_faces()
    mesh.remove_unreferenced_vertices()

    repair.fill_holes(mesh)
    repair.fix_normals(mesh)

    if hasattr(mesh, "area_faces"):
        areas = mesh.area_faces
        keep = areas > min_face_area
        if keep.sum() == 0:
            return None
        mesh.update_faces(keep)
        mesh.remove_unreferenced_vertices()

    if mesh.area < min_total_area:
        return None
    return mesh


def normalize_trimesh_to_unit_box(
    mesh: trimesh.Trimesh
) -> Tuple[trimesh.Trimesh, np.ndarray, float]:
    """Normalize mesh to unit bounding box with longest edge equal to 1.
    
    Centers the mesh at origin and scales so the longest dimension equals 1.
    
    Args:
        mesh: Input mesh to normalize.
        
    Returns:
        Tuple of (normalized_mesh, original_center, scale_factor).
    """
    lo, hi = mesh.bounds
    center = 0.5 * (lo + hi)
    extent = hi - lo
    max_extent = float(extent.max())
    
    if max_extent <= 0:
        return mesh, center, 1.0

    mesh.vertices = (mesh.vertices - center) / max_extent
    return mesh, center, max_extent


class SDF_Sampler:
    """Sample points and compute signed distance functions for mesh objects.
    
    Handles point sampling on mesh surfaces with multiple sampling strategies
    (on-surface, jittered near-surface) and computes SDF values for training.
    """

    def __init__(self, cfg, target_folder: Optional[str] = None):
        """Initialize SDF sampler with configuration.
        
        Args:
            cfg: Configuration object containing sampling parameters and paths.
            target_folder: Optional target folder for fine-tuning mode.
        """
        self.scale = cfg.Extract.Scale
        self.radii = [
            cfg.Extract.R1_radius,
            cfg.Extract.R2_radius,
            cfg.Extract.R3_radius,
            cfg.Extract.R4_radius,
            cfg.Extract.R5_radius,
        ]
        self.radii = [r * self.scale for r in self.radii]
        self.nums = [
            cfg.Extract.R1_num,
            cfg.Extract.R2_num,
            cfg.Extract.R3_num,
            cfg.Extract.R4_num,
            cfg.Extract.R5_num,
        ]
        self.min_per_face = int(cfg.Extract.Minimal_per_surface)
        self.dense_of_samples_on_surface = int(cfg.Extract.Dense_of_samples_on_surface)

        self.obj_files = glob.glob(
            os.path.join(cfg.Pathes.Objects_folder_path, "*.obj")
        )

        if target_folder is None:
            self.output_path = cfg.Pathes.Converted_SDF_folder_path
        else:
            self.output_path = target_folder
            print(self.output_path)

    # ========== Internal API ==========

    def _merge_sample_dicts(self, base_folder: str) -> None:
        """Merge fine-tuning samples with original training samples.
        
        Combines samples from pretrained model with new samples for incremental training.
        Reassigns indices to avoid conflicts and saves merged dictionaries.
        
        Args:
            base_folder: Base folder containing both original and fine-tune sample files.
        """
        samples_dict_1 = np.load(
            os.path.join(base_folder, "samples_dict.npy"), allow_pickle=True
        ).item()
        str2int_1 = np.load(
            os.path.join(base_folder, "idx_str2int_dict.npy"), allow_pickle=True
        ).item()
        int2str_1 = np.load(
            os.path.join(base_folder, "idx_int2str_dict.npy"), allow_pickle=True
        ).item()

        samples_dict_2 = np.load(
            os.path.join(base_folder, "f_samples_dict.npy"), allow_pickle=True
        ).item()
        str2int_2 = np.load(
            os.path.join(base_folder, "f_idx_str2int_dict.npy"), allow_pickle=True
        ).item()
        int2str_2 = np.load(
            os.path.join(base_folder, "f_idx_int2str_dict.npy"), allow_pickle=True
        ).item()

        merged_samples = dict()
        merged_str2int = dict()
        merged_int2str = dict()

        merged_samples.update(samples_dict_1)
        merged_str2int.update(str2int_1)
        merged_int2str.update(int2str_1)

        offset = max(merged_samples.keys()) + 1 if merged_samples else 0

        for old_idx, obj_id_str in int2str_2.items():
            new_idx = offset
            offset += 1
            merged_samples[new_idx] = samples_dict_2[old_idx]
            merged_str2int[obj_id_str] = new_idx
            merged_int2str[new_idx] = obj_id_str

        os.makedirs(base_folder, exist_ok=True)
        np.save(os.path.join(base_folder, "samples_dict.npy"), merged_samples)
        np.save(os.path.join(base_folder, "idx_str2int_dict.npy"), merged_str2int)
        np.save(os.path.join(base_folder, "idx_int2str_dict.npy"), merged_int2str)

        to_delete = [
            "f_samples_dict.npy",
            "f_idx_str2int_dict.npy",
            "f_idx_int2str_dict.npy",
        ]
        for fname in to_delete:
            fpath = os.path.join(base_folder, fname)
            if os.path.exists(fpath):
                os.remove(fpath)
                print(f"Deleted: {fname}")
            else:
                print(f"Not found (skip): {fname}")
        print("Status:✅ Merged files saved to", base_folder)

    def _combine_sample_latent(
        self, 
        samples: np.ndarray, 
        latent_class: np.ndarray
    ) -> np.ndarray:
        """Combine sample coordinates with latent class indices.
        
        Args:
            samples: Sample point coordinates of shape (N, 3).
            latent_class: Latent class index array of shape (1,).
            
        Returns:
            Combined array of shape (N, 4) with format [class_idx, x, y, z].
        """
        latent_class_full = np.tile(latent_class, (samples.shape[0], 1))
        return np.hstack((latent_class_full, samples))

    def _compute_triangle_areas(
        self, 
        vertices: np.ndarray, 
        faces: np.ndarray
    ) -> np.ndarray:
        """Compute areas of all triangular faces.
        
        Args:
            vertices: Vertex coordinates of shape (V, 3).
            faces: Face indices of shape (F, 3).
            
        Returns:
            Array of face areas of shape (F,).
        """
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        cross_prod = np.cross(v1 - v0, v2 - v0)
        return 0.5 * np.linalg.norm(cross_prod, axis=1)

    def _sample_points_on_face(
        self,
        v0: np.ndarray,
        v1: np.ndarray,
        v2: np.ndarray,
        num_points: int
    ) -> np.ndarray:
        """Sample random points uniformly on a triangular face.
        
        Uses barycentric coordinate sampling for uniform distribution.
        
        Args:
            v0: First vertex coordinates.
            v1: Second vertex coordinates.
            v2: Third vertex coordinates.
            num_points: Number of points to sample.
            
        Returns:
            Array of sampled points of shape (num_points, 3).
        """
        r1 = np.sqrt(np.random.rand(num_points, 1))
        r2 = np.random.rand(num_points, 1)
        points = (1 - r1) * v0 + r1 * (1 - r2) * v1 + r1 * r2 * v2
        return points

    def _sample_on_sphere_surface(
        self,
        center: np.ndarray,
        radius: float,
        num_points: int
    ) -> np.ndarray:
        """Sample points uniformly on sphere surface.
        
        Args:
            center: Center coordinates of sphere.
            radius: Sphere radius.
            num_points: Number of points to sample.
            
        Returns:
            Array of sampled points of shape (num_points, 3).
        """
        directions = np.random.normal(size=(num_points, 3))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        points = center + radius * directions
        return points

    def _jitter_points_nearby(self, points: np.ndarray) -> np.ndarray:
        """Create jittered samples near surface points at multiple radii.
        
        Generates additional samples around each surface point using configured
        radius and count parameters for each sampling layer.
        
        Args:
            points: Surface point coordinates of shape (N, 3).
            
        Returns:
            Array of combined surface and jittered points.
        """
        N = len(points)
        if N == 0:
            return points

        all_offsets = []
        for radius, n in zip(self.radii, self.nums):
            if n <= 0 or radius <= 0:
                continue
            offset_dirs = np.random.normal(size=(N * n, 3))
            offset_dirs /= np.linalg.norm(offset_dirs, axis=1, keepdims=True)
            offset_mags = np.random.uniform(0, radius, size=(N * n, 1))
            offsets = offset_dirs * offset_mags
            base_points = np.repeat(points, n, axis=0)
            new_points = base_points + offsets
            all_offsets.append(new_points)

        if all_offsets:
            all_points = np.vstack(all_offsets + [points])
        else:
            all_points = points
        return all_points

    def _sample_points_and_compute_sdf(
        self,
        verts: np.ndarray,
        faces: np.ndarray,
        total_area: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Sample points on mesh surface and compute their SDF values.
        
        Performs stratified sampling to ensure minimum samples per face,
        creates jittered near-surface samples, and computes signed distances.
        
        Args:
            verts: Vertex coordinates of shape (V, 3).
            faces: Face indices of shape (F, 3).
            total_area: Total surface area of mesh.
            
        Returns:
            Tuple of (points, sdf_values) where:
                - points: Array of sampled coordinates of shape (N, 3)
                - sdf_values: Array of SDF values of shape (N,)
            Returns empty arrays if mesh is invalid.
        """
        if faces is None or len(faces) == 0:
            print("Status:❌ _sample_points_and_compute_sdf: faces is empty.")
            return np.empty((0, 3)), np.empty((0,), dtype=np.float32)

        surface_sample_num = int(float(total_area) * self.dense_of_samples_on_surface)
        min_global = self.min_per_face * len(faces)
        surface_sample_num = max(surface_sample_num, min_global)

        if surface_sample_num <= 0:
            print(
                f"Status:❌ Mesh has near-zero total area ({total_area}), "
                f"computed surface_sample_num={surface_sample_num}. Skip."
            )
            return np.empty((0, 3)), np.empty((0,), dtype=np.float32)

        fid_surf, bc_surf = pcu.sample_mesh_random(verts, faces, surface_sample_num)
        p_surf_init = pcu.interpolate_barycentric_coords(
            faces, fid_surf, bc_surf, verts
        )

        face_counts = np.bincount(fid_surf, minlength=len(faces))
        deficits = np.maximum(0, self.min_per_face - face_counts)

        if deficits.any():
            tri_verts = verts[faces]
            extra = []
            for f_idx, n_missing in enumerate(deficits):
                if n_missing == 0:
                    continue
                v0, v1, v2 = tri_verts[f_idx]
                extra.append(self._sample_points_on_face(v0, v1, v2, n_missing))
            if extra:
                p_surf_init = np.vstack([p_surf_init, *extra])

        p_jitter = self._jitter_points_nearby(p_surf_init)
        p_total = np.vstack([p_jitter, p_surf_init])
        sdf, _, _ = pcu.signed_distance_to_mesh(p_total, verts, faces)
        return p_total, sdf

    def _sdf_analysis(self, samples_dict: Dict[int, Dict[str, Any]]) -> None:
        """Analyze SDF value distribution across all samples.
        
        Computes and prints statistics about positive/negative/zero SDF values.
        
        Args:
            samples_dict: Dictionary containing SDF data for all objects.
        """
        all_sdf = []
        for obj_idx in samples_dict:
            if "sdf" in samples_dict[obj_idx]:
                sdf = samples_dict[obj_idx]["sdf"]
                if sdf is not None and sdf.size > 0:
                    all_sdf.append(sdf)
        if not all_sdf:
            print("❌ No SDF data found in samples_dict.")
            return
        all_sdf = np.concatenate(all_sdf, axis=0)
        self._sdf_sign_ratio(all_sdf)

    def _sdf_sign_ratio(self, sdf_values: np.ndarray) -> None:
        """Print statistics about SDF value signs.
        
        Args:
            sdf_values: Array of SDF values to analyze.
        """
        sdf_values = np.asarray(sdf_values).flatten()
        total = len(sdf_values)
        if total == 0:
            print("SDF Sign Ratio: total=0 (empty input)")
            return
        pos_count = np.sum(sdf_values > 0)
        neg_count = np.sum(sdf_values < 0)
        zero_count = np.sum(sdf_values == 0)
        print("  SDF Sign Ratio Summary:")
        print(f"  Total       : {total}")
        print(f"  Positive    : {pos_count} ({pos_count / total:.2%})")
        print(f"  Negative    : {neg_count} ({neg_count / total:.2%})")
        print(f"  Zero        : {zero_count} ({zero_count / total:.2%})")

    # ========== Public API ==========

    def sample(self, target_folder: Optional[str] = None) -> None:
        """Sample points and compute SDF values for all mesh objects.
        
        Processes all OBJ files, samples points on surfaces, computes SDF values,
        and saves results. Handles both normal training and fine-tuning modes.
        
        Args:
            target_folder: Optional folder for fine-tuning mode. If provided,
                          generates separate files that will be merged with
                          existing training data.
        """
        samples_dict = dict()
        idx_str2int_dict = dict()
        idx_int2str_dict = dict()

        for obj_idx, obj_path in enumerate(
            tqdm(self.obj_files, desc="Processing OBJ files")
        ):
            obj_idx_str = os.path.splitext(os.path.basename(obj_path))[0]
            idx_str2int_dict[obj_idx_str] = obj_idx
            idx_int2str_dict[obj_idx] = obj_idx_str

            samples_dict[obj_idx] = dict()

            try:
                mesh_original = trimesh.load(obj_path, force="mesh", process=False)

                if not isinstance(mesh_original, trimesh.Trimesh):
                    print(
                        f"Status:❌ Error processing mesh {obj_path}: "
                        f"loaded object is not a trimesh.Trimesh"
                    )
                    continue

                mesh_original = clean_mesh(mesh_original)
                if mesh_original is None:
                    print(
                        f"Status:❌ Mesh {obj_path} is too degenerate after cleaning, skip."
                    )
                    continue

                if not mesh_original.is_watertight:
                    print(
                        f"Mesh {obj_path} is not watertight after cleaning, "
                        f"attempting to fill holes..."
                    )
                    mesh_original.fill_holes()
                    if not mesh_original.is_watertight:
                        print(
                            f"Warning: Mesh {obj_path} could not be fully repaired."
                        )

                obj_verts = np.asarray(mesh_original.vertices)
                obj_faces = np.asarray(mesh_original.faces, dtype=np.int32)

                if obj_faces.size == 0:
                    print(
                        f"Status:❌ Mesh {obj_path} has no faces, skip SDF sampling."
                    )
                    continue

                obj_total_area = self._compute_triangle_areas(
                    obj_verts, obj_faces
                ).sum()

            except Exception as e:
                print(f"Status:❌ Error processing mesh {obj_path}: {e}")
                continue

            p_total, sdf = self._sample_points_and_compute_sdf(
                obj_verts, obj_faces, obj_total_area
            )

            if p_total.size == 0 or sdf.size == 0:
                print(
                    f"Status:⚠️ Skip SDF for mesh {obj_path} "
                    f"(idx={obj_idx}) due to invalid geometry or zero samples."
                )
                samples_dict.pop(obj_idx, None)
                idx_str2int_dict.pop(obj_idx_str, None)
                idx_int2str_dict.pop(obj_idx, None)
                continue

            samples_dict[obj_idx]["sdf"] = sdf
            samples_dict[obj_idx]["samples_latent_class"] = self._combine_sample_latent(
                p_total, np.array([obj_idx], dtype=np.int32)
            )

        self._sdf_analysis(samples_dict)
        print("Status:✅ All SDF calculated")

        if target_folder is None:
            np.save(os.path.join(self.output_path, "samples_dict.npy"), samples_dict)
            np.save(
                os.path.join(self.output_path, "idx_str2int_dict.npy"),
                idx_str2int_dict,
            )
            np.save(
                os.path.join(self.output_path, "idx_int2str_dict.npy"),
                idx_int2str_dict,
            )
            print("Status:✅ Data files generated")
        else:
            np.save(os.path.join(self.output_path, "f_samples_dict.npy"), samples_dict)
            np.save(
                os.path.join(self.output_path, "f_idx_str2int_dict.npy"),
                idx_str2int_dict,
            )
            np.save(
                os.path.join(self.output_path, "f_idx_int2str_dict.npy"),
                idx_int2str_dict,
            )
            self._merge_sample_dicts(self.output_path)
            print("Status:✅ Fine-tune Data files generated")