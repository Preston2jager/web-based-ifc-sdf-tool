
import os
import shutil
import ifcopenshell
import ifcopenshell.geom
import numpy as np

from datetime import date
from typing import List, Tuple, Set, Dict, Optional
from collections import deque, defaultdict


class SDF_exporter:
    """Export IFC building elements to OBJ files with preprocessing.
    
    Handles IFC file expansion, geometry extraction, mesh splitting,
    and coordinate normalization for DeepSDF training.
    """

    def __init__(self, cfg):
        """Initialize exporter with configuration.
        
        Args:
            cfg: Configuration object containing export parameters and paths.
        """
        if bool(cfg.Extract.Data_expand) is False:
            self.Copies = 0
        else:
            self.Copies = cfg.Extract.Copies
        self.scale = cfg.Extract.Scale
        self.suffix = cfg.Extract.Suffix
        self.IFC_classes = cfg.Extract.Ifc_classes
        self.Raw_IFC_folder_path = cfg.Pathes.Raw_IFC_folder_path
        self.Expanded_IFC_folder_path = cfg.Pathes.Expanded_IFC_folder_path
        self.Objects_folder_path = cfg.Pathes.Objects_folder_path
        self.date = date.today().strftime("%Y%m%d")

        self.export_all_types = getattr(cfg.Extract, 'Export_all_types', False)
        self.global_box = getattr(cfg.Extract, 'Global_box', None)
        if self.global_box is None:
            self.global_box = [[-0.2, 0.2], [-0.2, 0.2], [-0.1, 0.3]]

    # ========== Internal API ==========

    def _ifc_move(self) -> None:
        """Copy IFC files from raw folder to expanded folder for processing."""
        original_files = []
        os.makedirs(self.Expanded_IFC_folder_path, exist_ok=True)
        
        for root, _, files in os.walk(self.Raw_IFC_folder_path):
            for filename in files:
                if filename.lower().endswith('.ifc'):
                    source_file_path = os.path.join(root, filename)
                    rel_path = os.path.relpath(source_file_path, self.Raw_IFC_folder_path)
                    target_file_path = os.path.join(self.Expanded_IFC_folder_path, rel_path)
                    os.makedirs(os.path.dirname(target_file_path), exist_ok=True)
                    shutil.copy(source_file_path, target_file_path)
                    original_files.append(target_file_path)
        
        file_count = len(original_files)
        print(
            f"Status:✅ {file_count} original IFC files moved into "
            f"{self.Expanded_IFC_folder_path}"
        )
        
        for files in original_files:
            self._ifc_create_copy(files, self.Copies)

    def _ifc_create_copy(self, source_file_path: str, copies: int) -> None:
        """Create copies of IFC file with regenerated GUIDs.
        
        Args:
            source_file_path: Path to source IFC file.
            copies: Number of copies to create.
        """
        if copies == 0:
            return

        dir_path = os.path.dirname(source_file_path)
        file_name_with_ext = os.path.basename(source_file_path)
        file_base, file_ext = os.path.splitext(file_name_with_ext)
        
        for i in range(1, copies + 1):
            target_file_name = f"{file_base}_copy{i}{file_ext}"
            target_file_path = os.path.join(dir_path, target_file_name)
            shutil.copy(source_file_path, target_file_path)
            print(f"IFC Expansion: {target_file_path} Created ✅")
            self._ifc_regenerate_guids(target_file_path)

    def _ifc_regenerate_guids(self, ifc_file: str) -> None:
        """Regenerate all GlobalIds in IFC file to make copies unique.
        
        Args:
            ifc_file: Path to IFC file to modify.
        """
        ifc_file_obj = ifcopenshell.open(ifc_file)
        for entity in ifc_file_obj.by_type("IfcRoot"):
            entity.GlobalId = ifcopenshell.guid.new()
        ifc_file_obj.write(ifc_file)

    def _ifcs_to_obj(self) -> None:
        """Convert all IFC files in expanded folder to OBJ format."""
        index = 1
        settings = ifcopenshell.geom.settings()
        settings.set(settings.USE_WORLD_COORDS, True)
        os.makedirs(self.Objects_folder_path, exist_ok=True)

        total_elements = 0

        for root, _, files in os.walk(self.Expanded_IFC_folder_path):
            for filename in files:
                if filename.lower().endswith('.ifc'):
                    ifc_file_path = os.path.join(root, filename)
                    ifc_file = ifcopenshell.open(ifc_file_path)

                    classes_to_process = self._get_classes_to_process(ifc_file)

                    print(f"\nProcessing file: {filename}")
                    print(f"  Detected object types: {classes_to_process}")

                    elements_count = self._obj_export(
                        self.Objects_folder_path,
                        ifc_file,
                        classes_to_process,
                        index,
                        settings
                    )
                    total_elements += elements_count
                    index += 1

        print(f"\n" + "="*60)
        print(
            f"Status:✅ All .obj files from {self.Expanded_IFC_folder_path} "
            f"have been scaled to {self.scale}"
        )
        print(f"Status:✅ In total {total_elements} elements have been converted to .obj files")
        print("="*60)

    def _get_classes_to_process(self, ifc_file) -> List[str]:
        """Get list of IFC classes to process from file.
        
        If export_all_types is enabled, automatically detects all building element
        types in the file. Otherwise uses configured IFC_classes list.
        
        Args:
            ifc_file: Opened IFC file object.
            
        Returns:
            List of IFC class names to process.
        """
        if self.export_all_types:
            detected_classes = set()

            base_classes = [
                'IfcBuildingElement',
                'IfcSpatialElement',
                'IfcElement',
            ]

            for base_class in base_classes:
                try:
                    elements = ifc_file.by_type(base_class)
                    for elem in elements:
                        if hasattr(elem, 'Representation') and elem.Representation is not None:
                            detected_classes.add(elem.is_a())
                except:
                    pass

            if not detected_classes:
                common_types = [
                    'IfcWall', 'IfcSlab', 'IfcBeam', 'IfcColumn', 'IfcWindow', 'IfcDoor',
                    'IfcBuildingElementProxy', 'IfcSpace', 'IfcCovering', 'IfcFurnishingElement',
                    'IfcPlate', 'IfcMember', 'IfcFooting', 'IfcRoof', 'IfcStair', 'IfcRailing',
                    'IfcCurtainWall', 'IfcBuildingElementPart'
                ]
                for type_name in common_types:
                    if ifc_file.by_type(type_name):
                        detected_classes.add(type_name)

            classes_list = sorted(list(detected_classes))

            if not classes_list:
                print("  ⚠️  Warning: No building elements detected, using configured type list")
                return self.IFC_classes

            print(f"  🔍 Auto-detection mode: Found {len(classes_list)} object types")
            return classes_list
        else:
            available_classes = []
            for class_name in self.IFC_classes:
                if ifc_file.by_type(class_name):
                    available_classes.append(class_name)

            if not available_classes:
                print(f"  ⚠️  Warning: Configured types {self.IFC_classes} not found in file!")
                print(f"  💡 Suggestion: Set Export_all_types = True in config to auto-detect all types")

            return available_classes

    def _obj_export(
        self,
        objs_dir: str,
        ifc_file,
        ifc_classes: List[str],
        index: int,
        settings
    ) -> int:
        """Export IFC elements to OBJ files.
        
        Processes each element, extracts geometry, handles special cases
        (windows/doors), and writes to OBJ files.
        
        Args:
            objs_dir: Output directory for OBJ files.
            ifc_file: Opened IFC file object.
            ifc_classes: List of IFC class names to process.
            index: File index for naming.
            settings: IFC geometry settings.
            
        Returns:
            Number of elements successfully exported.
        """
        output_path = objs_dir
        special_classes = {"IfcWindow", "IfcDoor"}

        elements_exported = 0

        for class_name in ifc_classes:
            elements = ifc_file.by_type(class_name)

            if not elements:
                continue

            print(f"    Processing {class_name}: {len(elements)} objects")

            for elem in elements:
                try:
                    class_name = elem.is_a()

                    if class_name in special_classes:
                        settings.set(settings.USE_WORLD_COORDS, False)
                        vertices, triangles = [], []
                        try:
                            shp = ifcopenshell.geom.create_shape(settings, elem)
                            verts_l = np.asarray(shp.geometry.verts, dtype=float).reshape(-1, 3)
                            T = np.asarray(shp.transformation.matrix, dtype=float).reshape(4, 4).T
                            lo, hi = verts_l.min(0), verts_l.max(0)
                            bbox_l = np.array([
                                [lo[0], lo[1], lo[2]],
                                [hi[0], lo[1], lo[2]],
                                [hi[0], hi[1], lo[2]],
                                [lo[0], hi[1], lo[2]],
                                [lo[0], lo[1], hi[2]],
                                [hi[0], lo[1], hi[2]],
                                [hi[0], hi[1], hi[2]],
                                [lo[0], hi[1], hi[2]],
                            ])
                            bbox_w = (T @ np.c_[bbox_l, np.ones(8)].T).T[:, :3]
                            vertices.extend(bbox_w)
                            faces_box = [
                                [0, 2, 1], [0, 3, 2],
                                [4, 5, 6], [4, 6, 7],
                                [0, 1, 5], [0, 5, 4],
                                [1, 2, 6], [1, 6, 5],
                                [2, 3, 7], [2, 7, 6],
                                [3, 0, 4], [3, 4, 7],
                            ]
                            if np.linalg.det(T[:3, :3]) < 0:
                                faces_box = [f[::-1] for f in faces_box]
                            triangles.extend(faces_box)
                        except Exception as e:
                            print(f"      ⚠️  [{elem.GlobalId}] AABB export failed: {e}")
                            vertices, triangles = [], []

                        if vertices and triangles:
                            split, segments = False, None
                            self._write_to_obj(
                                output_path, vertices, triangles,
                                elem.GlobalId, class_name, index,
                                split, segments
                            )
                            elements_exported += 1
                        continue

                    settings.set(settings.USE_WORLD_COORDS, True)
                    vertices, triangles = self._get_geometry(elem, settings)

                    if len(vertices) == 0 or len(triangles) == 0:
                        print(f"      ⚠️  [{elem.GlobalId}] No geometry data, skipping")
                        continue

                    face_graph = self._build_face_graph(triangles)
                    segments = self._find_connected_components(triangles, face_graph)

                    if len(segments) > 1:
                        print(
                            f"      ℹ️  Element {elem.GlobalId} has "
                            f"{len(segments)} separate parts; exporting each part as "
                            f"separate OBJ with suffix _1, _2, ..."
                        )
                        split = True
                    else:
                        segments, split = False, None

                    self._write_to_obj(
                        output_path, vertices, triangles,
                        elem.GlobalId, class_name, index,
                        split, segments
                    )
                    elements_exported += 1

                except Exception as e:
                    print(f"      ❌ Error processing element {elem.GlobalId}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

        return elements_exported

    def _get_geometry(
        self, 
        element, 
        settings
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract vertices and faces from IFC element.
        
        Args:
            element: IFC element to extract geometry from.
            settings: IFC geometry settings.
            
        Returns:
            Tuple of (vertices, faces) as numpy arrays. Returns empty arrays on error.
        """
        try:
            shape = ifcopenshell.geom.create_shape(settings, element)
            verts = np.array(shape.geometry.verts).reshape(-1, 3)
            faces = np.array(shape.geometry.faces).reshape(-1, 3)
            return verts, faces
        except Exception as e:
            print(f"      ⚠️  Geometry extraction failed: {e}")
            return np.array([]), np.array([])

    def _build_face_graph(self, faces: np.ndarray) -> Dict[int, Set[int]]:
        """Build adjacency graph of faces sharing edges.
        
        Args:
            faces: Array of face indices of shape (F, 3).
            
        Returns:
            Dictionary mapping face indices to sets of adjacent face indices.
        """
        edge_to_faces = defaultdict(set)
        face_graph = defaultdict(set)
        
        for i, face in enumerate(faces):
            edges = {
                (min(face[0], face[1]), max(face[0], face[1])),
                (min(face[1], face[2]), max(face[1], face[2])),
                (min(face[2], face[0]), max(face[2], face[0]))
            }
            for edge in edges:
                edge_to_faces[edge].add(i)
        
        for edge, face_set in edge_to_faces.items():
            face_list = list(face_set)
            for i in range(len(face_list)):
                for j in range(i + 1, len(face_list)):
                    face_graph[face_list[i]].add(face_list[j])
                    face_graph[face_list[j]].add(face_list[i])
        
        return face_graph

    def _find_connected_components(
        self, 
        faces: np.ndarray, 
        face_graph: Dict[int, Set[int]]
    ) -> List[Set[int]]:
        """Find connected components in face graph using BFS.
        
        Identifies separate mesh parts that are not connected by shared edges.
        
        Args:
            faces: Array of face indices.
            face_graph: Adjacency graph of faces.
            
        Returns:
            List of sets, each containing face indices of a connected component.
        """
        visited = set()
        segments = []

        def bfs(start_face):
            queue = deque([start_face])
            component = set()
            while queue:
                current = queue.popleft()
                if current in visited:
                    continue
                visited.add(current)
                component.add(current)
                queue.extend(face_graph[current] - visited)
            return component

        for i in range(len(faces)):
            if i not in visited:
                group = bfs(i)
                segments.append(group)
        return segments

    def _normalize_to_global_box(self, vertices: np.ndarray) -> np.ndarray:
        """Normalize vertices to fit within global bounding box.
        
        Transforms vertices to fit within configured global box while preserving
        aspect ratio. Performs translation to origin, uniform scaling, and
        translation to target center.
        
        Args:
            vertices: Vertex coordinates of shape (N, 3).
            
        Returns:
            Normalized vertex coordinates.
        """
        vertices = np.asarray(vertices, dtype=float)

        if len(vertices) == 0:
            return vertices

        min_coords = vertices.min(axis=0)
        max_coords = vertices.max(axis=0)
        current_size = max_coords - min_coords
        current_center = (min_coords + max_coords) / 2

        target_min = np.array([
            self.global_box[0][0], 
            self.global_box[1][0], 
            self.global_box[2][0]
        ])
        target_max = np.array([
            self.global_box[0][1], 
            self.global_box[1][1], 
            self.global_box[2][1]
        ])
        target_size = target_max - target_min
        target_center = (target_min + target_max) / 2

        scale_factors = target_size / (current_size + 1e-8)
        uniform_scale = np.min(scale_factors)

        vertices_centered = vertices - current_center
        vertices_scaled = vertices_centered * uniform_scale
        vertices_normalized = vertices_scaled + target_center

        return vertices_normalized

    def _write_to_obj(
        self,
        objs_dir: str,
        vertices: np.ndarray,
        faces: np.ndarray,
        uid: str,
        ifc_class: str,
        index: int,
        split: bool = False,
        groups: Optional[List[Set[int]]] = None
    ) -> None:
        """Write geometry to OBJ file(s).
        
        Applies scaling and normalization transformations before writing.
        If split is True, writes each connected component to a separate file.
        
        Args:
            objs_dir: Output directory.
            vertices: Vertex coordinates.
            faces: Face indices.
            uid: Unique identifier for the element.
            ifc_class: IFC class name.
            index: File index for naming.
            split: Whether to split into multiple files.
            groups: Optional list of face index sets for split export.
        """
        vertices = np.asarray(vertices, dtype=float)
        faces = np.asarray(faces, dtype=int)

        vertices = vertices * self.scale
        vertices = self._normalize_to_global_box(vertices)

        if split and groups:
            for split_num, group in enumerate(groups, start=1):
                used_vertex_indices = set()
                for face_index in group:
                    used_vertex_indices.update(faces[face_index])
                used_vertex_indices = sorted(used_vertex_indices)
                index_mapping = {
                    old_idx: new_idx
                    for new_idx, old_idx in enumerate(used_vertex_indices, start=1)
                }
                file_name = f"{self.date}_{index}_{ifc_class}_{uid}_{split_num}"
                output_file = os.path.join(objs_dir, f"{file_name}.obj")
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write("# Split OBJ File (segment export)\n")
                    for old_idx in used_vertex_indices:
                        vertex = vertices[old_idx]
                        vertex_str = "v " + " ".join(map(str, vertex)) + "\n"
                        f.write(vertex_str)
                    f.write(f"\no Object_{split_num}\n")
                    for face_index in group:
                        remapped = [index_mapping[v] for v in faces[face_index]]
                        f.write("f " + " ".join(str(idx) for idx in remapped) + "\n")
                print(f"      ✅ Part {split_num} saved to {output_file}")
        else:
            file_name = f"{self.date}_{index}_{ifc_class}_{uid}.obj"
            output_file = os.path.join(objs_dir, file_name)
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("# OBJ File\n")
                for vertex in vertices:
                    vertex_str = "v " + " ".join(map(str, vertex)) + "\n"
                    f.write(vertex_str)
                f.write("\no Object\n")
                for face in faces:
                    f.write("f " + " ".join(str(idx + 1) for idx in face) + "\n")
            print(f"      ✅ Saved to {output_file}")

    # ========== Public API ==========

    def export(self) -> None:
        """Execute the complete IFC to OBJ export pipeline.
        
        Copies IFC files, creates duplicates if configured, and converts
        all elements to OBJ format with appropriate transformations.
        """
        self._ifc_move()
        self._ifcs_to_obj()