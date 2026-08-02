"""CollisionDetector class providing diagnostic validation for robot trajectories."""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Tuple

from simulator.position import Position
from simulator.warehouse import Warehouse


class CollisionType(Enum):
    """Enumeration of multi-robot collision types."""

    NONE = auto()
    VERTEX_COLLISION = auto()
    EDGE_SWAP_COLLISION = auto()
    STATIC_OBSTACLE_COLLISION = auto()
    OUT_OF_BOUNDS = auto()


@dataclass
class CollisionDiagnostic:
    """Dataclass holding detailed diagnostic output for a detected collision event."""

    collision_type: CollisionType
    timestep: int
    position: Position
    robot_ids: List[str] = field(default_factory=list)
    message: str = ""


class CollisionDetector:
    """Detects vertex collisions, edge-swap collisions, static obstacle collisions, and boundary violations."""

    def detect_collisions(
        self,
        warehouse: Warehouse,
        robot_paths: Dict[str, List[Position]],
        start_timestep: int = 0,
    ) -> List[CollisionDiagnostic]:
        """Validates all robot paths and returns detailed collision diagnostics.

        Args:
            warehouse: Warehouse grid layout reference.
            robot_paths: Map of robot_id -> list of Positions representing path over timesteps.
            start_timestep: Initial timestep offset.

        Returns:
            List of CollisionDiagnostic records.
        """
        diagnostics: List[CollisionDiagnostic] = []
        if not robot_paths:
            return diagnostics

        max_path_len = max(len(p) for p in robot_paths.values())

        # 1. Validate static obstacles and boundaries for each robot
        for robot_id, path in robot_paths.items():
            for idx, pos in enumerate(path):
                t = start_timestep + idx
                if not warehouse.is_in_bounds(pos):
                    diagnostics.append(
                        CollisionDiagnostic(
                            collision_type=CollisionType.OUT_OF_BOUNDS,
                            timestep=t,
                            position=pos,
                            robot_ids=[robot_id],
                            message=f"Robot '{robot_id}' path at step {t} position {pos} is out of bounds.",
                        )
                    )
                else:
                    cell = warehouse.get_cell(pos)
                    if not cell.cell_type.is_traversable:
                        diagnostics.append(
                            CollisionDiagnostic(
                                collision_type=CollisionType.STATIC_OBSTACLE_COLLISION,
                                timestep=t,
                                position=pos,
                                robot_ids=[robot_id],
                                message=f"Robot '{robot_id}' path at step {t} intersects non-traversable cell {cell.cell_type.name} at {pos}.",
                            )
                        )

        # 2. Check multi-robot vertex and edge-swap collisions timestep by timestep
        robot_ids = list(robot_paths.keys())

        for idx in range(max_path_len):
            t = start_timestep + idx

            # Vertex positions at timestep t
            vertex_occupancy: Dict[Position, List[str]] = {}
            for r_id in robot_ids:
                path = robot_paths[r_id]
                # If robot finished path, it remains at its final position
                pos = path[min(idx, len(path) - 1)]
                vertex_occupancy.setdefault(pos, []).append(r_id)

            # Check vertex collisions
            for pos, occupying_robots in vertex_occupancy.items():
                if len(occupying_robots) > 1:
                    diagnostics.append(
                        CollisionDiagnostic(
                            collision_type=CollisionType.VERTEX_COLLISION,
                            timestep=t,
                            position=pos,
                            robot_ids=occupying_robots,
                            message=f"Vertex collision at step {t} at position {pos} between robots {occupying_robots}.",
                        )
                    )

            # Check edge-swap collisions (for step idx > 0)
            if idx > 0:
                for i in range(len(robot_ids)):
                    for j in range(i + 1, len(robot_ids)):
                        r1, r2 = robot_ids[i], robot_ids[j]
                        path1, path2 = robot_paths[r1], robot_paths[r2]

                        pos1_prev = path1[min(idx - 1, len(path1) - 1)]
                        pos1_curr = path1[min(idx, len(path1) - 1)]

                        pos2_prev = path2[min(idx - 1, len(path2) - 1)]
                        pos2_curr = path2[min(idx, len(path2) - 1)]

                        # Swap condition: r1 moves posA -> posB while r2 moves posB -> posA
                        if pos1_prev == pos2_curr and pos1_curr == pos2_prev and pos1_prev != pos1_curr:
                            diagnostics.append(
                                CollisionDiagnostic(
                                    collision_type=CollisionType.EDGE_SWAP_COLLISION,
                                    timestep=t,
                                    position=pos1_curr,
                                    robot_ids=[r1, r2],
                                    message=f"Edge swap collision at step {t} between robots '{r1}' and '{r2}' on edge {pos1_prev} <-> {pos1_curr}.",
                                )
                            )

        return diagnostics

    def has_collisions(
        self,
        warehouse: Warehouse,
        robot_paths: Dict[str, List[Position]],
        start_timestep: int = 0,
    ) -> bool:
        """Returns True if any collision exists across the given robot paths."""
        return len(self.detect_collisions(warehouse, robot_paths, start_timestep)) > 0
