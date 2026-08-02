"""Time-expanded ReservationTable for space-time multi-robot path planning."""

from typing import Dict, List, Optional, Set, Tuple

from simulator.position import Position


class ReservationTable:
    """Manages time-expanded spatial vertex and edge reservations for multi-robot path planning."""

    def __init__(self) -> None:
        # Maps (position, timestep) -> robot_id
        self._vertex_reservations: Dict[Tuple[Position, int], str] = {}
        # Maps (from_pos, to_pos, timestep) -> robot_id
        self._edge_reservations: Dict[Tuple[Position, Position, int], str] = {}

    def is_vertex_reserved(
        self, position: Position, timestep: int, ignore_robot_id: Optional[str] = None
    ) -> bool:
        """Checks if a position is reserved at a specific timestep by another robot."""
        res_robot = self._vertex_reservations.get((position, timestep))
        if res_robot is None:
            return False
        if ignore_robot_id is not None and res_robot == ignore_robot_id:
            return False
        return True

    def is_edge_reserved(
        self,
        from_pos: Position,
        to_pos: Position,
        timestep: int,
        ignore_robot_id: Optional[str] = None,
    ) -> bool:
        """Checks if moving from_pos -> to_pos at timestep conflicts with another robot's edge reservation or swap."""
        # Direct edge reservation check
        res_direct = self._edge_reservations.get((from_pos, to_pos, timestep))
        if res_direct is not None and res_direct != ignore_robot_id:
            return True

        # Reverse edge swap check (prevent head-on position swaps)
        res_swap = self._edge_reservations.get((to_pos, from_pos, timestep))
        if res_swap is not None and res_swap != ignore_robot_id:
            return True

        return False

    def reserve_vertex(self, robot_id: str, position: Position, timestep: int) -> bool:
        """Reserves a vertex (position, timestep) for a robot."""
        if self.is_vertex_reserved(position, timestep, ignore_robot_id=robot_id):
            return False
        self._vertex_reservations[(position, timestep)] = robot_id
        return True

    def reserve_edge(
        self, robot_id: str, from_pos: Position, to_pos: Position, timestep: int
    ) -> bool:
        """Reserves an edge (from_pos -> to_pos, timestep) for a robot."""
        if self.is_edge_reserved(from_pos, to_pos, timestep, ignore_robot_id=robot_id):
            return False
        self._edge_reservations[(from_pos, to_pos, timestep)] = robot_id
        return True

    def reserve_path(
        self, robot_id: str, path: List[Position], start_timestep: int = 0
    ) -> bool:
        """Reserves entire path trajectory across timesteps for a robot.

        If any vertex or edge reservation fails, rolls back all reservations for this path.
        """
        reserved_vertices: List[Tuple[Position, int]] = []
        reserved_edges: List[Tuple[Position, Position, int]] = []

        for i, pos in enumerate(path):
            t = start_timestep + i
            # Check & reserve vertex
            if self.is_vertex_reserved(pos, t, ignore_robot_id=robot_id):
                self._rollback(robot_id, reserved_vertices, reserved_edges)
                return False

            # Check & reserve edge if i > 0
            if i > 0:
                prev_pos = path[i - 1]
                if self.is_edge_reserved(prev_pos, pos, t, ignore_robot_id=robot_id):
                    self._rollback(robot_id, reserved_vertices, reserved_edges)
                    return False

            self._vertex_reservations[(pos, t)] = robot_id
            reserved_vertices.append((pos, t))

            if i > 0:
                prev_pos = path[i - 1]
                self._edge_reservations[(prev_pos, pos, t)] = robot_id
                reserved_edges.append((prev_pos, pos, t))

        return True

    def release_path(
        self, robot_id: str, path: List[Position], start_timestep: int = 0
    ) -> None:
        """Releases all vertex and edge reservations for a robot along a path."""
        for i, pos in enumerate(path):
            t = start_timestep + i
            if self._vertex_reservations.get((pos, t)) == robot_id:
                del self._vertex_reservations[(pos, t)]

            if i > 0:
                prev_pos = path[i - 1]
                if self._edge_reservations.get((prev_pos, pos, t)) == robot_id:
                    del self._edge_reservations[(prev_pos, pos, t)]

    def _rollback(
        self,
        robot_id: str,
        vertices: List[Tuple[Position, int]],
        edges: List[Tuple[Position, Position, int]],
    ) -> None:
        for v in vertices:
            if self._vertex_reservations.get(v) == robot_id:
                del self._vertex_reservations[v]
        for e in edges:
            if self._edge_reservations.get(e) == robot_id:
                del self._edge_reservations[e]

    def clear(self) -> None:
        """Clears all reservations."""
        self._vertex_reservations.clear()
        self._edge_reservations.clear()

    def __repr__(self) -> str:
        return (
            f"ReservationTable(vertex_count={len(self._vertex_reservations)}, "
            f"edge_count={len(self._edge_reservations)})"
        )
