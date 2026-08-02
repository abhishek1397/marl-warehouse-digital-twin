"""SimulationEngine class controlling the main discrete-event execution loop."""

from enum import Enum, auto
from typing import Any, Dict, List, Optional

from simulator.astar import AStarPlanner
from simulator.battery_manager import BatteryManager
from simulator.charging_station import ChargingStation
from simulator.collision_detector import CollisionDetector
from simulator.exceptions import WarehouseError
from simulator.metrics import MetricsCollector
from simulator.package import Package, PackageStatus
from simulator.planner import MultiRobotPlanner, PlanningRequest
from simulator.position import Position
from simulator.reservation_table import ReservationTable
from simulator.robot import Robot, RobotState
from simulator.scheduler import TaskScheduler
from simulator.shelf import Shelf
from simulator.task import Task, TaskStatus, TaskType
from simulator.task_manager import TaskManager
from simulator.traffic_controller import TrafficController
from simulator.warehouse import Warehouse


class SimulationState(Enum):
    """Enumeration of simulation execution lifecycle states."""

    UNINITIALIZED = auto()
    READY = auto()
    RUNNING = auto()
    PAUSED = auto()
    COMPLETED = auto()


class SimulationEngine:
    """Core simulation engine driving discrete time-step execution for the warehouse digital twin."""

    def __init__(
        self,
        warehouse: Warehouse,
        task_manager: Optional[TaskManager] = None,
        battery_manager: Optional[BatteryManager] = None,
        metrics_collector: Optional[MetricsCollector] = None,
        planner: Optional[MultiRobotPlanner] = None,
        traffic_controller: Optional[TrafficController] = None,
        task_scheduler: Optional[TaskScheduler] = None,
    ) -> None:
        self._warehouse: Warehouse = warehouse
        self._task_manager: TaskManager = task_manager or TaskManager()
        self._battery_manager: BatteryManager = battery_manager or BatteryManager()
        self._metrics_collector: MetricsCollector = metrics_collector or MetricsCollector()

        self._planner: MultiRobotPlanner = planner or MultiRobotPlanner()
        self._traffic_controller: TrafficController = traffic_controller or TrafficController()
        self._task_scheduler: TaskScheduler = task_scheduler or TaskScheduler()
        self._collision_detector: CollisionDetector = CollisionDetector()

        self._fleet: Dict[str, Robot] = {}
        self._charging_stations: Dict[str, ChargingStation] = {}
        self._shelves: Dict[str, Shelf] = {}
        self._packages: Dict[str, Package] = {}

        # Cached active path routes: robot_id -> List[Position]
        self._active_routes: Dict[str, List[Position]] = {}

        self._current_step: int = 0
        self._state: SimulationState = SimulationState.UNINITIALIZED

    @property
    def warehouse(self) -> Warehouse:
        """Returns warehouse reference."""
        return self._warehouse

    @property
    def task_manager(self) -> TaskManager:
        """Returns task manager reference."""
        return self._task_manager

    @property
    def battery_manager(self) -> BatteryManager:
        """Returns battery manager reference."""
        return self._battery_manager

    @property
    def metrics_collector(self) -> MetricsCollector:
        """Returns metrics collector reference."""
        return self._metrics_collector

    @property
    def planner(self) -> MultiRobotPlanner:
        """Returns classical multi-robot planner reference."""
        return self._planner

    @property
    def traffic_controller(self) -> TrafficController:
        """Returns traffic controller reference."""
        return self._traffic_controller

    @property
    def task_scheduler(self) -> TaskScheduler:
        """Returns active task scheduler reference."""
        return self._task_scheduler

    @property
    def fleet(self) -> Dict[str, Robot]:
        """Returns map of registered robots."""
        return self._fleet

    @property
    def current_step(self) -> int:
        """Returns current simulation step count."""
        return self._current_step

    @property
    def state(self) -> SimulationState:
        """Returns current simulation execution state."""
        return self._state

    def add_robot(self, robot: Robot) -> None:
        """Registers a robot into the simulation environment."""
        if robot.robot_id in self._fleet:
            raise WarehouseError(f"Robot ID '{robot.robot_id}' already registered.")
        self._fleet[robot.robot_id] = robot
        self._warehouse.place_object(robot.position, robot)

    def add_charging_station(self, station: ChargingStation) -> None:
        """Registers a charging station into the simulation environment."""
        if station.station_id in self._charging_stations:
            raise WarehouseError(f"Charging station ID '{station.station_id}' already registered.")
        self._charging_stations[station.station_id] = station

    def add_shelf(self, shelf: Shelf) -> None:
        """Registers a shelf into the simulation environment."""
        if shelf.shelf_id in self._shelves:
            raise WarehouseError(f"Shelf ID '{shelf.shelf_id}' already registered.")
        self._shelves[shelf.shelf_id] = shelf

    def add_package(self, package: Package) -> None:
        """Registers a package into the simulation environment."""
        if package.package_id in self._packages:
            raise WarehouseError(f"Package ID '{package.package_id}' already registered.")
        self._packages[package.package_id] = package

    def initialize(self) -> None:
        """Validates environment state and readies the engine for execution."""
        self._current_step = 0
        self._state = SimulationState.READY

    def step(self) -> Dict[str, Any]:
        """Executes a single discrete simulation step."""
        if self._state not in {SimulationState.READY, SimulationState.RUNNING}:
            raise WarehouseError(f"Cannot step simulation in state {self._state.name}.")

        # 1. Process emergency battery recharges and task allocations
        idle_robots = [r for r in self._fleet.values() if r.is_idle()]
        unassigned = self._task_manager.get_pending_tasks()

        for robot in list(idle_robots):
            if self._battery_manager.is_low_battery(robot) and self._charging_stations:
                station = next(iter(self._charging_stations.values()))
                recharge_task = self._task_manager.create_task(
                    task_id=f"recharge_{robot.robot_id}_{self._current_step}",
                    task_type=TaskType.RECHARGE_BATTERY,
                    pickup_position=station.position,
                    drop_position=station.position,
                    created_at_step=self._current_step,
                )
                robot.assign_task(recharge_task)
                idle_robots.remove(robot)

        # Schedule remaining tasks with active TaskScheduler Strategy
        scheduled_pairs = self._task_scheduler.schedule(unassigned, idle_robots)
        for task, robot in scheduled_pairs:
            robot.assign_task(task)
            if task in unassigned:
                self._task_manager._unassigned_queue.remove(task)

        # 2. Plan or update paths for robots needing routes
        for robot in self._fleet.values():
            if robot.assigned_task and robot.robot_id not in self._active_routes:
                self._plan_route_for_robot(robot)

        # 3. Update robot movement and actions
        for robot in self._fleet.values():
            self._update_robot(robot)

        # 4. Check for deadlocks and congestion via TrafficController
        stalled_ids = self._traffic_controller.detect_deadlocks(self._fleet)
        for stalled_id in stalled_ids:
            if stalled_id in self._fleet:
                stalled_robot = self._fleet[stalled_id]
                if stalled_robot.assigned_task:
                    self._plan_route_for_robot(stalled_robot, is_replan=True)
                    self._traffic_controller.record_deadlock_prevented()

        # 5. Update battery levels
        self._battery_manager.update_fleet(self._fleet)

        # 6. Record metrics
        completed_deliveries = len(self._task_manager.get_completed_tasks())
        snapshot = self._metrics_collector.record_step(
            self._current_step, self._fleet, completed_deliveries
        )

        # 7. Check completion
        if self._task_manager.all_tasks_completed() and self._task_manager.total_tasks > 0:
            self._state = SimulationState.COMPLETED

        self._current_step += 1
        return snapshot

    def _plan_route_for_robot(self, robot: Robot, is_replan: bool = False) -> None:
        """Plans optimal space-time path for a robot using A* planner."""
        task = robot.assigned_task
        if task is None:
            return

        target_pos = (
            task.pickup_position
            if robot.state == RobotState.MOVING_TO_PICKUP
            or robot.state == RobotState.MOVING_TO_CHARGE
            else task.drop_position
        )

        path_res = self._planner.astar_planner.plan(
            grid=self._warehouse.grid,
            start=robot.position,
            goal=target_pos,
            start_timestep=self._current_step,
            reservation_table=self._planner.reservation_table,
            robot_id=robot.robot_id,
        )

        self._metrics_collector.record_planning_event(
            planning_time_ms=path_res.planning_time_ms,
            is_replan=is_replan,
        )

        if path_res.success and len(path_res.path) > 1:
            self._active_routes[robot.robot_id] = path_res.path[1:]
            self._planner.reservation_table.reserve_path(
                robot_id=robot.robot_id,
                path=path_res.path,
                start_timestep=self._current_step,
            )
        else:
            self._active_routes[robot.robot_id] = []

    def _update_robot(self, robot: Robot) -> None:
        """Executes position step along planned route or step_towards fallback."""
        task = robot.assigned_task
        if task is None:
            return

        route = self._active_routes.get(robot.robot_id, [])

        if robot.state == RobotState.MOVING_TO_PICKUP:
            if robot.position == task.pickup_position:
                if task.package is not None:
                    robot.pick_up_package(task.package)
                else:
                    robot.state = RobotState.MOVING_TO_DROP
                task.status = TaskStatus.IN_PROGRESS
                self._active_routes.pop(robot.robot_id, None)
                self._plan_route_for_robot(robot)
            else:
                self._step_robot_forward(robot, route, task.pickup_position)

        elif robot.state == RobotState.MOVING_TO_DROP:
            if robot.position == task.drop_position:
                if robot.carrying_package is not None:
                    robot.drop_package()
                self._task_manager.complete_task(task.task_id, self._current_step)
                robot.clear_task()
                robot.increment_tasks_completed()
                self._active_routes.pop(robot.robot_id, None)
            else:
                self._step_robot_forward(robot, route, task.drop_position)

        elif robot.state == RobotState.MOVING_TO_CHARGE:
            if robot.position == task.pickup_position:
                station = self._find_station_at(robot.position)
                if station and station.is_available():
                    station.dock_robot(robot.robot_id)
                    robot.state = RobotState.CHARGING
                    self._active_routes.pop(robot.robot_id, None)
            else:
                self._step_robot_forward(robot, route, task.pickup_position)

        elif robot.state == RobotState.CHARGING:
            if robot.battery_percentage >= 99.9:
                station = self._find_station_with_robot(robot.robot_id)
                if station:
                    station.undock_robot(robot.robot_id)
                if robot.assigned_task:
                    self._task_manager.complete_task(robot.assigned_task.task_id, self._current_step)
                robot.clear_task()

    def _step_robot_forward(
        self, robot: Robot, route: List[Position], target: Position
    ) -> None:
        if route:
            next_pos = route.pop(0)
            robot.position = next_pos
            robot._total_distance_travelled += 1
        else:
            robot.step_towards(target)

    def _find_station_at(self, pos: Position) -> Optional[ChargingStation]:
        for station in self._charging_stations.values():
            if station.position == pos:
                return station
        return None

    def _find_station_with_robot(self, robot_id: str) -> Optional[ChargingStation]:
        for station in self._charging_stations.values():
            if station.is_docked(robot_id):
                return station
        return None

    def run(self, max_steps: int = 1000) -> Dict[str, Any]:
        """Runs the simulation until all tasks complete or max_steps is reached."""
        if self._state == SimulationState.UNINITIALIZED:
            self.initialize()

        self._state = SimulationState.RUNNING

        while self._state == SimulationState.RUNNING and self._current_step < max_steps:
            self.step()

        completed_count = len(self._task_manager.get_completed_tasks())
        return self._metrics_collector.compute_summary(
            self._fleet, completed_count, self._current_step
        )

    def pause(self) -> None:
        """Pauses simulation execution."""
        if self._state == SimulationState.RUNNING:
            self._state = SimulationState.PAUSED

    def resume(self) -> None:
        """Resumes paused simulation execution."""
        if self._state == SimulationState.PAUSED:
            self._state = SimulationState.RUNNING

    def reset(self) -> None:
        """Resets simulation engine to initial clean state."""
        self._current_step = 0
        self._fleet.clear()
        self._charging_stations.clear()
        self._shelves.clear()
        self._packages.clear()
        self._active_routes.clear()
        self._task_manager = TaskManager()
        self._metrics_collector = MetricsCollector()
        self._planner = MultiRobotPlanner()
        self._traffic_controller = TrafficController()
        self._task_scheduler = TaskScheduler()
        self._state = SimulationState.UNINITIALIZED
