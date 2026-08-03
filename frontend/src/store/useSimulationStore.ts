import { create } from 'zustand';
import { ApiClient, ApiSimulationState } from '../api/client';

export type AlgorithmType = 'A*' | 'PPO' | 'PPO + PBRS' | 'PPO + DAM' | 'IPPO' | 'MAPPO' | 'Spatial MAPPO';

export interface RobotEntity {
  id: string;
  x: number;
  y: number;
  battery: number;
  state: string;
  assignedPackageId?: string;
  targetPosition?: [number, number];
  currentAction?: string;
  isCollision?: boolean;
  plannedPath?: [number, number][];
  color: string;
}

export interface GridEntity {
  id: string;
  x: number;
  y: number;
  type: 'shelf' | 'charging_station' | 'package' | 'obstacle' | 'goal';
}

export interface LiveMetrics {
  episode: number;
  step: number;
  reward: number;
  throughput: number;
  collisions: number;
  idleRobots: number;
  batteryAvg: number;
  packagesDelivered: number;
  fps: number;
  policyEntropy: number;
}

interface SimulationStoreState {
  algorithm: AlgorithmType;
  isRunning: boolean;
  isPaused: boolean;
  showDebugOverlay: boolean;
  speed: number;
  gridSize: number;
  robotCount: number;
  packageCount: number;
  robots: RobotEntity[];
  gridEntities: GridEntity[];
  metrics: LiveMetrics;

  setAlgorithm: (algorithm: AlgorithmType) => Promise<void>;
  setRunning: (isRunning: boolean) => Promise<void>;
  setPaused: (isPaused: boolean) => Promise<void>;
  toggleDebugOverlay: () => void;
  setSpeed: (speed: number) => void;
  setGridSize: (size: number) => Promise<void>;
  setRobotCount: (count: number) => Promise<void>;
  setPackageCount: (count: number) => void;
  resetSimulation: () => Promise<void>;
  stepSimulation: () => Promise<void>;
  syncStateFromBackend: (apiState: ApiSimulationState) => void;
  initializeBackendSimulation: () => Promise<void>;
}

const colors = ['#3b82f6', '#22c55e', '#f97316', '#a855f7', '#00f0ff', '#ec4899'];

export const useSimulationStore = create<SimulationStoreState>((set, get) => ({
  algorithm: 'Spatial MAPPO',
  isRunning: false,
  isPaused: false,
  showDebugOverlay: true,
  speed: 1,
  gridSize: 8,
  robotCount: 2,
  packageCount: 10,
  robots: [],
  gridEntities: [],
  metrics: {
    episode: 1,
    step: 0,
    reward: 0.0,
    throughput: 0.0,
    collisions: 0,
    idleRobots: 0,
    batteryAvg: 100.0,
    packagesDelivered: 0,
    fps: 60,
    policyEntropy: 1.0,
  },

  toggleDebugOverlay: () => set((state) => ({ showDebugOverlay: !state.showDebugOverlay })),

  syncStateFromBackend: (apiState) => {
    if (!apiState || !apiState.is_initialized) return;

    const mappedRobots: RobotEntity[] = apiState.robots.map((r, i) => {
      return {
        id: r.id,
        x: r.position[0],
        y: r.position[1],
        battery: Number(r.battery_level.toFixed(1)),
        state: r.state,
        assignedPackageId: r.assigned_task,
        targetPosition: (r as any).target_position ? (r as any).target_position : undefined,
        currentAction: (r as any).current_action ? (r as any).current_action : undefined,
        isCollision: Boolean((r as any).is_collision),
        plannedPath: (r as any).planned_path ? (r as any).planned_path : undefined,
        color: colors[i % colors.length],
      };
    });

    const mappedEntities: GridEntity[] = apiState.entities.map((e) => ({
      id: e.id,
      x: e.position[0],
      y: e.position[1],
      type: e.type,
    }));

    set({
      isRunning: apiState.is_running,
      isPaused: apiState.is_paused,
      gridSize: apiState.grid_size[0],
      robots: mappedRobots,
      gridEntities: mappedEntities,
      metrics: {
        episode: apiState.metrics.episode,
        step: apiState.metrics.step,
        reward: Number(apiState.metrics.reward.toFixed(2)),
        throughput: Number(apiState.metrics.throughput.toFixed(2)),
        collisions: apiState.metrics.collisions,
        idleRobots: apiState.metrics.idle_robots,
        batteryAvg: Number(apiState.metrics.battery_avg.toFixed(1)),
        packagesDelivered: apiState.metrics.packages_delivered,
        fps: apiState.metrics.fps || 60,
        policyEntropy: apiState.metrics.policy_entropy || 1.0,
      },
    });
  },

  initializeBackendSimulation: async () => {
    const { gridSize, robotCount } = get();
    try {
      const state = await ApiClient.createSimulation(gridSize, gridSize, robotCount);
      get().syncStateFromBackend(state);
    } catch (e) {
      console.warn('Backend server not reachable during initialization, using default offline setup.');
    }
  },

  setAlgorithm: async (algorithm) => {
    set({ algorithm });
    try {
      await ApiClient.selectAlgorithm(algorithm);
    } catch (e) {
      /* ignore offline */
    }
  },

  setRunning: async (isRunning) => {
    set({ isRunning, isPaused: false });
    try {
      const state = isRunning ? await ApiClient.startSimulation() : await ApiClient.pauseSimulation();
      get().syncStateFromBackend(state);
    } catch (e) {
      /* ignore offline */
    }
  },

  setPaused: async (isPaused) => {
    set({ isPaused });
    try {
      const state = isPaused ? await ApiClient.pauseSimulation() : await ApiClient.startSimulation();
      get().syncStateFromBackend(state);
    } catch (e) {
      /* ignore offline */
    }
  },

  setSpeed: (speed) => set({ speed }),

  setGridSize: async (gridSize) => {
    set({ gridSize });
    const { robotCount } = get();
    try {
      const state = await ApiClient.createSimulation(gridSize, gridSize, robotCount);
      get().syncStateFromBackend(state);
    } catch (e) {
      /* ignore offline */
    }
  },

  setRobotCount: async (robotCount) => {
    set({ robotCount });
    const { gridSize } = get();
    try {
      const state = await ApiClient.createSimulation(gridSize, gridSize, robotCount);
      get().syncStateFromBackend(state);
    } catch (e) {
      /* ignore offline */
    }
  },

  setPackageCount: (packageCount) => set({ packageCount }),

  resetSimulation: async () => {
    try {
      const state = await ApiClient.resetSimulation();
      get().syncStateFromBackend(state);
    } catch (e) {
      set({ isRunning: false, isPaused: false });
    }
  },

  stepSimulation: async () => {
    try {
      const state = await ApiClient.stepSimulation(1);
      get().syncStateFromBackend(state);
    } catch (e) {
      /* offline fallback step */
      const { robots, gridSize, metrics } = get();
      const updatedRobots = robots.map((robot) => {
        const dx = Math.floor(Math.random() * 3) - 1;
        const dy = Math.floor(Math.random() * 3) - 1;
        const newX = Math.max(0, Math.min(gridSize - 1, robot.x + dx));
        const newY = Math.max(0, Math.min(gridSize - 1, robot.y + dy));
        return { ...robot, x: newX, y: newY };
      });
      set({
        robots: updatedRobots,
        metrics: { ...metrics, step: metrics.step + 1 },
      });
    }
  },
}));
