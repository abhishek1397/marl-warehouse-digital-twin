export type AlgorithmType = 'PPO' | 'IPPO' | 'MAPPO' | 'Spatial MAPPO';

export interface SimulationState {
  isRunning: boolean;
  isPaused: boolean;
  stepCount: number;
  activeRobots: number;
  gridSize: [number, number];
  fps: number;
}

export interface Experiment {
  id: string;
  name: string;
  algorithm: AlgorithmType;
  status: 'running' | 'completed' | 'failed' | 'queued';
  meanReward: number;
  successRate: number;
  collisions: number;
  createdAt: string;
}

export interface StatItem {
  label: string;
  value: string | number;
  change?: string;
  trend?: 'up' | 'down' | 'neutral';
  icon?: string;
}
