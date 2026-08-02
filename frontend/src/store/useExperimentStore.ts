import { create } from 'zustand';
import { Experiment } from '../types';

interface ExperimentStoreState {
  experiments: Experiment[];
  selectedExperimentId: string | null;
  setSelectedExperimentId: (id: string | null) => void;
}

export const useExperimentStore = create<ExperimentStoreState>((set) => ({
  experiments: [
    {
      id: 'exp_001',
      name: 'Single-Agent Gym PPO Baseline',
      algorithm: 'PPO',
      status: 'completed',
      meanReward: -9.06,
      successRate: 1.0,
      collisions: 0,
      createdAt: '2026-08-02 14:00',
    },
    {
      id: 'exp_002',
      name: 'IPPO Fleet Benchmark (4 Robots)',
      algorithm: 'IPPO',
      status: 'completed',
      meanReward: -240.0,
      successRate: 1.0,
      collisions: 0,
      createdAt: '2026-08-02 18:00',
    },
    {
      id: 'exp_003',
      name: 'Spatial MAPPO CNN Critic Fleet Benchmark',
      algorithm: 'Spatial MAPPO',
      status: 'completed',
      meanReward: -40.0,
      successRate: 1.0,
      collisions: 0,
      createdAt: '2026-08-02 19:30',
    },
  ],
  selectedExperimentId: 'exp_003',
  setSelectedExperimentId: (id) => set({ selectedExperimentId: id }),
}));
