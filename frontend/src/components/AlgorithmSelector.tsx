import React from 'react';
import { Cpu } from 'lucide-react';
import { useSimulationStore, AlgorithmType } from '../store/useSimulationStore';

const algorithms: AlgorithmType[] = [
  'A*',
  'PPO',
  'PPO + PBRS',
  'PPO + DAM',
  'IPPO',
  'MAPPO',
  'Spatial MAPPO',
];

export const AlgorithmSelector: React.FC = () => {
  const { algorithm, setAlgorithm } = useSimulationStore();

  return (
    <div className="space-y-2">
      <label className="text-xs font-semibold text-slate-300 uppercase tracking-wider font-mono flex items-center">
        <Cpu className="w-3.5 h-3.5 mr-1.5 text-accent" />
        Algorithm Variant
      </label>
      <div className="grid grid-cols-1 gap-1.5">
        {algorithms.map((algo) => (
          <button
            key={algo}
            onClick={() => setAlgorithm(algo)}
            className={`w-full text-left px-3 py-2 rounded-lg text-xs font-medium transition-all ${
              algorithm === algo
                ? 'bg-accent/20 text-white border border-accent shadow-glow font-bold'
                : 'bg-surface-dark text-slate-400 border border-surface-border hover:bg-slate-800 hover:text-slate-200'
            }`}
          >
            {algo}
          </button>
        ))}
      </div>
    </div>
  );
};
