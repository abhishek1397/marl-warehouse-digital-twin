import React from 'react';
import { Cpu, Zap, Layers, GitBranch } from 'lucide-react';
import { Container } from '../components/Container';
import { PageHeader } from '../components/PageHeader';
import { SectionCard } from '../components/SectionCard';
import { Badge } from '../components/Badge';

export const AlgorithmsPage: React.FC = () => {
  return (
    <Container className="py-6 space-y-6">
      <PageHeader
        title="MARL Algorithm Suite"
        subtitle="Comparative theoretical and empirical breakdown of reinforcement learning algorithms."
      />

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* PPO */}
        <SectionCard title="1. Single-Agent PPO" subtitle="Proximal Policy Optimization Baseline">
          <div className="space-y-3 text-xs text-slate-300">
            <div className="flex items-center justify-between">
              <span className="text-slate-400">Paradigm:</span>
              <span className="font-mono text-white">Single-Agent Gym</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-slate-400">Action Space:</span>
              <span className="font-mono text-white">Discrete(5)</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-slate-400">CLIP Loss:</span>
              <span className="font-mono text-white">eps = 0.2</span>
            </div>
          </div>
        </SectionCard>

        {/* IPPO */}
        <SectionCard title="2. Independent PPO (IPPO)" subtitle="Fully Decentralized Multi-Agent PPO">
          <div className="space-y-3 text-xs text-slate-300">
            <div className="flex items-center justify-between">
              <span className="text-slate-400">Paradigm:</span>
              <span className="font-mono text-white">Decentralized Actors & Critics</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-slate-400">Parameter Sharing:</span>
              <Badge variant="success">Shared Policy</Badge>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-slate-400">Action Masking:</span>
              <Badge variant="success">DAM Active</Badge>
            </div>
          </div>
        </SectionCard>

        {/* MAPPO */}
        <SectionCard title="3. MAPPO (MLP Critic)" subtitle="Centralized Training Decentralized Execution">
          <div className="space-y-3 text-xs text-slate-300">
            <div className="flex items-center justify-between">
              <span className="text-slate-400">Paradigm:</span>
              <span className="font-mono text-white">CTDE</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-slate-400">Centralized Critic:</span>
              <span className="font-mono text-white">Flat MLP V(S)</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-slate-400">Fleet Scaling bottleneck:</span>
              <Badge variant="warning">State Dimension Explosion</Badge>
            </div>
          </div>
        </SectionCard>

        {/* Spatial MAPPO */}
        <SectionCard title="4. Spatial MAPPO (S-MAPPO)" subtitle="2D CNN Centralized Critic">
          <div className="space-y-3 text-xs text-slate-300">
            <div className="flex items-center justify-between">
              <span className="text-slate-400">Paradigm:</span>
              <span className="font-mono text-white">CTDE Spatial CNN</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-slate-400">Spatial Channels:</span>
              <span className="font-mono text-white">5 Channels (5 x H x W)</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-slate-400">Parameter Scaling:</span>
              <Badge variant="success">O(1) Constant Complexity</Badge>
            </div>
          </div>
        </SectionCard>
      </div>
    </Container>
  );
};
