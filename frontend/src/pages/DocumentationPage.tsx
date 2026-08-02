import React from 'react';
import { BookOpen, Code, Terminal } from 'lucide-react';
import { Container } from '../components/Container';
import { PageHeader } from '../components/PageHeader';
import { SectionCard } from '../components/SectionCard';

export const DocumentationPage: React.FC = () => {
  return (
    <Container className="py-6 space-y-6">
      <PageHeader
        title="Platform Documentation"
        subtitle="Developer guide, package architecture, and API documentation."
      />

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <SectionCard title="1. Warehouse Simulator" subtitle="simulator/">
          <p className="text-xs text-slate-300">
            Digital twin simulator written in pure Python/NumPy containing Grid, Cell, Robot, Shelf, ChargingStation, AStarPlanner, and ReservationTable.
          </p>
        </SectionCard>

        <SectionCard title="2. MARL Algorithm Package" subtitle="marl/">
          <p className="text-xs text-slate-300">
            Multi-agent reinforcement learning library containing PPO, IPPO, MAPPO, Spatial MAPPO, RolloutBuffer, and PettingZoo Parallel API.
          </p>
        </SectionCard>

        <SectionCard title="3. Research Diagnostics" subtitle="research/">
          <p className="text-xs text-slate-300">
            Scientific verification framework containing multi-seed statistical runners, GAE verifier, CTDE validator, and scalability profiler.
          </p>
        </SectionCard>
      </div>
    </Container>
  );
};
