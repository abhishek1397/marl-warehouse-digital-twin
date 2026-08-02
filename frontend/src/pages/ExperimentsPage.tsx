import React from 'react';
import { Activity, Plus, Search } from 'lucide-react';
import { Container } from '../components/Container';
import { PageHeader } from '../components/PageHeader';
import { SectionCard } from '../components/SectionCard';
import { Button } from '../components/Button';
import { Badge } from '../components/Badge';
import { useExperimentStore } from '../store/useExperimentStore';

export const ExperimentsPage: React.FC = () => {
  const { experiments } = useExperimentStore();

  return (
    <Container className="py-6 space-y-6">
      <PageHeader
        title="Experiment Manager"
        subtitle="Track, filter, and compare training runs and multi-seed statistical evaluations."
        action={
          <Button icon={<Plus className="w-4 h-4" />}>
            New Experiment
          </Button>
        }
      />

      <SectionCard title="Registered Experiments Table" subtitle="Showing all past training benchmarks">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs border-collapse">
            <thead>
              <tr className="border-b border-surface-border text-slate-400 uppercase font-mono tracking-wider">
                <th className="py-3 px-4">ID</th>
                <th className="py-3 px-4">Experiment Name</th>
                <th className="py-3 px-4">Algorithm</th>
                <th className="py-3 px-4">Status</th>
                <th className="py-3 px-4 font-mono">Mean Reward</th>
                <th className="py-3 px-4 font-mono">Success Rate</th>
                <th className="py-3 px-4 font-mono">Collisions</th>
                <th className="py-3 px-4">Created At</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-surface-border/50 text-slate-300">
              {experiments.map((exp) => (
                <tr key={exp.id} className="hover:bg-slate-800/40 transition-colors">
                  <td className="py-3 px-4 font-mono text-accent">{exp.id}</td>
                  <td className="py-3 px-4 font-medium text-white">{exp.name}</td>
                  <td className="py-3 px-4">
                    <Badge variant="info">{exp.algorithm}</Badge>
                  </td>
                  <td className="py-3 px-4">
                    <Badge variant={exp.status === 'completed' ? 'success' : 'warning'}>
                      {exp.status}
                    </Badge>
                  </td>
                  <td className="py-3 px-4 font-mono">{exp.meanReward.toFixed(2)}</td>
                  <td className="py-3 px-4 font-mono">{(exp.successRate * 100).toFixed(0)}%</td>
                  <td className="py-3 px-4 font-mono text-emerald-400">{exp.collisions}</td>
                  <td className="py-3 px-4 text-slate-400">{exp.createdAt}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </SectionCard>
    </Container>
  );
};
