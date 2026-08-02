import React, { useEffect } from 'react';
import { motion } from 'framer-motion';
import { Bot, LineChart, BarChart2, Shield, Activity, Battery, Zap } from 'lucide-react';
import { Container } from '../components/Container';
import { PageHeader } from '../components/PageHeader';
import { SectionCard } from '../components/SectionCard';
import { Badge } from '../components/Badge';
import { SimulationControls } from '../components/SimulationControls';
import { WarehouseGrid } from '../components/WarehouseGrid';
import { MetricsPanel } from '../components/MetricsPanel';
import { useSimulationStore } from '../store/useSimulationStore';

export const SimulationPage: React.FC = () => {
  const { initializeBackendSimulation } = useSimulationStore();

  useEffect(() => {
    initializeBackendSimulation();
  }, [initializeBackendSimulation]);
  return (
    <Container className="py-6 space-y-8 max-w-[1600px]">
      <PageHeader
        title="Industrial Robotics Control Center"
        subtitle="Real-time multi-agent warehouse fleet visualization, telemetry diagnostics, and simulation control."
        badge={<Badge variant="success" className="px-3 py-1 text-xs">Live Digital Twin</Badge>}
      />

      {/* THREE-COLUMN INDUSTRIAL CONTROL LAYOUT */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* LEFT COLUMN: SIMULATION CONTROLS (3 COLS) */}
        <div className="lg:col-span-3 space-y-6">
          <SimulationControls />
        </div>

        {/* CENTER COLUMN: WAREHOUSE GRID VISUALIZATION (6 COLS) */}
        <div className="lg:col-span-6 space-y-6">
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <h2 className="text-base font-bold text-white uppercase tracking-wider font-mono flex items-center">
                <Bot className="w-4 h-4 mr-2 text-accent" />
                2D Spatial Grid Renderer
              </h2>
              <Badge variant="info">Interactive Canvas</Badge>
            </div>
            <WarehouseGrid />
          </div>
        </div>

        {/* RIGHT COLUMN: LIVE METRICS DASHBOARD (3 COLS) */}
        <div className="lg:col-span-3 space-y-6">
          <MetricsPanel />
        </div>
      </div>

      {/* BOTTOM PANEL: PLACEHOLDER CHART CONTAINERS (6 CARDS) */}
      <div className="space-y-4 pt-4 border-t border-surface-border">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-xl font-extrabold text-white">Diagnostic Metric Curves</h2>
            <p className="text-xs text-slate-400">Real-time telemetry and MARL optimization curves</p>
          </div>
          <Badge variant="neutral">Placeholder Analytics</Badge>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
          {/* Chart 1: Reward Curve */}
          <SectionCard title="Episode Reward Curve" subtitle="Mean Reward vs Steps">
            <div className="h-40 bg-surface-dark/80 rounded-lg border border-surface-border flex flex-col items-center justify-center space-y-2 p-4">
              <LineChart className="w-8 h-8 text-accent/50" />
              <span className="text-xs font-mono text-slate-400">Reward Curve Chart Container</span>
              <div className="w-full h-1 bg-gradient-to-r from-blue-600 via-emerald-400 to-accent rounded"></div>
            </div>
          </SectionCard>

          {/* Chart 2: Throughput */}
          <SectionCard title="Fleet Throughput" subtitle="Deliveries per Step">
            <div className="h-40 bg-surface-dark/80 rounded-lg border border-surface-border flex flex-col items-center justify-center space-y-2 p-4">
              <BarChart2 className="w-8 h-8 text-electric/50" />
              <span className="text-xs font-mono text-slate-400">Throughput Chart Container</span>
              <div className="w-full h-1 bg-gradient-to-r from-cyan-500 to-blue-400 rounded"></div>
            </div>
          </SectionCard>

          {/* Chart 3: Collisions */}
          <SectionCard title="Dynamic Collisions" subtitle="Collisions vs Time (DAM Active)">
            <div className="h-40 bg-surface-dark/80 rounded-lg border border-surface-border flex flex-col items-center justify-center space-y-2 p-4">
              <Shield className="w-8 h-8 text-emerald-400/50" />
              <span className="text-xs font-mono text-emerald-400">0 Collisions Detected</span>
              <div className="w-full h-1 bg-emerald-500 rounded"></div>
            </div>
          </SectionCard>

          {/* Chart 4: Battery Usage */}
          <SectionCard title="Battery Level Distribution" subtitle="Average Fleet Battery State">
            <div className="h-40 bg-surface-dark/80 rounded-lg border border-surface-border flex flex-col items-center justify-center space-y-2 p-4">
              <Battery className="w-8 h-8 text-amber-400/50" />
              <span className="text-xs font-mono text-slate-400">Battery State Container</span>
              <div className="w-full h-1 bg-gradient-to-r from-emerald-400 via-amber-400 to-rose-500 rounded"></div>
            </div>
          </SectionCard>

          {/* Chart 5: Training Loss */}
          <SectionCard title="Critic Value Loss" subtitle="Centralized Critic MSE Loss">
            <div className="h-40 bg-surface-dark/80 rounded-lg border border-surface-border flex flex-col items-center justify-center space-y-2 p-4">
              <Activity className="w-8 h-8 text-purple-400/50" />
              <span className="text-xs font-mono text-slate-400">Critic Loss Container</span>
              <div className="w-full h-1 bg-gradient-to-r from-purple-500 to-indigo-400 rounded"></div>
            </div>
          </SectionCard>

          {/* Chart 6: Policy Entropy */}
          <SectionCard title="Policy Entropy" subtitle="Actor Exploration Entropy">
            <div className="h-40 bg-surface-dark/80 rounded-lg border border-surface-border flex flex-col items-center justify-center space-y-2 p-4">
              <Zap className="w-8 h-8 text-indigo-400/50" />
              <span className="text-xs font-mono text-slate-400">Entropy Curve Container</span>
              <div className="w-full h-1 bg-gradient-to-r from-indigo-500 to-accent rounded"></div>
            </div>
          </SectionCard>
        </div>
      </div>
    </Container>
  );
};
