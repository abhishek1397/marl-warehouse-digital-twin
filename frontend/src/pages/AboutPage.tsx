import React from 'react';
import { Bot, Shield, Award } from 'lucide-react';
import { Container } from '../components/Container';
import { PageHeader } from '../components/PageHeader';
import { SectionCard } from '../components/SectionCard';

export const AboutPage: React.FC = () => {
  return (
    <Container className="py-6 space-y-6">
      <PageHeader
        title="About the Platform"
        subtitle="AI Robotics MARL Research Platform for Autonomous Multi-Robot Logistics."
      />

      <SectionCard title="Project Overview">
        <div className="space-y-4 text-xs text-slate-300 leading-relaxed max-w-3xl">
          <p>
            The Warehouse Digital Twin MARL Platform is a production-grade multi-agent reinforcement learning system designed to solve complex multi-robot warehouse fleet routing, charging, and package delivery tasks.
          </p>
          <p>
            The platform brings together classical path planning (A* with time-space reservation tables) and modern MARL algorithms (PPO, IPPO, MAPPO, Spatial MAPPO with CNN Centralized Critic) into a unified, reproducible research environment.
          </p>
        </div>
      </SectionCard>
    </Container>
  );
};
