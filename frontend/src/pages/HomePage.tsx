import React from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { 
  Bot, Cpu, Zap, ShieldCheck, Play, FileText, Github, BookOpen, 
  Layers, Server, Cloud, Activity, CheckCircle2, ArrowDown, ArrowRight, 
  BarChart3, Database, Workflow, Terminal, Compass 
} from 'lucide-react';
import { Container } from '../components/Container';
import { Button } from '../components/Button';
import { SectionCard } from '../components/SectionCard';
import { Badge } from '../components/Badge';

export const HomePage: React.FC = () => {
  const navigate = useNavigate();

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: { staggerChildren: 0.1, delayChildren: 0.1 },
    },
  };

  const itemVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { opacity: 1, y: 0, transition: { duration: 0.5, ease: 'easeOut' } },
  };

  return (
    <div className="space-y-24 py-6">
      {/* HERO SECTION */}
      <section className="relative overflow-hidden pt-12 pb-20 border-b border-surface-border bg-grid-pattern">
        {/* Glowing Orbs */}
        <div className="absolute top-1/4 left-1/2 -translate-x-1/2 -translate-y-1/2 w-96 h-96 bg-blue-500/15 rounded-full blur-3xl pointer-events-none"></div>
        <div className="absolute top-1/3 left-1/4 w-64 h-64 bg-cyan-500/10 rounded-full blur-3xl pointer-events-none"></div>

        <Container className="relative z-10">
          <motion.div
            variants={containerVariants}
            initial="hidden"
            animate="visible"
            className="text-center max-w-4xl mx-auto space-y-6"
          >
            <motion.div variants={itemVariants} className="inline-block">
              <Badge variant="info" className="px-3.5 py-1 text-xs tracking-wide uppercase font-mono">
                AI MARL Robotics Research Platform
              </Badge>
            </motion.div>

            <motion.h1 
              variants={itemVariants}
              className="text-4xl sm:text-6xl md:text-7xl font-black text-white tracking-tight leading-none font-sans"
            >
              Warehouse Digital Twin
              <span className="block text-transparent bg-clip-text bg-gradient-to-r from-accent via-blue-400 to-electric mt-2">
                Multi-Agent Reinforcement Learning
              </span>
            </motion.h1>

            {/* 3 Paragraph Project Description */}
            <motion.div variants={itemVariants} className="space-y-4 text-slate-300 text-sm sm:text-base leading-relaxed text-justify max-w-3xl mx-auto font-normal">
              <p>
                An end-to-end research platform designed for multi-robot warehouse fleet coordination, battery charging scheduling, and collision-free package delivery. The digital twin simulator models real-world grid constraints, charging station capacity, dynamic obstacles, and task scheduling under bounded battery dynamics.
              </p>
              <p>
                By bridging classical time-space path planning (A* with reservation tables) and modern Multi-Agent Reinforcement Learning (MARL) operating under the Centralized Training Decentralized Execution (CTDE) paradigm, the platform evaluates multi-agent fleet scalability from single-agent baselines to 32-robot autonomous fleets.
              </p>
              <p>
                Key scientific innovations include Ng et al. (1999) Potential-Based Reward Shaping (PBRS) for optimal policy invariance, Dynamic Action Masking (DAM) achieving 100% collision elimination, and Spatial MAPPO featuring 2D Convolutional Neural Network (CNN) Centralized Value Networks.
              </p>
            </motion.div>

            {/* Buttons Group */}
            <motion.div variants={itemVariants} className="pt-4 flex flex-wrap items-center justify-center gap-4">
              <Button
                size="lg"
                variant="primary"
                icon={<Play className="w-5 h-5 fill-current" />}
                onClick={() => navigate('/simulation')}
              >
                Launch Simulation
              </Button>
              <Button
                size="lg"
                variant="outline"
                icon={<FileText className="w-5 h-5" />}
                onClick={() => navigate('/research')}
              >
                Explore Research
              </Button>
              <a
                href="https://github.com"
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center justify-center font-medium rounded-lg px-6 py-3 text-base text-slate-300 bg-surface border border-surface-border hover:bg-slate-800 hover:text-white transition-all duration-200"
              >
                <Github className="w-5 h-5 mr-2" />
                GitHub
              </a>
              <Button
                size="lg"
                variant="ghost"
                icon={<BookOpen className="w-5 h-5" />}
                onClick={() => navigate('/docs')}
              >
                Documentation
              </Button>
            </motion.div>
          </motion.div>
        </Container>
      </section>

      {/* PROJECT OVERVIEW (4 FEATURE CARDS) */}
      <Container>
        <div className="text-center max-w-2xl mx-auto mb-12 space-y-2">
          <Badge variant="neutral">Core Platform Pillars</Badge>
          <h2 className="text-3xl font-extrabold text-white tracking-tight">Project Overview</h2>
          <p className="text-xs sm:text-sm text-slate-400">Fundamental sub-systems powering the digital twin research environment</p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <motion.div whileHover={{ y: -4 }} className="glass-card p-6 rounded-xl border border-surface-border flex flex-col justify-between space-y-4">
            <div className="space-y-3">
              <div className="p-3 w-fit rounded-lg bg-blue-500/10 text-accent border border-blue-500/20">
                <Layers className="w-6 h-6" />
              </div>
              <h3 className="text-base font-bold text-white">Warehouse Digital Twin</h3>
              <p className="text-xs text-slate-400 leading-relaxed">
                High-fidelity 2D grid simulator with cell states, A* reservation tables, charging stations, shelves, and battery managers.
              </p>
            </div>
            <span className="text-[11px] font-mono text-accent font-semibold">Simulator Core →</span>
          </motion.div>

          <motion.div whileHover={{ y: -4 }} className="glass-card p-6 rounded-xl border border-surface-border flex flex-col justify-between space-y-4">
            <div className="space-y-3">
              <div className="p-3 w-fit rounded-lg bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                <Cpu className="w-6 h-6" />
              </div>
              <h3 className="text-base font-bold text-white">Multi-Agent RL Suite</h3>
              <p className="text-xs text-slate-400 leading-relaxed">
                Comprehensive MARL algorithm library featuring Gym PPO, IPPO, MAPPO (MLP Critic), and Spatial MAPPO (CNN Critic).
              </p>
            </div>
            <span className="text-[11px] font-mono text-emerald-400 font-semibold">MARL Package →</span>
          </motion.div>

          <motion.div whileHover={{ y: -4 }} className="glass-card p-6 rounded-xl border border-surface-border flex flex-col justify-between space-y-4">
            <div className="space-y-3">
              <div className="p-3 w-fit rounded-lg bg-cyan-500/10 text-electric border border-cyan-500/20">
                <Cloud className="w-6 h-6" />
              </div>
              <h3 className="text-base font-bold text-white">Cloud-Native Deployment</h3>
              <p className="text-xs text-slate-400 leading-relaxed">
                FastAPI microservice backend architecture with REST API endpoints, CORS configuration, and Docker container readiness.
              </p>
            </div>
            <span className="text-[11px] font-mono text-electric font-semibold">FastAPI Backend →</span>
          </motion.div>

          <motion.div whileHover={{ y: -4 }} className="glass-card p-6 rounded-xl border border-surface-border flex flex-col justify-between space-y-4">
            <div className="space-y-3">
              <div className="p-3 w-fit rounded-lg bg-purple-500/10 text-purple-400 border border-purple-500/20">
                <Activity className="w-6 h-6" />
              </div>
              <h3 className="text-base font-bold text-white">Interactive Platform</h3>
              <p className="text-xs text-slate-400 leading-relaxed">
                Modern React 18 + Vite frontend with glassmorphism UI, real-time simulation controls, and diagnostic telemetry.
              </p>
            </div>
            <span className="text-[11px] font-mono text-purple-400 font-semibold">Web Interface →</span>
          </motion.div>
        </div>
      </Container>

      {/* RESEARCH PIPELINE VERTICAL TIMELINE */}
      <Container>
        <div className="text-center max-w-2xl mx-auto mb-12 space-y-2">
          <Badge variant="info">Incremental Milestones</Badge>
          <h2 className="text-3xl font-extrabold text-white tracking-tight">Research Journey Pipeline</h2>
          <p className="text-xs sm:text-sm text-slate-400">Chronological trajectory of MARL algorithm design and empirical verification</p>
        </div>

        <div className="relative max-w-3xl mx-auto">
          {/* Central Vertical Line */}
          <div className="absolute left-4 sm:left-1/2 top-0 bottom-0 w-0.5 bg-gradient-to-b from-blue-500 via-accent to-purple-500 -translate-x-1/2 hidden sm:block"></div>

          <div className="space-y-8">
            {[
              { title: "1. Classical A* Path Planning", desc: "Space-Time Reservation Table avoiding dynamic collisions with time-indexed coordinates.", tag: "Planning Engine" },
              { title: "2. Single-Agent PPO Baseline", desc: "Validated PPO implementation on single-robot Gymnasium warehouse environment.", tag: "RL Baseline" },
              { title: "3. Potential-Based Reward Shaping (PBRS)", desc: "Theoretical PBRS implementation following Ng et al. (1999) preserving optimal policy invariance.", tag: "Reward Engineering" },
              { title: "4. Dynamic Action Masking (DAM)", desc: "Environment-level action masking preventing illegal obstacle sampling with 100% collision elimination.", tag: "Constraint Enforcement" },
              { title: "5. Independent PPO (IPPO)", desc: "Decentralized multi-robot fleet baseline with shared actor parameter optimization.", tag: "MARL Baseline" },
              { title: "6. MAPPO (MLP Critic CTDE)", desc: "Centralized Training Decentralized Execution paradigm with flat global state value networks.", tag: "CTDE Paradigm" },
              { title: "7. Spatial MAPPO (S-MAPPO 2D CNN)", desc: "2D Spatial CNN Centralized Critic processing multi-channel state grid tensors with O(1) scalability.", tag: "Primary Innovation" },
            ].map((step, idx) => (
              <div key={idx} className="relative flex flex-col sm:flex-row items-center justify-between group">
                <div className={`w-full sm:w-5/12 ${idx % 2 === 0 ? 'sm:text-right sm:pr-8' : 'sm:order-2 sm:pl-8'}`}>
                  <div className="glass-panel p-5 rounded-xl border border-surface-border space-y-2 group-hover:border-accent/40 transition-all">
                    <div className="flex items-center justify-between sm:justify-start space-x-2">
                      <h4 className="text-sm font-bold text-white">{step.title}</h4>
                      <Badge variant="success" className="text-[10px]">Completed</Badge>
                    </div>
                    <p className="text-xs text-slate-400 leading-relaxed">{step.desc}</p>
                    <span className="inline-block text-[10px] font-mono text-accent bg-blue-500/10 px-2 py-0.5 rounded border border-blue-500/20">{step.tag}</span>
                  </div>
                </div>

                {/* Timeline Center Dot */}
                <div className="absolute left-4 sm:left-1/2 -translate-x-1/2 w-8 h-8 rounded-full bg-surface-dark border-2 border-accent flex items-center justify-center text-accent z-10 shadow-glow hidden sm:flex">
                  <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                </div>

                <div className="w-full sm:w-5/12 hidden sm:block"></div>
              </div>
            ))}
          </div>
        </div>
      </Container>

      {/* TECHNOLOGY STACK */}
      <Container>
        <div className="text-center max-w-2xl mx-auto mb-12 space-y-2">
          <Badge variant="neutral">Full-Stack Tech Stack</Badge>
          <h2 className="text-3xl font-extrabold text-white tracking-tight">Technology Stack</h2>
          <p className="text-xs sm:text-sm text-slate-400">Production technologies across frontend, backend, AI/ML, and cloud layers</p>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
          <div className="glass-card p-5 rounded-xl border border-surface-border space-y-3">
            <h4 className="text-xs font-bold uppercase text-accent tracking-wider font-mono">Frontend</h4>
            <ul className="text-xs text-slate-300 space-y-1.5 font-medium">
              <li>• React 18</li>
              <li>• TypeScript</li>
              <li>• Vite</li>
              <li>• Tailwind CSS</li>
              <li>• Zustand</li>
              <li>• Framer Motion</li>
            </ul>
          </div>

          <div className="glass-card p-5 rounded-xl border border-surface-border space-y-3">
            <h4 className="text-xs font-bold uppercase text-emerald-400 tracking-wider font-mono">Backend</h4>
            <ul className="text-xs text-slate-300 space-y-1.5 font-medium">
              <li>• Python 3.11</li>
              <li>• FastAPI</li>
              <li>• Uvicorn</li>
              <li>• Pydantic v2</li>
              <li>• CORS Middleware</li>
            </ul>
          </div>

          <div className="glass-card p-5 rounded-xl border border-surface-border space-y-3">
            <h4 className="text-xs font-bold uppercase text-electric tracking-wider font-mono">AI / ML</h4>
            <ul className="text-xs text-slate-300 space-y-1.5 font-medium">
              <li>• PyTorch 2.x</li>
              <li>• PettingZoo Parallel</li>
              <li>• Gymnasium</li>
              <li>• NumPy & SciPy</li>
              <li>• GAE Advantage</li>
            </ul>
          </div>

          <div className="glass-card p-5 rounded-xl border border-surface-border space-y-3">
            <h4 className="text-xs font-bold uppercase text-purple-400 tracking-wider font-mono">Cloud</h4>
            <ul className="text-xs text-slate-300 space-y-1.5 font-medium">
              <li>• Docker Containers</li>
              <li>• REST Protocols</li>
              <li>• Async Event Loops</li>
              <li>• Scalable Runners</li>
            </ul>
          </div>

          <div className="glass-card p-5 rounded-xl border border-surface-border space-y-3">
            <h4 className="text-xs font-bold uppercase text-amber-400 tracking-wider font-mono">DevOps</h4>
            <ul className="text-xs text-slate-300 space-y-1.5 font-medium">
              <li>• Pytest Suite</li>
              <li>• Coverage Analysis</li>
              <li>• Unified Logger</li>
              <li>• Checkpoint Manager</li>
            </ul>
          </div>
        </div>
      </Container>

      {/* VISUAL ARCHITECTURE SECTION */}
      <Container>
        <div className="text-center max-w-2xl mx-auto mb-12 space-y-2">
          <Badge variant="info">End-to-End Flow</Badge>
          <h2 className="text-3xl font-extrabold text-white tracking-tight">System Architecture Diagram</h2>
          <p className="text-xs sm:text-sm text-slate-400">Data flow from browser user interface to digital twin simulation and MARL optimization</p>
        </div>

        <div className="glass-panel p-8 rounded-2xl border border-surface-border overflow-x-auto">
          <div className="flex flex-col md:flex-row items-center justify-between gap-4 min-w-[700px]">
            <div className="glass-card p-4 rounded-xl text-center border border-accent/40 w-40">
              <span className="text-xs font-mono text-accent font-bold uppercase block mb-1">Layer 1</span>
              <span className="text-sm font-bold text-white block">Browser UI</span>
              <span className="text-[10px] text-slate-400 block mt-1">React 18 + Tailwind</span>
            </div>

            <ArrowRight className="w-6 h-6 text-accent hidden md:block" />
            <ArrowDown className="w-6 h-6 text-accent md:hidden" />

            <div className="glass-card p-4 rounded-xl text-center border border-emerald-500/40 w-40">
              <span className="text-xs font-mono text-emerald-400 font-bold uppercase block mb-1">Layer 2</span>
              <span className="text-sm font-bold text-white block">FastAPI</span>
              <span className="text-[10px] text-slate-400 block mt-1">Uvicorn Server</span>
            </div>

            <ArrowRight className="w-6 h-6 text-emerald-400 hidden md:block" />
            <ArrowDown className="w-6 h-6 text-emerald-400 md:hidden" />

            <div className="glass-card p-4 rounded-xl text-center border border-electric/40 w-40">
              <span className="text-xs font-mono text-electric font-bold uppercase block mb-1">Layer 3</span>
              <span className="text-sm font-bold text-white block">Simulator</span>
              <span className="text-[10px] text-slate-400 block mt-1">Digital Twin Grid</span>
            </div>

            <ArrowRight className="w-6 h-6 text-electric hidden md:block" />
            <ArrowDown className="w-6 h-6 text-electric md:hidden" />

            <div className="glass-card p-4 rounded-xl text-center border border-purple-500/40 w-40">
              <span className="text-xs font-mono text-purple-400 font-bold uppercase block mb-1">Layer 4</span>
              <span className="text-sm font-bold text-white block">MARL Engine</span>
              <span className="text-[10px] text-slate-400 block mt-1">PPO / IPPO / S-MAPPO</span>
            </div>

            <ArrowRight className="w-6 h-6 text-purple-400 hidden md:block" />
            <ArrowDown className="w-6 h-6 text-purple-400 md:hidden" />

            <div className="glass-card p-4 rounded-xl text-center border border-amber-500/40 w-40">
              <span className="text-xs font-mono text-amber-400 font-bold uppercase block mb-1">Layer 5</span>
              <span className="text-sm font-bold text-white block">Metrics</span>
              <span className="text-[10px] text-slate-400 block mt-1">Unified Diagnostics</span>
            </div>
          </div>
        </div>
      </Container>

      {/* PROJECT HIGHLIGHTS */}
      <Container>
        <div className="text-center max-w-2xl mx-auto mb-12 space-y-2">
          <Badge variant="neutral">Key Features</Badge>
          <h2 className="text-3xl font-extrabold text-white tracking-tight">Project Highlights</h2>
          <p className="text-xs sm:text-sm text-slate-400">Verified platform capabilities and scientific contributions</p>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
          <SectionCard title="Production Architecture" subtitle="Clean Python & PyTorch Package">
            <p className="text-xs text-slate-300">Modular design with strict separation of simulator, parallel environment, algorithm implementations, and diagnostic utilities.</p>
          </SectionCard>

          <SectionCard title="Multiple MARL Algorithms" subtitle="PPO, IPPO, MAPPO, S-MAPPO">
            <p className="text-xs text-slate-300">Supports single-agent PPO, independent IPPO, flat MLP MAPPO, and 2D spatial CNN Spatial MAPPO algorithm variants.</p>
          </SectionCard>

          <SectionCard title="Scientific Evaluation" subtitle="Multi-Fleet Diagnostic Suite">
            <p className="text-xs text-slate-300">Empirical diagnostic analysis measuring critic explained variance, collision avoidance rates, Jain's fairness index, and step latency.</p>
          </SectionCard>

          <SectionCard title="Statistical Validation" subtitle="10-Seed Hypothesis Testing">
            <p className="text-xs text-slate-300">Statistical testing framework computing paired t-tests, Wilcoxon signed-rank tests, 95% confidence intervals, and Cohen's d effect sizes.</p>
          </SectionCard>

          <SectionCard title="Interactive Visualization" subtitle="Spatial Activation Heatmaps">
            <p className="text-xs text-slate-300">Feature visualizers rendering 2D spatial CNN activation heatmaps, channel importance maps, and trajectory animations.</p>
          </SectionCard>

          <SectionCard title="Cloud Deployment Ready" subtitle="FastAPI & Microservice Ready">
            <p className="text-xs text-slate-300">Fully configured FastAPI REST backend with health checks, CORS policy, and React 18 production build optimization.</p>
          </SectionCard>
        </div>
      </Container>
    </div>
  );
};
