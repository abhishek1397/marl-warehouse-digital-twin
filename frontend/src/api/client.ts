/**
 * REST API Client for Warehouse Digital Twin Backend.
 */

const API_BASE_URL = 'http://127.0.0.1:8000/api';

export interface ApiSimulationState {
  is_initialized: boolean;
  is_running: boolean;
  is_paused: boolean;
  step_count: number;
  grid_size: [number, number];
  robots: Array<{
    id: string;
    position: [number, number];
    battery_level: number;
    state: string;
    assigned_task?: string;
  }>;
  entities: Array<{
    id: string;
    position: [number, number];
    type: 'shelf' | 'charging_station' | 'package' | 'obstacle' | 'goal';
  }>;
  metrics: {
    episode: number;
    step: number;
    reward: number;
    throughput: number;
    collisions: number;
    idle_robots: number;
    battery_avg: number;
    packages_delivered: number;
    fps: number;
    policy_entropy: number;
  };
}

async function request<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const url = `${API_BASE_URL}${endpoint}`;
  try {
    const res = await fetch(url, {
      headers: {
        'Content-Type': 'application/json',
      },
      ...options,
    });
    if (!res.ok) {
      const errText = await res.text();
      throw new Error(`API Error ${res.status}: ${errText}`);
    }
    return (await res.json()) as T;
  } catch (err) {
    console.warn(`[ApiClient] Request to ${endpoint} failed, falling back to local state:`, err);
    throw err;
  }
}

export const ApiClient = {
  fetchHealth: async () => request<{ status: string }>('/health'),

  fetchState: async () => request<ApiSimulationState>('/simulation/state'),

  createSimulation: async (gridWidth: number, gridHeight: number, numRobots: number) =>
    request<ApiSimulationState>('/simulation/create', {
      method: 'POST',
      body: JSON.stringify({
        grid_width: gridWidth,
        grid_height: gridHeight,
        num_robots: numRobots,
        enable_pbrs: true,
        enable_dam: true,
      }),
    }),

  startSimulation: async () => request<ApiSimulationState>('/simulation/start', { method: 'POST' }),

  pauseSimulation: async () => request<ApiSimulationState>('/simulation/pause', { method: 'POST' }),

  resetSimulation: async () => request<ApiSimulationState>('/simulation/reset', { method: 'POST' }),

  stepSimulation: async (steps: number = 1) =>
    request<ApiSimulationState>('/simulation/step', {
      method: 'POST',
      body: JSON.stringify({ steps }),
    }),

  selectAlgorithm: async (algorithmName: string) =>
    request<{ message: string }>('/algorithms/select', {
      method: 'POST',
      body: JSON.stringify({ algorithm_name: algorithmName }),
    }),
};
