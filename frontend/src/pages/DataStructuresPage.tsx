import React, { useState, useMemo } from 'react';
import { Container } from '../components/Container';
import {
  Database,
  Search,
  Filter,
  Code,
  HelpCircle,
  ChevronDown,
  ChevronUp,
  Cpu,
  Play,
  Zap,
  CheckCircle2,
  GitBranch,
  Clock,
  HardDrive,
  Layers,
  ArrowRight,
  Terminal,
  Activity,
  FileCode,
} from 'lucide-react';

export interface DataStructureItem {
  id: string;
  name: string;
  category: 'Simulation' | 'Planning' | 'RL' | 'Backend' | 'Frontend';
  definition: string;
  whySelected: string;
  filePath: string;
  classesUsed: string[];
  functionsUsed: string[];
  timeComplexity: {
    access: string;
    search: string;
    insertion: string;
    deletion: string;
  };
  spaceComplexity: string;
  advantages: string[];
  limitations: string[];
  alternatives: string[];
  codeSnippet: string;
  interviewQuestions: {
    question: string;
    answer: string;
  }[];
  memoryLayoutDiagram?: string;
}

export const DATA_STRUCTURES_CATALOG: DataStructureItem[] = [
  {
    id: 'grid-matrix',
    name: 'Warehouse 2D Cell Matrix (2D Array)',
    category: 'Simulation',
    definition: 'A 2-dimensional contiguous matrix array representing the spatial floor plan of the warehouse digital twin.',
    whySelected: 'Enables O(1) constant-time coordinate lookup _cells[y][x] and instant 4-cardinal / 8-diagonal neighbor retrieval.',
    filePath: 'simulator/grid.py',
    classesUsed: ['Grid', 'Warehouse', 'Cell'],
    functionsUsed: ['get_cell()', 'is_in_bounds()', 'get_neighbors()', 'get_walkable_neighbors()'],
    timeComplexity: {
      access: 'O(1)',
      search: 'O(W × H)',
      insertion: 'O(1)',
      deletion: 'N/A (Fixed Bounds)',
    },
    spaceComplexity: 'O(W × H)',
    advantages: [
      'Constant-time O(1) coordinate indexing without hash calculation overhead.',
      'Optimal CPU L1/L2 cache locality for contiguous row-major spatial iterations.',
      'Simple, deterministic boundary checking.',
    ],
    limitations: [
      'Allocates contiguous memory for all cells, even for highly sparse warehouse maps.',
      'Fixed dimensions at initialization; resizing requires full memory re-allocation.',
    ],
    alternatives: ['Sparse Hash Map Grid', 'QuadTree', 'Octree'],
    codeSnippet: `class Grid:
    """Manages a 2D matrix of Cell instances representing the warehouse floor."""
    def __init__(self, width: int = 20, height: int = 20):
        self._width = width
        self._height = height
        self._cells = [
            [Cell(position=Position(x=x, y=y)) for x in range(width)]
            for y in range(height)
        ]

    def get_cell(self, position: Position) -> Cell:
        if not self.is_in_bounds(position):
            raise OutOfBoundsError(position.x, position.y)
        return self._cells[position.y][position.x]`,
    interviewQuestions: [
      {
        question: 'Why choose a 2D Array over a Hash Map for representing a warehouse grid?',
        answer: 'A 2D array provides true O(1) random access via index arithmetic _cells[y][x] with zero hashing overhead, zero bucket collisions, and superior CPU L1/L2 cache line prefetching. Hash maps carry overhead per key lookup and higher memory fragmentation for dense grids.',
      },
      {
        question: 'What are the spatial memory trade-offs when scaling from a 20x20 grid to a 200x200 grid?',
        answer: 'Memory scales quadratically: O(W × H). A 200x200 grid contains 40,000 Cell objects. For extremely large sparse worlds (>10,000x10,000), a QuadTree or Hash Map sparse grid is preferred to avoid allocating empty space.',
      },
    ],
    memoryLayoutDiagram: `[Row 0]: [Cell(0,0)] -> [Cell(1,0)] -> [Cell(2,0)] ...
[Row 1]: [Cell(0,1)] -> [Cell(1,1)] -> [Cell(2,1)] ...
[Row Y]: [Cell(0,Y)] -> [Cell(X,Y)] ... (Direct Row-Major Offset Arithmetic)`,
  },
  {
    id: 'position-dataclass',
    name: 'Immutable Position Dataclass',
    category: 'Simulation',
    definition: 'An immutable frozen dataclass representing discrete 2D spatial coordinates (x, y).',
    whySelected: 'Guarantees hashability for use as dictionary keys and set elements while encapsulating Manhattan (L1) and Euclidean (L2) distance functions.',
    filePath: 'simulator/position.py',
    classesUsed: ['Position'],
    functionsUsed: ['manhattan_distance()', 'euclidean_distance()', 'get_neighbor()', 'to_tuple()'],
    timeComplexity: {
      access: 'O(1)',
      search: 'O(1)',
      insertion: 'O(1)',
      deletion: 'N/A',
    },
    spaceComplexity: 'O(1)',
    advantages: [
      'Immutable @dataclass(frozen=True) prevents accidental coordinate mutation.',
      'Implements __hash__ and __eq__ automatically for instant dictionary and set lookups.',
      'Clean domain methods for L1 Manhattan distance calculations.',
    ],
    limitations: [
      'Creating new Position instances in tight inner loops adds Python object creation overhead compared to raw tuples.',
    ],
    alternatives: ['NamedTuple', 'Raw Integer Tuple (x, y)', 'Cython / C++ Struct'],
    codeSnippet: `@dataclass(frozen=True, order=True)
class Position:
    x: int
    y: int

    def manhattan_distance(self, other: "Position") -> int:
        return abs(self.x - other.x) + abs(self.y - other.y)

    def get_neighbor(self, direction: Direction) -> "Position":
        return Position(self.x + direction.dx, self.y + direction.dy)`,
    interviewQuestions: [
      {
        question: 'Why make the Position dataclass frozen and immutable?',
        answer: 'Frozen dataclasses generate a deterministic __hash__ implementation based on their attribute values. If coordinates were mutable, changing x or y while stored inside a Set or Dict key would corrupt the hash table bucket placement, leading to unrecoverable key lookup bugs.',
      },
    ],
  },
  {
    id: 'task-fifo-queue',
    name: 'Task Management FIFO Double-Ended Queue (deque)',
    category: 'Simulation',
    definition: 'A doubly-linked list queue (collections.deque) managing pending package pickup and delivery tasks.',
    whySelected: 'Enforces O(1) task assignment from the head (popleft()) for idle robots while guaranteeing Single-Robot Task Allocation.',
    filePath: 'simulator/task_manager.py',
    classesUsed: ['TaskManager'],
    functionsUsed: ['assign_next_task()', 'create_task()', 'cancel_task()', 'get_pending_tasks()'],
    timeComplexity: {
      access: 'O(N)',
      search: 'O(N)',
      insertion: 'O(1) append',
      deletion: 'O(1) popleft',
    },
    spaceComplexity: 'O(T) where T is pending tasks',
    advantages: [
      'O(1) push and pop from both ends without memory shifts.',
      'Prevents duplicate task assignments across multi-robot fleets.',
    ],
    limitations: [
      'O(N) search and random access compared to arrays.',
    ],
    alternatives: ['Priority Queue', 'Ring Buffer'],
    codeSnippet: `class TaskManager:
    def __init__(self):
        self._tasks: Dict[str, Task] = {}
        self._unassigned_queue: deque[Task] = deque()

    def assign_next_task(self, robot: Robot) -> Optional[Task]:
        if not robot.is_idle():
            return None
        if not self._unassigned_queue:
            return None
        task = self._unassigned_queue.popleft()
        robot.assign_task(task)
        return task`,
    interviewQuestions: [
      {
        question: 'Why use collections.deque for the unassigned task queue instead of a Python list?',
        answer: 'Popping from the front of a Python list via list.pop(0) takes O(N) time because all remaining elements must shift left in contiguous memory. collections.deque is implemented as a doubly-linked list of memory blocks, making popleft() an O(1) operation.',
      },
    ],
  },
  {
    id: 'astar-min-heap',
    name: 'Space-Time A* Search Min-Heap Priority Queue (heapq)',
    category: 'Planning',
    definition: 'A binary min-heap priority queue storing search state nodes ordered by evaluation score f(n) = g(n) + h(n).',
    whySelected: 'Guarantees that the space-time node with the lowest estimated total cost is expanded next during pathfinding.',
    filePath: 'simulator/astar.py',
    classesUsed: ['SpaceTimeAStar'],
    functionsUsed: ['plan_path()', '_reconstruct_path()', 'heappush()', 'heappop()'],
    timeComplexity: {
      access: 'O(1) min element',
      search: 'O(N)',
      insertion: 'O(log N)',
      deletion: 'O(log N) min pop',
    },
    spaceComplexity: 'O(N) search tree nodes',
    advantages: [
      'Logarithmic O(log N) insertion and minimum node extraction.',
      'Guarantees optimal space-time path expansion order.',
    ],
    limitations: [
      'Python heapq does not support O(1) decrease-key operations.',
    ],
    alternatives: ['Fibonacci Heap', 'Pairing Heap', 'D-ary Heap'],
    codeSnippet: `def plan_path(self, start: Position, goal: Position, start_time: int = 0):
    open_set = []
    # Node tuple: (f_score, h_score, tie_breaker, position, time_step)
    heapq.heappush(open_set, (h_start, h_start, count, start, start_time))
    
    while open_set:
        f, h, _, curr_pos, t = heapq.heappop(open_set)
        if curr_pos == goal:
            return self._reconstruct_path(parent_map, curr_state)`,
    interviewQuestions: [
      {
        question: 'How does Space-Time A* differ from standard 2D A* in terms of data structures?',
        answer: 'Standard A* searches states (x, y), whereas Space-Time A* searches states (x, y, t). The priority queue and visited sets must store 3D state tuples (x, y, t) to allow robots to wait in place or detour around dynamic dynamic reservations.',
      },
    ],
    memoryLayoutDiagram: `Binary Min-Heap Array Layout:
[0: Min Node (f=2)]
├── [1: Node (f=4)]
└── [2: Node (f=3)]
    ├── [3: Node (f=7)]
    └── [4: Node (f=5)]`,
  },
  {
    id: 'reservation-table',
    name: 'Space-Time Reservation Table (Hash Maps)',
    category: 'Planning',
    definition: 'A centralized space-time conflict resolution table using two Python Hash Maps (Dict) to lock vertex and edge transitions.',
    whySelected: 'Provides O(1) space-time vertex collision and swap collision checks across multi-robot trajectories.',
    filePath: 'simulator/reservation_table.py',
    classesUsed: ['ReservationTable', 'MultiRobotPlanner'],
    functionsUsed: ['reserve_path()', 'is_vertex_reserved()', 'is_edge_reserved()', 'reserve_vertex()'],
    timeComplexity: {
      access: 'O(1) query',
      search: 'O(1) hash check',
      insertion: 'O(1) reservation',
      deletion: 'O(1) clear',
    },
    spaceComplexity: 'O(R × L) where R is robots and L is path length',
    advantages: [
      'Instantaneous O(1) space-time collision lookup.',
      'Prevents both vertex collisions (two robots at same x,y,t) and edge swap collisions (robots swapping x1,y1 <-> x2,y2 at t).',
    ],
    limitations: [
      'Memory grows linearly with trajectory time horizon.',
    ],
    alternatives: ['3D Space-Time Grid Matrix bool[x][y][t]', 'Interval Trees'],
    codeSnippet: `class ReservationTable:
    def __init__(self):
        # Vertex Map: (x, y, t) -> agent_id
        self._vertex_reservations: Dict[Tuple[int, int, int], str] = {}
        # Edge Map: ((x1, y1), (x2, y2), t) -> agent_id
        self._edge_reservations: Dict[Tuple[Tuple[int, int], Tuple[int, int], int], str] = {}

    def is_vertex_reserved(self, pos: Position, t: int) -> bool:
        return (pos.x, pos.y, t) in self._vertex_reservations`,
    interviewQuestions: [
      {
        question: 'Why are two separate hash maps needed for space-time reservation?',
        answer: 'Vertex reservation maps (x, y, t) prevent two robots from occupying the exact same cell at the same time. Edge reservation maps ((x1,y1), (x2,y2), t) prevent two robots from swapping places simultaneously across an edge, which would cause physical collisions in a real warehouse.',
      },
    ],
  },
  {
    id: 'rollout-buffer-tensor',
    name: 'Multi-Agent Rollout Buffer (PyTorch Tensors)',
    category: 'RL',
    definition: 'A contiguous tensor buffer storing rollout transition tuples (s_t, a_t, r_t, s_t+1, log_prob_t, V(s_t), mask_t) for MARL updates.',
    whySelected: 'Enables fast parallel GPU mini-batch sampling and Generalized Advantage Estimation (GAE) computation.',
    filePath: 'marl/storage/rollout_buffer.py',
    classesUsed: ['RolloutBuffer', 'IPPORolloutManager'],
    functionsUsed: ['add()', 'get_generator()', 'compute_returns_and_advantage()'],
    timeComplexity: {
      access: 'O(1)',
      search: 'N/A',
      insertion: 'O(1) step fill',
      deletion: 'O(1) reset',
    },
    spaceComplexity: 'O(T × N × D) for T steps, N agents, D dims',
    advantages: [
      'Contiguous PyTorch GPU memory layout for high-throughput mini-batch training.',
      'Vectorized GAE advantage calculation.',
    ],
    limitations: [
      'Pre-allocated fixed buffer size requires RAM pre-allocation.',
    ],
    alternatives: ['Prioritized Experience Replay (PER)', 'Circular Buffer'],
    codeSnippet: `class RolloutBuffer:
    def __init__(self, buffer_size: int, obs_dim: int, action_dim: int):
        self.observations = torch.zeros((buffer_size, obs_dim))
        self.actions = torch.zeros((buffer_size, action_dim))
        self.rewards = torch.zeros((buffer_size, 1))
        self.advantages = torch.zeros((buffer_size, 1))

    def compute_returns_and_advantage(self, last_values: torch.Tensor, gamma: float = 0.99):
        # GAE-Lambda vector calculation
        ...`,
    interviewQuestions: [
      {
        question: 'What is the role of Dynamic Action Masking inside the Rollout Buffer?',
        answer: 'The rollout buffer stores boolean validity masks mask_t of shape (batch, action_dim). During PPO loss computation, invalid actions (such as moving into static shelves) are set to -inf prior to softmax, ensuring zero probability is assigned to non-traversable moves.',
      },
    ],
  },
  {
    id: 'spatial-cnn-tensor',
    name: '5-Channel 2D Spatial Grid Tensor',
    category: 'RL',
    definition: 'A 4D PyTorch tensor of shape (B, 5, H, W) encoding global warehouse entity positions across 5 binary spatial channels.',
    whySelected: 'Feeds global spatial state context to the 2D Spatial CNN Centralized Critic in Spatial MAPPO with O(1) parameter complexity.',
    filePath: 'marl/algorithms/mappo/spatial_critic.py',
    classesUsed: ['SpatialCentralizedCritic', 'SpatialMAPPOPolicy'],
    functionsUsed: ['forward()', 'evaluate_actions()'],
    timeComplexity: {
      access: 'O(1) tensor slice',
      search: 'N/A',
      insertion: 'O(1)',
      deletion: 'N/A',
    },
    spaceComplexity: 'O(B × 5 × H × W)',
    advantages: [
      'O(1) neural parameter complexity regardless of fleet size.',
      'Preserves exact 2D spatial grid topology for convolution layers.',
    ],
    limitations: [
      'High GPU VRAM consumption for large grid dimensions (200x200).',
    ],
    alternatives: ['Concatenated 1D Vector State', 'Graph Neural Network (GNN) Adjacency Matrix'],
    codeSnippet: `class SpatialCentralizedCritic(nn.Module):
    def __init__(self, input_channels: int = 5):
        super().__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 1)
        )`,
    interviewQuestions: [
      {
        question: 'Why does Spatial MAPPO use a 5-Channel 2D Grid representation instead of a concatenated 1D vector?',
        answer: 'A concatenated 1D vector scales linearly with the number of agents N, causing the network input size and parameter count to blow up. A 5-channel 2D spatial tensor represents any fleet size in fixed O(1) space and allows 2D ConvNet layers to exploit spatial locality.',
      },
    ],
  },
  {
    id: 'zustand-store',
    name: 'Zustand Centralized Reactive Store',
    category: 'Frontend',
    definition: 'A centralized reactive state management store handling playback state, speed, robot fleet entities, and live telemetry.',
    whySelected: 'Provides high-performance state management with zero boilerplate and fine-grained subscriber re-renders.',
    filePath: 'frontend/src/store/useSimulationStore.ts',
    classesUsed: ['SimulationStoreState', 'RobotEntity'],
    functionsUsed: ['syncStateFromBackend()', 'stepSimulation()', 'toggleDebugOverlay()'],
    timeComplexity: {
      access: 'O(1)',
      search: 'O(N)',
      insertion: 'O(1)',
      deletion: 'O(1)',
    },
    spaceComplexity: 'O(N + M) entities',
    advantages: [
      'Decouples React component rendering from state logic.',
      'Fine-grained state selector subscriptions avoid unnecessary re-renders.',
    ],
    limitations: [
      'Single global state store requires clean slicing.',
    ],
    alternatives: ['Redux Toolkit', 'React Context API', 'MobX'],
    codeSnippet: `export const useSimulationStore = create<SimulationStoreState>((set, get) => ({
  algorithm: 'Spatial MAPPO',
  isRunning: false,
  showDebugOverlay: true,
  robots: [],

  toggleDebugOverlay: () => set((state) => ({ showDebugOverlay: !state.showDebugOverlay })),
  syncStateFromBackend: (apiState) => {
    const mappedRobots = apiState.robots.map((r, i) => ({
      id: r.id,
      x: r.position[0],
      y: r.position[1],
      state: r.state,
      color: colors[i % colors.length]
    }));
    set({ robots: mappedRobots });
  }
}));`,
    interviewQuestions: [
      {
        question: 'Why use Zustand over React Context for rendering high-frequency simulation steps?',
        answer: 'React Context triggers a re-render of all consumer components whenever any part of the context value changes. Zustand uses selector subscriptions (e.g. useSimulationStore(state => state.showDebugOverlay)) to ensure components re-render ONLY when their specific slice of state changes.',
      },
    ],
  },
];

export const DataStructuresPage: React.FC = () => {
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState<string>('All');
  const [expandedItems, setExpandedItems] = useState<Record<string, boolean>>({});
  const [activeTab, setActiveTab] = useState<'catalog' | 'interview'>('catalog');

  const toggleExpand = (id: string) => {
    setExpandedItems((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  const filteredItems = useMemo(() => {
    return DATA_STRUCTURES_CATALOG.filter((item) => {
      const matchesCategory = selectedCategory === 'All' || item.category === selectedCategory;
      const q = searchQuery.toLowerCase();
      const matchesSearch =
        item.name.toLowerCase().includes(q) ||
        item.definition.toLowerCase().includes(q) ||
        item.filePath.toLowerCase().includes(q) ||
        item.classesUsed.some((c) => c.toLowerCase().includes(q)) ||
        item.functionsUsed.some((f) => f.toLowerCase().includes(q)) ||
        item.alternatives.some((a) => a.toLowerCase().includes(q));

      return matchesCategory && matchesSearch;
    });
  }, [searchQuery, selectedCategory]);

  return (
    <div className="py-8 space-y-8">
      <Container>
        {/* Header Title Section */}
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-surface-border pb-6">
          <div>
            <div className="flex items-center space-x-2 text-accent font-mono text-xs uppercase tracking-widest mb-1">
              <Database className="w-4 h-4" />
              <span>Architectural Analysis & Inspection</span>
            </div>
            <h1 className="text-3xl font-extrabold text-white tracking-tight">
              Data Structures & Algorithms Used
            </h1>
            <p className="text-sm text-slate-400 max-w-2xl mt-1">
              Comprehensive repository inspection detailing memory models, time/space complexities, domain selections, and implementation code snippets across simulation, planning, MARL, and web layers.
            </p>
          </div>

          <div className="flex items-center space-x-2">
            <button
              onClick={() => setActiveTab('catalog')}
              className={`px-4 py-2 rounded-lg text-xs font-bold font-mono transition-all flex items-center space-x-2 ${
                activeTab === 'catalog'
                  ? 'bg-accent text-white shadow-glow'
                  : 'bg-surface-light text-slate-400 border border-surface-border hover:text-white'
              }`}
            >
              <Layers className="w-4 h-4" />
              <span>Catalog View</span>
            </button>

            <button
              onClick={() => setActiveTab('interview')}
              className={`px-4 py-2 rounded-lg text-xs font-bold font-mono transition-all flex items-center space-x-2 ${
                activeTab === 'interview'
                  ? 'bg-emerald-500 text-white shadow-glow'
                  : 'bg-surface-light text-slate-400 border border-surface-border hover:text-white'
              }`}
            >
              <HelpCircle className="w-4 h-4" />
              <span>Interview Mode</span>
            </button>
          </div>
        </div>

        {/* Quick Domain Matrix Overview */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mt-6">
          <div className="glass-panel p-4 rounded-xl border border-surface-border space-y-2">
            <div className="flex items-center space-x-2 text-cyan-400 font-mono text-xs font-bold">
              <Cpu className="w-4 h-4" />
              <span>Path Planning DSA</span>
            </div>
            <p className="text-[11px] text-slate-300">
              Min-Heap Priority Queue, Space-Time Reservation Table, Closed Sets, Parent Maps.
            </p>
            <div className="text-[10px] font-mono text-slate-400">Target: Space-Time A* Search</div>
          </div>

          <div className="glass-panel p-4 rounded-xl border border-surface-border space-y-2">
            <div className="flex items-center space-x-2 text-indigo-400 font-mono text-xs font-bold">
              <Activity className="w-4 h-4" />
              <span>MARL & RL DSA</span>
            </div>
            <p className="text-[11px] text-slate-300">
              Rollout Tensor Buffers, 5-Channel 2D Grid Tensors, GAE Arrays, Bounded Deques.
            </p>
            <div className="text-[10px] font-mono text-slate-400">Target: IPPO & Spatial MAPPO</div>
          </div>

          <div className="glass-panel p-4 rounded-xl border border-surface-border space-y-2">
            <div className="flex items-center space-x-2 text-amber-400 font-mono text-xs font-bold">
              <Play className="w-4 h-4" />
              <span>Simulator DSA</span>
            </div>
            <p className="text-[11px] text-slate-300">
              2D Cell Array Matrix, Position Dataclass, FIFO Task Deque, Entity Hash Maps.
            </p>
            <div className="text-[10px] font-mono text-slate-400">Target: Warehouse Environment</div>
          </div>

          <div className="glass-panel p-4 rounded-xl border border-surface-border space-y-2">
            <div className="flex items-center space-x-2 text-emerald-400 font-mono text-xs font-bold">
              <Layers className="w-4 h-4" />
              <span>Web Platform DSA</span>
            </div>
            <p className="text-[11px] text-slate-300">
              Zustand Store Slices, Pydantic Schema Models, SVG Path Coordinate Arrays.
            </p>
            <div className="text-[10px] font-mono text-slate-400">Target: Digital Twin Web UI</div>
          </div>
        </div>

        {/* Search & Category Filter Controls */}
        <div className="mt-8 glass-panel p-4 rounded-xl border border-surface-border space-y-4">
          <div className="flex flex-col md:flex-row gap-4 items-center justify-between">
            {/* Search Input */}
            <div className="relative w-full md:w-96">
              <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-slate-400" />
              <input
                type="text"
                placeholder='Search "queue", "heap", "matrix", "tensor", "hash"...'
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full bg-surface-dark border border-surface-border rounded-lg pl-9 pr-4 py-2 text-xs text-white placeholder-slate-500 focus:outline-none focus:border-accent transition-all font-mono"
              />
            </div>

            {/* Category Filter Chips */}
            <div className="flex items-center space-x-2 overflow-x-auto w-full md:w-auto pb-1 md:pb-0">
              <span className="text-xs font-mono text-slate-400 flex items-center mr-1">
                <Filter className="w-3.5 h-3.5 mr-1" /> Filter:
              </span>
              {['All', 'Simulation', 'Planning', 'RL', 'Backend', 'Frontend'].map((cat) => (
                <button
                  key={cat}
                  onClick={() => setSelectedCategory(cat)}
                  className={`px-3 py-1 rounded-md text-xs font-mono transition-all whitespace-nowrap ${
                    selectedCategory === cat
                      ? 'bg-accent text-white font-bold shadow-glow'
                      : 'bg-surface-light text-slate-400 hover:text-white border border-surface-border'
                  }`}
                >
                  {cat}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* CATALOG VIEW TABS */}
        {activeTab === 'catalog' && (
          <div className="mt-6 space-y-6">
            <div className="flex justify-between items-center text-xs font-mono text-slate-400">
              <span>Showing {filteredItems.length} of {DATA_STRUCTURES_CATALOG.length} Data Structures</span>
              <span>All file paths are repository-relative</span>
            </div>

            {filteredItems.map((item) => {
              const isExpanded = expandedItems[item.id] || false;
              return (
                <div
                  key={item.id}
                  className="glass-panel rounded-xl border border-surface-border overflow-hidden transition-all duration-200 hover:border-slate-600"
                >
                  {/* Card Header */}
                  <div className="p-5 flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-surface-border/60 bg-surface-dark/40">
                    <div className="space-y-1">
                      <div className="flex items-center space-x-3">
                        <h3 className="text-lg font-bold text-white font-mono">{item.name}</h3>
                        <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold uppercase bg-accent/20 text-accent border border-accent/40">
                          {item.category}
                        </span>
                      </div>
                      <p className="text-xs text-slate-300 max-w-3xl">{item.definition}</p>
                    </div>

                    <div className="flex items-center space-x-3 shrink-0">
                      <div className="px-3 py-1 rounded-md bg-slate-900 border border-slate-700 text-cyan-300 font-mono text-xs font-bold">
                        Access: {item.timeComplexity.access}
                      </div>
                      <button
                        onClick={() => toggleExpand(item.id)}
                        className="p-2 rounded-lg bg-surface-light border border-surface-border text-slate-300 hover:text-white transition-all flex items-center space-x-1 text-xs font-mono"
                      >
                        <span>{isExpanded ? 'Hide Details' : 'Inspect Code & Memory'}</span>
                        {isExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                      </button>
                    </div>
                  </div>

                  {/* Card Body */}
                  <div className="p-5 space-y-4 text-xs">
                    {/* Implementation Metadata Section */}
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 bg-background/50 p-4 rounded-lg border border-surface-border font-mono">
                      <div>
                        <span className="text-slate-500 block uppercase text-[10px] tracking-wider">Repository File</span>
                        <a
                          href={`file:///${item.filePath}`}
                          className="text-accent hover:underline font-bold flex items-center space-x-1 mt-0.5"
                        >
                          <FileCode className="w-3.5 h-3.5" />
                          <span>{item.filePath}</span>
                        </a>
                      </div>

                      <div>
                        <span className="text-slate-500 block uppercase text-[10px] tracking-wider">Classes Using It</span>
                        <div className="text-slate-200 mt-0.5">{item.classesUsed.join(', ')}</div>
                      </div>

                      <div>
                        <span className="text-slate-500 block uppercase text-[10px] tracking-wider">Functions Using It</span>
                        <div className="text-indigo-300 mt-0.5">{item.functionsUsed.join(', ')}</div>
                      </div>
                    </div>

                    {/* Rationale & Selection */}
                    <div>
                      <span className="font-bold text-white font-mono uppercase tracking-wider block mb-1 text-[11px]">
                        Why Selected:
                      </span>
                      <p className="text-slate-300 bg-surface-dark p-3 rounded-lg border border-surface-border">
                        {item.whySelected}
                      </p>
                    </div>

                    {/* Time & Space Complexity Grid */}
                    <div className="grid grid-cols-2 md:grid-cols-5 gap-2 font-mono text-[11px]">
                      <div className="p-2.5 rounded bg-surface-dark border border-surface-border">
                        <span className="text-slate-500 block text-[9px] uppercase">Access Time</span>
                        <span className="text-cyan-400 font-bold">{item.timeComplexity.access}</span>
                      </div>
                      <div className="p-2.5 rounded bg-surface-dark border border-surface-border">
                        <span className="text-slate-500 block text-[9px] uppercase">Search Time</span>
                        <span className="text-amber-400 font-bold">{item.timeComplexity.search}</span>
                      </div>
                      <div className="p-2.5 rounded bg-surface-dark border border-surface-border">
                        <span className="text-slate-500 block text-[9px] uppercase">Insert Time</span>
                        <span className="text-emerald-400 font-bold">{item.timeComplexity.insertion}</span>
                      </div>
                      <div className="p-2.5 rounded bg-surface-dark border border-surface-border">
                        <span className="text-slate-500 block text-[9px] uppercase">Delete Time</span>
                        <span className="text-rose-400 font-bold">{item.timeComplexity.deletion}</span>
                      </div>
                      <div className="p-2.5 rounded bg-surface-dark border border-surface-border col-span-2 md:col-span-1">
                        <span className="text-slate-500 block text-[9px] uppercase">Space Complexity</span>
                        <span className="text-indigo-400 font-bold">{item.spaceComplexity}</span>
                      </div>
                    </div>

                    {/* Advantages & Limitations */}
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-2">
                      <div className="space-y-1">
                        <span className="font-bold text-emerald-400 font-mono text-[11px] uppercase tracking-wider block">
                          Advantages
                        </span>
                        <ul className="space-y-1 text-slate-300">
                          {item.advantages.map((adv, idx) => (
                            <li key={idx} className="flex items-start">
                              <CheckCircle2 className="w-3 h-3 text-emerald-400 mr-1.5 shrink-0 mt-0.5" />
                              <span>{adv}</span>
                            </li>
                          ))}
                        </ul>
                      </div>

                      <div className="space-y-1">
                        <span className="font-bold text-rose-400 font-mono text-[11px] uppercase tracking-wider block">
                          Limitations
                        </span>
                        <ul className="space-y-1 text-slate-300">
                          {item.limitations.map((lim, idx) => (
                            <li key={idx} className="flex items-start">
                              <span className="text-rose-400 mr-1.5 font-bold">•</span>
                              <span>{lim}</span>
                            </li>
                          ))}
                        </ul>
                      </div>

                      <div className="space-y-1">
                        <span className="font-bold text-amber-400 font-mono text-[11px] uppercase tracking-wider block">
                          Possible Alternatives
                        </span>
                        <div className="flex flex-wrap gap-1.5 pt-1">
                          {item.alternatives.map((alt, idx) => (
                            <span key={idx} className="px-2 py-0.5 rounded bg-slate-800 border border-slate-700 text-slate-300 font-mono text-[10px]">
                              {alt}
                            </span>
                          ))}
                        </div>
                      </div>
                    </div>

                    {/* EXPANDED CODE DRAWER */}
                    {isExpanded && (
                      <div className="pt-4 space-y-4 border-t border-surface-border/60">
                        {item.memoryLayoutDiagram && (
                          <div>
                            <span className="font-bold text-cyan-300 font-mono text-[11px] uppercase tracking-wider block mb-1">
                              Memory Layout & Data Flow Illustration:
                            </span>
                            <pre className="p-3 rounded-lg bg-slate-950 border border-slate-800 text-cyan-300 font-mono text-[11px] leading-relaxed overflow-x-auto">
                              {item.memoryLayoutDiagram}
                            </pre>
                          </div>
                        )}

                        <div>
                          <div className="flex items-center justify-between mb-1 font-mono text-[11px]">
                            <span className="font-bold text-indigo-300 uppercase tracking-wider">
                              Implementation Code Snippet ({item.filePath})
                            </span>
                            <span className="text-slate-500">Repository-Relative Path</span>
                          </div>
                          <pre className="p-4 rounded-xl bg-slate-950 border border-slate-800 text-emerald-400 font-mono text-[11px] leading-relaxed overflow-x-auto">
                            <code>{item.codeSnippet}</code>
                          </pre>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        )}

        {/* INTERVIEW MODE TAB */}
        {activeTab === 'interview' && (
          <div className="mt-6 space-y-6">
            <div className="glass-panel p-5 rounded-xl border border-emerald-500/30 bg-emerald-500/5 space-y-2">
              <div className="flex items-center space-x-2 text-emerald-400 font-bold font-mono text-sm">
                <HelpCircle className="w-5 h-5" />
                <span>Technical Interview Question Bank</span>
              </div>
              <p className="text-xs text-slate-300">
                Curated technical interview questions focusing on data structure selection trade-offs, space-time complexities, and memory architectures across robotics, pathfinding, and MARL systems.
              </p>
            </div>

            {DATA_STRUCTURES_CATALOG.flatMap((item) =>
              item.interviewQuestions.map((q, idx) => (
                <div key={`${item.id}_q_${idx}`} className="glass-panel p-5 rounded-xl border border-surface-border space-y-3">
                  <div className="flex items-start space-x-3">
                    <span className="px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-300 font-mono text-[10px] font-bold uppercase shrink-0 mt-0.5">
                      {item.name}
                    </span>
                    <h4 className="text-sm font-bold text-white font-sans">{q.question}</h4>
                  </div>
                  <div className="p-4 rounded-lg bg-surface-dark border border-surface-border text-xs text-slate-300 leading-relaxed font-sans">
                    <span className="font-bold text-accent font-mono block mb-1 uppercase text-[10px]">Model Answer:</span>
                    {q.answer}
                  </div>
                </div>
              ))
            )}
          </div>
        )}
      </Container>
    </div>
  );
};
