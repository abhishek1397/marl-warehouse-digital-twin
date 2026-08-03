import React, { useRef, useEffect, useState } from 'react';
import { useSimulationStore } from '../store/useSimulationStore';
import { GridCell } from './GridCell';
import { RobotSprite } from './RobotSprite';
import { ShelfSprite } from './ShelfSprite';
import { PackageSprite } from './PackageSprite';
import { ChargingStationSprite } from './ChargingStationSprite';
import { ObstacleSprite } from './ObstacleSprite';

export const WarehouseGrid: React.FC = () => {
  const containerRef = useRef<HTMLDivElement>(null);
  const { gridSize, robots, gridEntities } = useSimulationStore();
  const [cellSize, setCellSize] = useState<number>(30);

  useEffect(() => {
    const updateSize = () => {
      if (containerRef.current) {
        const width = containerRef.current.clientWidth - 32;
        const computedSize = Math.floor(width / gridSize);
        setCellSize(Math.max(20, Math.min(45, computedSize)));
      }
    };

    updateSize();
    window.addEventListener('resize', updateSize);
    return () => window.removeEventListener('resize', updateSize);
  }, [gridSize]);

  const cells = [];
  for (let y = 0; y < gridSize; y++) {
    for (let x = 0; x < gridSize; x++) {
      cells.push({ x, y });
    }
  }

  return (
    <div ref={containerRef} className="w-full flex flex-col items-center justify-center p-4 bg-surface-dark/90 rounded-xl border border-surface-border glass-panel relative overflow-hidden">
      <div className="absolute top-2 right-3 font-mono text-[10px] text-slate-500">
        Grid Size: {gridSize} x {gridSize} | Scale: {cellSize}px/cell
      </div>

      <div
        style={{
          width: gridSize * cellSize,
          height: gridSize * cellSize,
          display: 'grid',
          gridTemplateColumns: `repeat(${gridSize}, ${cellSize}px)`,
          gridTemplateRows: `repeat(${gridSize}, ${cellSize}px)`,
          position: 'relative',
        }}
        className="bg-background rounded-lg border border-surface-border shadow-2xl overflow-hidden"
      >
        {/* Render Background Grid Cells */}
        {cells.map((cell) => (
          <GridCell key={`cell_${cell.x}_${cell.y}`} x={cell.x} y={cell.y} cellSize={cellSize} />
        ))}

        {/* Render Planned Path Lines in Unique Robot Colors */}
        <svg
          className="absolute inset-0 w-full h-full pointer-events-none z-10"
          style={{ width: gridSize * cellSize, height: gridSize * cellSize }}
        >
          {robots.map((robot) => {
            if (!robot.plannedPath || robot.plannedPath.length < 2) return null;
            const pointsStr = robot.plannedPath
              .map(([ptX, ptY]) => `${ptX * cellSize + cellSize / 2},${ptY * cellSize + cellSize / 2}`)
              .join(' ');
            return (
              <polyline
                key={`path_${robot.id}`}
                points={pointsStr}
                fill="none"
                stroke={robot.color}
                strokeWidth="2.5"
                strokeDasharray="5,4"
                strokeLinecap="round"
                opacity="0.85"
              />
            );
          })}
        </svg>

        {/* Render Static/Item Entities */}
        {gridEntities.map((entity) => {
          if (entity.type === 'shelf') return <ShelfSprite key={entity.id} entity={entity} cellSize={cellSize} />;
          if (entity.type === 'package') return <PackageSprite key={entity.id} entity={entity} cellSize={cellSize} />;
          if (entity.type === 'charging_station') return <ChargingStationSprite key={entity.id} entity={entity} cellSize={cellSize} />;
          if (entity.type === 'obstacle') return <ObstacleSprite key={entity.id} entity={entity} cellSize={cellSize} />;
          return null;
        })}

        {/* Render Robot Sprites with Smooth Framer Motion Translations */}
        {robots.map((robot) => (
          <RobotSprite key={robot.id} robot={robot} cellSize={cellSize} />
        ))}
      </div>
    </div>
  );
};
