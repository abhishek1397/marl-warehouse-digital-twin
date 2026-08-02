import React from 'react';

interface GridCellProps {
  x: number;
  y: number;
  cellSize: number;
}

export const GridCell: React.FC<GridCellProps> = ({ x, y, cellSize }) => {
  return (
    <div
      style={{
        width: cellSize,
        height: cellSize,
      }}
      className="border border-surface-border/40 hover:border-accent/30 transition-colors flex items-end justify-end p-0.5"
    >
      <span className="text-[8px] font-mono text-slate-700 pointer-events-none select-none">
        {x},{y}
      </span>
    </div>
  );
};
