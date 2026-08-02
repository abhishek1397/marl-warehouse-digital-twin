import React from 'react';

interface PageHeaderProps {
  title: string;
  subtitle?: string;
  badge?: React.ReactNode;
  action?: React.ReactNode;
}

export const PageHeader: React.FC<PageHeaderProps> = ({
  title,
  subtitle,
  badge,
  action,
}) => {
  return (
    <div className="flex flex-col md:flex-row md:items-center md:justify-between mb-8 pb-4 border-b border-surface-border">
      <div>
        <div className="flex items-center space-x-3">
          <h1 className="text-2xl sm:text-3xl font-extrabold text-white tracking-tight font-sans">
            {title}
          </h1>
          {badge}
        </div>
        {subtitle && (
          <p className="mt-1 text-sm text-slate-400 font-normal max-w-3xl">
            {subtitle}
          </p>
        )}
      </div>
      {action && <div className="mt-4 md:mt-0 flex items-center space-x-3">{action}</div>}
    </div>
  );
};
