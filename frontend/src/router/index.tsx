import React from 'react';
import { Routes, Route } from 'react-router-dom';
import { HomePage } from '../pages/HomePage';
import { SimulationPage } from '../pages/SimulationPage';
import { AlgorithmsPage } from '../pages/AlgorithmsPage';
import { ExperimentsPage } from '../pages/ExperimentsPage';
import { ResearchPage } from '../pages/ResearchPage';
import { DocumentationPage } from '../pages/DocumentationPage';
import { AboutPage } from '../pages/AboutPage';

export const AppRouter: React.FC = () => {
  return (
    <Routes>
      <Route path="/" element={<HomePage />} />
      <Route path="/simulation" element={<SimulationPage />} />
      <Route path="/algorithms" element={<AlgorithmsPage />} />
      <Route path="/experiments" element={<ExperimentsPage />} />
      <Route path="/research" element={<ResearchPage />} />
      <Route path="/docs" element={<DocumentationPage />} />
      <Route path="/about" element={<AboutPage />} />
    </Routes>
  );
};
