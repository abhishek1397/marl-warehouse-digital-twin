import React from 'react';
import { BrowserRouter as Router } from 'react-router-dom';
import { Navbar } from './components/Navbar';
import { Footer } from './components/Footer';
import { AppRouter } from './router';

export const App: React.FC = () => {
  return (
    <Router>
      <div className="flex flex-col min-h-screen bg-background text-slate-100 font-sans">
        <Navbar />
        <main className="flex-grow">
          <AppRouter />
        </main>
        <Footer />
      </div>
    </Router>
  );
};

export default App;
