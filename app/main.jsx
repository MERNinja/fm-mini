import React from 'react';
import { createRoot } from 'react-dom/client';
import { BrowserRouter, Routes, Route, Link } from 'react-router-dom';
import ChatPage from './pages/ChatPage';
import NodeListPage from './pages/NodeListPage';
import ThemeToggle from './components/ThemeToggle';
import { ThemeProvider } from './context/ThemeContext';
import './styles.css';

const App = () => {
  return (
    <ThemeProvider>
      <BrowserRouter>
        <div className="flex flex-col min-h-screen bg-gray-50 text-gray-900 dark:bg-gray-900 dark:text-gray-100">
          {/* <header className="bg-secondary-light dark:bg-secondary-dark text-white py-4">
            <div className="container mx-auto px-4 flex justify-between items-center">
              <nav>
                <ul className="flex space-x-6">
                  <li>
                    <Link
                      to="/"
                      className="text-white hover:text-primary-light dark:hover:text-blue-400 font-medium text-lg"
                    >
                      Chat
                    </Link>
                  </li>
                  <li>
                    <Link
                      to="/nodes"
                      className="text-white hover:text-primary-light dark:hover:text-blue-400 font-medium text-lg"
                    >
                      Network Status
                    </Link>
                  </li>
                </ul>
              </nav>
              <ThemeToggle />
            </div>
          </header> */}
          <main className="flex-1 container mx-auto px-4 py-8 mt-8">
            <Routes>
              <Route path="/" element={<ChatPage />} />
              <Route path="/nodes" element={<NodeListPage />} />
            </Routes>
          </main>
        </div>
      </BrowserRouter>
    </ThemeProvider>
  );
};

const root = createRoot(document.getElementById('root'));
root.render(<App />);
