import React from 'react';
import { createRoot } from 'react-dom/client';
import { BrowserRouter, Routes, Route, Link } from 'react-router-dom';
import ChatPage from './pages/ChatPage';
import NodeListPage from './pages/NodeListPage';
import './styles.css';

const App = () => {
    return (
        <BrowserRouter>
            <div className="app-container">
                <header>
                    <nav>
                        <ul>
                            <li><Link to="/">Chat</Link></li>
                            <li><Link to="/nodes">Network Status</Link></li>
                        </ul>
                    </nav>
                </header>
                <main>
                    <Routes>
                        <Route path="/" element={<ChatPage />} />
                        <Route path="/nodes" element={<NodeListPage />} />
                    </Routes>
                </main>
            </div>
        </BrowserRouter>
    );
};

const root = createRoot(document.getElementById('root'));
root.render(<App />); 