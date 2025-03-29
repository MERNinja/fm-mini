import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

// https://vitejs.dev/config/
export default defineConfig({
    plugins: [react()],
    server: {
        port: 3000,
        proxy: {
            '/socket.io': {
                target: 'http://localhost:8080',
                ws: true,
            },
            '/api': {
                target: 'http://localhost:8080',
            }
        }
    },
    resolve: {
        alias: {
            '@': path.resolve(__dirname, './app')
        }
    },
    build: {
        outDir: 'dist',
        emptyOutDir: true,
    }
}); 