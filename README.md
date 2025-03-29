# FM-MINI

A web application that allows distributed WebLLM nodes to connect and communicate. It includes a chat interface with WebLLM integration and a node monitoring page, similar to [chat.webllm.ai](https://chat.webllm.ai/).

## Features

- **Chat Interface**: Interact with a WebLLM model that runs directly in your browser
- **WebLLM Integration**: Load and run LLM models locally in the browser using WebGPU
- **Node Network**: Each browser with a loaded WebLLM model becomes a node in the network
- **Node Monitoring**: View all connected nodes, their status, and information
- **Real-time Communication**: Built with Socket.io for seamless node interaction
- **Modern UI**: Built with TailwindCSS for a responsive and customizable interface
- **Dark/Light Theme**: Supports both light and dark themes with system preference detection
- **Model Switching**: Change between different models at any time without reloading the page
- **Auto-Load Models**: Option to automatically load the last used model when revisiting the page

## Prerequisites

- Node.js (v14+)
- NPM or Yarn
- A browser that supports WebGPU (Chrome 113+ or Edge 113+ recommended)

## Quick Start

1. Clone the repository:
```
git clone <repository-url>
cd fm-mini
```

2. Install dependencies:
```
npm install
```

3. Start both frontend and backend with a single command:
```
npm run dev
```

4. Open your browser and navigate to:
```
http://localhost:3000
```

## How It Works

- **Frontend**: Built with React and Vite for fast development and optimized production builds
- **Backend**: Express.js server with Socket.io for real-time communication
- **WebLLM**: Loads and runs LLM models directly in the browser using WebGPU
- **Node Network**: When a model is loaded, the browser registers as a node in the network
- **Message Exchange**: All message exchanges are broadcast to other connected nodes
- **Styling**: TailwindCSS for utility-first styling with theme support

## Project Structure

- `app/` - Frontend React application
  - `components/` - Reusable React components (ThemeToggle, etc.)
  - `context/` - React context providers (ThemeContext)
  - `pages/` - Page components (ChatPage and NodeListPage)
  - `main.jsx` - Main entry point for React application
  - `styles.css` - Global styles and Tailwind imports
- `server.js` - Backend Socket.io server
- `vite.config.js` - Vite configuration
- `tailwind.config.js` - TailwindCSS configuration
- `postcss.config.js` - PostCSS configuration for Tailwind

## Available Models

The application supports various WebLLM models including:
- Llama-2-7B
- Mistral-7B-v0.1
- RedPajama-INCITE-7B-v0.1
- vicuna-7b-v1.5
- phi-2
- And more as they become available in WebLLM

## UI Features

### Theme Switching
- The application supports both light and dark themes
- Theme preference is auto-detected from system settings
- User theme choice is persisted in localStorage
- Toggle between themes using the button in the header

### Responsive Design
- Mobile-friendly interface that adapts to different screen sizes
- Optimized layout for both desktop and mobile browsing

### Model Persistence
- The last selected model is remembered between sessions
- Optional auto-load feature to automatically load the last used model on page refresh
- User preferences are stored in localStorage for persistence

## Development

- The frontend development server runs on port 3000 with hot module reloading
- The backend server runs on port 5000 with nodemon for automatic restarts on file changes
- When running `npm run dev`, both servers start concurrently
- API requests and WebSocket connections are proxied from the frontend to the backend

## Building for Production

1. Build the frontend:
```
npm run build
```

2. Start the production server:
```
npm start
```

3. Access the application at:
```
http://localhost:5000
```

## Troubleshooting

- If you see errors related to WebGPU support, make sure your browser supports WebGPU
- If you experience CORS issues, ensure both the frontend and backend servers are running
- WebLLM may require significant memory to load models. If you encounter memory issues:
  - Close other browser tabs and applications
  - Use a browser with more efficient memory handling
- If you encounter issues with the Socket.io connection:
  - Check that the proxying configuration in `vite.config.js` is correct
  - Verify that both development servers are running

## License

MIT