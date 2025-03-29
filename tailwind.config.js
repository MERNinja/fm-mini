/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./app/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            colors: {
                primary: {
                    light: '#3498db',
                    dark: '#0f3460'
                },
                secondary: {
                    light: '#2c3e50',
                    dark: '#16213e'
                },
                success: {
                    light: '#2ecc71',
                    dark: '#25a25a'
                },
                warning: {
                    light: '#f39c12',
                    dark: '#d68910'
                },
                error: {
                    light: '#e74c3c',
                    dark: '#c0392b'
                }
            }
        },
    },
    darkMode: 'class',
    plugins: [],
} 