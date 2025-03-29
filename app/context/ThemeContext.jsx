import React, { createContext, useState, useEffect, useContext } from 'react';

// Create a context for theme
const ThemeContext = createContext();

// Hook to use the theme context
export const useTheme = () => useContext(ThemeContext);

// Theme provider component
export const ThemeProvider = ({ children }) => {
  // Check if user has a preference in localStorage or prefers dark color scheme
  const getInitialTheme = () => {
    if (typeof window !== 'undefined' && window.localStorage) {
      const storedPrefs = window.localStorage.getItem('theme');
      if (typeof storedPrefs === 'string') {
        return storedPrefs;
      }

      const userMedia = window.matchMedia('(prefers-color-scheme: dark)');
      if (userMedia.matches) {
        return 'dark';
      }
    }

    return 'dark'; // Default theme
  };

  const [theme, setTheme] = useState(getInitialTheme);

  // Toggle theme function
  const toggleTheme = () => {
    setTheme(theme === 'dark' ? 'light' : 'dark');
  };

  // Apply theme to document and save to localStorage when it changes
  useEffect(() => {
    const root = window.document.documentElement;

    // Remove the old theme class
    root.classList.remove(theme === 'dark' ? 'light' : 'dark');
    // Add the new theme class
    root.classList.add(theme);

    // Save to localStorage
    localStorage.setItem('theme', theme);
  }, [theme]);

  return (
    <ThemeContext.Provider value={{ theme, toggleTheme }}>
      {children}
    </ThemeContext.Provider>
  );
};

export default ThemeContext;
