// vite.config.js
import { defineConfig } from 'vite';
import { resolve } from 'path';

export default defineConfig({
  server: {
    port: 3001  // Changed from 3000 to avoid port conflicts
  },
  build: {
    rollupOptions: {
      input: {
        main: resolve(__dirname, 'public/index.html')
      }
    }
  },
  // Specify the entry point for development mode
  root: './public',
  publicDir: './public',
  resolve: {
    alias: {
      '/': './public'
    }
  }
});
