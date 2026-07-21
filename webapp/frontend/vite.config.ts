import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  build: {
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes('/node_modules/echarts')) return 'charts'
          if (
            id.includes('/node_modules/antd/') ||
            id.includes('/node_modules/@ant-design/') ||
            id.includes('/node_modules/@rc-component/')
          ) {
            return 'antd'
          }
          if (id.includes('/node_modules/')) return 'vendor'
        },
      },
    },
  },
  server: {
    port: 5173,
    fs: {
      strict: false,
      allow: ['.'],
    },
    proxy: {
      '/api': 'http://127.0.0.1:8000',
    },
  },
})
