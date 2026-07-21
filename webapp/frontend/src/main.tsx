import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { ConfigProvider, theme } from 'antd'
import zhCN from 'antd/locale/zh_CN'
import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import App from './App'
import { ErrorBoundary } from './ErrorBoundary'
import './styles.css'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: { staleTime: 10_000, retry: 1 },
  },
})

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <ErrorBoundary>
      <QueryClientProvider client={queryClient}>
        <ConfigProvider
          locale={zhCN}
          theme={{
            algorithm: theme.defaultAlgorithm,
            token: {
              colorPrimary: '#9a4f32',
              colorInfo: '#9a4f32',
              colorSuccess: '#008A3E',
              colorWarning: '#9b6a24',
              colorError: '#ad4545',
              colorBgBase: '#f6f5f2',
              colorBgContainer: '#fffdf9',
              colorBgElevated: '#fffdf9',
              colorText: '#2a2925',
              colorTextSecondary: '#756d61',
              colorBorder: '#e5e1d9',
              borderRadius: 12,
              fontFamily: '"SF Pro Text", "PingFang SC", "Microsoft YaHei", sans-serif',
            },
            components: {
              Layout: { siderBg: '#000000', headerBg: '#fffdf9' },
              Menu: {
                darkItemBg: '#000000',
                darkItemColor: 'rgba(255, 255, 255, 0.65)',
                darkItemHoverBg: 'rgba(255, 255, 255, 0.08)',
                darkItemHoverColor: '#ffffff',
                darkItemSelectedBg: '#ffffff',
                darkItemSelectedColor: '#000000',
                itemBorderRadius: 10,
              },
              Table: { headerBg: '#faf8f3', borderColor: '#e6e2da', headerColor: '#665f55', rowHoverBg: '#fbf4e9' },
              Button: { primaryShadow: 'none', defaultShadow: 'none' },
              Input: { activeBorderColor: '#9a4f32', hoverBorderColor: '#b27a5f' },
              Select: { optionSelectedBg: '#f5e3d2' },
            },
          }}
        >
          <BrowserRouter>
            <App />
          </BrowserRouter>
        </ConfigProvider>
      </QueryClientProvider>
    </ErrorBoundary>
  </React.StrictMode>,
)
