import {
  ApartmentOutlined,
  AppstoreOutlined,
  BranchesOutlined,
  DatabaseOutlined,
  DownOutlined,
  ExperimentOutlined,
  HomeOutlined,
  NodeIndexOutlined,
} from '@ant-design/icons'
import { useQuery } from '@tanstack/react-query'
import { Button, Dropdown, Layout, Space, Tooltip, Typography } from 'antd'
import { lazy, Suspense, useEffect } from 'react'
import { Navigate, Route, Routes, useLocation, useNavigate } from 'react-router-dom'

import { api } from './api/client'
import { QueueIndicator } from './components/QueueIndicator'
import { ScrollManager } from './components/ScrollManager'

const Compare = lazy(() => import('./pages/Compare'))
const FactorCreate = lazy(() => import('./pages/FactorCreate'))
const FactorLibrary = lazy(() => import('./pages/FactorLibrary'))
const GeneticMining = lazy(() => import('./pages/GeneticMining'))
const Home = lazy(() => import('./pages/Home'))
const MethodLibrary = lazy(() => import('./pages/MethodLibrary'))
const ModelTests = lazy(() => import('./pages/ModelTests'))
const PipelineTemplates = lazy(() => import('./pages/PipelineTemplates'))
const Queue = lazy(() => import('./pages/Queue'))
const RunDetail = lazy(() => import('./pages/RunDetail'))

const { Header, Content } = Layout

const TITLES: Array<[string, string]> = [
  ['/queue', '任务队列'],
  ['/methods', '评价模块库'],
  ['/mining', '遗传规划挖掘'],
  ['/pipelines', '流水线模板'],
  ['/compare', '多因子对比'],
  ['/models', '模型测试'],
  ['/runs', '回测结果'],
  ['/test-factors/new', '新增测试因子'],
  ['/test-factors', '测试库'],
  ['/factors', '因子库'],
]

/** 因子库与测试库自带左侧因子栏，内容区贴边铺满；其余页面走常规居中版心。 */
function isLibraryRoute(pathname: string) {
  if (pathname.startsWith('/test-factors/new')) return false
  return pathname.startsWith('/test-factors') || pathname.startsWith('/factors')
}

export default function App() {
  const location = useLocation()
  const navigate = useNavigate()
  // 行情数据时效常驻头部：陈旧数据是回测结论失真的头号来源
  const timeline = useQuery({
    queryKey: ['market-timeline'],
    queryFn: api.marketTimeline,
    staleTime: 5 * 60_000,
  })
  useEffect(() => {
    const entry = TITLES.find(([prefix]) => location.pathname.startsWith(prefix))
    document.title = entry ? `${entry[1]} · 因子评价平台` : '因子评价平台'
  }, [location.pathname])
  const isHome = location.pathname === '/'
  const pageTitle = isHome
    ? 'Factor Lab'
    : TITLES.find(([prefix]) => location.pathname.startsWith(prefix))?.[1] || '因子评价平台'
  const flush = isLibraryRoute(location.pathname)

  return (
    <Layout className="app-shell">
      <ScrollManager />
      <Header className="app-header">
        <div className="header-context">
          <Dropdown
            trigger={['click']}
            menu={{
              selectable: false,
              onClick: ({ key }) => navigate(key),
              items: [
                { key: '/', icon: <HomeOutlined />, label: '主页' },
                { type: 'divider' },
                { key: '/test-factors', icon: <ExperimentOutlined />, label: '测试库' },
                { key: '/factors', icon: <DatabaseOutlined />, label: '因子库' },
                { key: '/mining', icon: <BranchesOutlined />, label: '遗传规划挖掘' },
                { key: '/pipelines', icon: <ApartmentOutlined />, label: '流水线模板' },
                { key: '/models', icon: <NodeIndexOutlined />, label: '模型测试' },
              ],
            }}
          >
            <Button type="text" className="brand-button" aria-label="切换板块">
              <span className="brand-mark">F</span>
              <span className="brand-button-label">
                <Typography.Text className="header-page-title">{pageTitle}</Typography.Text>
                <Typography.Text className="header-page-subtitle">
                  <AppstoreOutlined /> 切换板块
                </Typography.Text>
              </span>
              <DownOutlined className="brand-button-caret" />
            </Button>
          </Dropdown>
        </div>
        <Space size={14}>
          {timeline.data && (
            <Tooltip
              title={`本地行情矩阵样本区间 ${timeline.data.start_day} ~ ${timeline.data.end_day}，共 ${timeline.data.count} 个交易日。`}
            >
              <Typography.Text type="secondary" className="data-freshness">
                行情数据截至 {timeline.data.end_day}
              </Typography.Text>
            </Tooltip>
          )}
          <Tooltip title="因子在 t 日收盘后形成信号，下一交易日开盘建仓；默认持有 1 天时，收益为 open[t+2] / open[t+1] - 1。">
            <Space className="return-convention" size={6}>
              <span className="return-convention-dot" aria-hidden="true" />
              <Typography.Text type="secondary">下一开盘至开盘收益</Typography.Text>
            </Space>
          </Tooltip>
          <QueueIndicator />
        </Space>
      </Header>
      <Content className={`app-content${flush ? ' app-content-flush' : ''}`}>
        <Suspense fallback={<div className="page-loading">页面加载中…</div>}>
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/test-factors" element={<FactorLibrary mode="test" />} />
            <Route path="/test-factors/new" element={<FactorCreate />} />
            <Route path="/test-factors/:batchId/:name" element={<FactorLibrary mode="test" />} />
            <Route path="/factors" element={<FactorLibrary mode="factor" />} />
            <Route path="/factors/new" element={<Navigate to="/test-factors/new" replace />} />
            <Route path="/factors/:batchId/:name" element={<FactorLibrary mode="factor" />} />
            <Route path="/mining" element={<GeneticMining />} />
            <Route path="/queue" element={<Queue />} />
            <Route path="/runs/:runId" element={<RunDetail />} />
            <Route path="/pipelines" element={<PipelineTemplates />} />
            <Route path="/methods" element={<MethodLibrary />} />
            <Route path="/compare" element={<Compare />} />
            <Route path="/models" element={<ModelTests />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </Suspense>
      </Content>
    </Layout>
  )
}
