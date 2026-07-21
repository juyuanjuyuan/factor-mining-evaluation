import {
  ApartmentOutlined,
  BarChartOutlined,
  DatabaseOutlined,
  ExperimentOutlined,
  MenuFoldOutlined,
  MenuUnfoldOutlined,
  OrderedListOutlined,
  ProfileOutlined,
} from '@ant-design/icons'
import { useQuery } from '@tanstack/react-query'
import { Button, Layout, Menu, Space, Tooltip, Typography } from 'antd'
import { lazy, Suspense, useEffect, useState } from 'react'
import { Link, Navigate, Route, Routes, useLocation } from 'react-router-dom'

import { api } from './api/client'
import { ScrollManager } from './components/ScrollManager'

const Compare = lazy(() => import('./pages/Compare'))
const FactorCreate = lazy(() => import('./pages/FactorCreate'))
const FactorDetail = lazy(() => import('./pages/FactorDetail'))
const FactorLibrary = lazy(() => import('./pages/FactorLibrary'))
const MethodLibrary = lazy(() => import('./pages/MethodLibrary'))
const ModelTests = lazy(() => import('./pages/ModelTests'))
const PipelineTemplates = lazy(() => import('./pages/PipelineTemplates'))
const Queue = lazy(() => import('./pages/Queue'))
const RunDetail = lazy(() => import('./pages/RunDetail'))

const { Header, Sider, Content } = Layout

const TITLES: Array<[string, string]> = [
  ['/queue', '任务队列'],
  ['/methods', '评价模块库'],
  ['/pipelines', '流水线模板'],
  ['/compare', '多因子对比'],
  ['/models', '模型测试'],
  ['/runs', '回测结果'],
  ['/test-factors/new', '新增测试因子'],
  ['/test-factors', '测试库'],
  ['/factors', '因子库'],
]

export default function App() {
  const location = useLocation()
  const [collapsed, setCollapsed] = useState(false)
  // 行情数据时效常驻头部：陈旧数据是回测结论失真的头号来源
  const timeline = useQuery({
    queryKey: ['market-timeline'],
    queryFn: api.marketTimeline,
    staleTime: 5 * 60_000,
  })
  // 回测结果从任务队列产生，/runs 页面高亮「任务队列」保持来源一致
  const selected = location.pathname.startsWith('/queue') || location.pathname.startsWith('/runs')
    ? '/queue'
    : location.pathname.startsWith('/methods')
      ? '/methods'
    : location.pathname.startsWith('/pipelines')
      ? '/pipelines'
      : location.pathname.startsWith('/compare')
        ? '/compare'
        : location.pathname.startsWith('/models')
          ? '/models'
        : location.pathname.startsWith('/test-factors')
          ? '/test-factors'
        : '/factors'
  useEffect(() => {
    const entry = TITLES.find(([prefix]) => location.pathname.startsWith(prefix))
    document.title = entry ? `${entry[1]} · 因子评价平台` : '因子评价平台'
  }, [location.pathname])
  const pageTitle = TITLES.find(([prefix]) => location.pathname.startsWith(prefix))?.[1] || '因子评价平台'
  return (
    <Layout className="app-shell">
      <ScrollManager />
      <Sider
        width={232}
        collapsedWidth={72}
        collapsible
        trigger={null}
        breakpoint="lg"
        collapsed={collapsed}
        onCollapse={setCollapsed}
        className="app-sider"
      >
        <div className="brand">
          <div className="brand-mark">F</div>
          <div>
            <Typography.Text className="brand-title">Factor Lab</Typography.Text>
            <div className="brand-subtitle">因子评价平台</div>
          </div>
        </div>
        <Menu
          className="app-menu"
          mode="inline"
          theme="dark"
          selectedKeys={[selected]}
          items={[
            { key: '/test-factors', icon: <ExperimentOutlined />, label: <Link to="/test-factors">测试库</Link> },
            { key: '/factors', icon: <DatabaseOutlined />, label: <Link to="/factors">因子库</Link> },
            { key: '/queue', icon: <OrderedListOutlined />, label: <Link to="/queue">任务队列</Link> },
            { key: '/pipelines', icon: <ApartmentOutlined />, label: <Link to="/pipelines">流水线模板</Link> },
            { key: '/methods', icon: <ProfileOutlined />, label: <Link to="/methods">评价模块库</Link> },
            { key: '/compare', icon: <BarChartOutlined />, label: <Link to="/compare">多因子对比</Link> },
            { key: '/models', icon: <ExperimentOutlined />, label: <Link to="/models">模型测试</Link> },
          ]}
        />
      </Sider>
      <Layout>
        <Header className="app-header">
          <div className="header-context">
            <Button
              type="text"
              className="sider-toggle"
              aria-label={collapsed ? '展开导航栏' : '收起导航栏'}
              icon={collapsed ? <MenuUnfoldOutlined /> : <MenuFoldOutlined />}
              onClick={() => setCollapsed((value) => !value)}
            />
            <div>
              <Typography.Text className="header-page-title">{pageTitle}</Typography.Text>
              <Typography.Text className="header-page-subtitle">研究工作台</Typography.Text>
            </div>
          </div>
          <Space size={16}>
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
          </Space>
        </Header>
        <Content className="app-content">
          <Suspense fallback={<div className="page-loading">页面加载中…</div>}>
            <Routes>
              <Route path="/" element={<Navigate to="/test-factors" replace />} />
              <Route path="/test-factors" element={<FactorLibrary mode="test" />} />
              <Route path="/test-factors/new" element={<FactorCreate />} />
              <Route path="/test-factors/:batchId/:name" element={<FactorDetail />} />
              <Route path="/factors" element={<FactorLibrary mode="factor" />} />
              <Route path="/factors/new" element={<Navigate to="/test-factors/new" replace />} />
              <Route path="/factors/:batchId/:name" element={<FactorDetail />} />
              <Route path="/queue" element={<Queue />} />
              <Route path="/runs/:runId" element={<RunDetail />} />
              <Route path="/pipelines" element={<PipelineTemplates />} />
              <Route path="/methods" element={<MethodLibrary />} />
              <Route path="/compare" element={<Compare />} />
              <Route path="/models" element={<ModelTests />} />
              <Route path="*" element={<Navigate to="/test-factors" replace />} />
            </Routes>
          </Suspense>
        </Content>
      </Layout>
    </Layout>
  )
}
