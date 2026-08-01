import {
  ApartmentOutlined,
  ArrowRightOutlined,
  BranchesOutlined,
  DatabaseOutlined,
  ExperimentOutlined,
  NodeIndexOutlined,
} from '@ant-design/icons'
import { useQuery } from '@tanstack/react-query'
import { Typography } from 'antd'
import { Link } from 'react-router-dom'
import { api } from '../api/client'

const ACTIVE = ['queued', 'running', 'cancelling']

export default function Home() {
  const testFactors = useQuery({ queryKey: ['test-factors'], queryFn: api.testFactors })
  const factors = useQuery({ queryKey: ['factors'], queryFn: api.factors })
  const templates = useQuery({ queryKey: ['templates'], queryFn: api.templates })
  const jobs = useQuery({ queryKey: ['jobs'], queryFn: api.jobs })
  const campaigns = useQuery({ queryKey: ['genetic-campaigns'], queryFn: api.geneticCampaigns })
  const timeline = useQuery({
    queryKey: ['market-timeline'],
    queryFn: api.marketTimeline,
    staleTime: 5 * 60_000,
  })
  const activeJobs = (jobs.data || []).filter((job) => ACTIVE.includes(job.status)).length
  const runningCampaigns = (campaigns.data || []).filter((campaign) => campaign.status === 'running').length

  const sections = [
    {
      path: '/test-factors',
      icon: <ExperimentOutlined />,
      title: '测试库',
      description: '新因子在这里反复回测，确认表现后再提交到因子库。',
      stat: testFactors.data ? `${testFactors.data.length} 个测试因子` : '读取中…',
    },
    {
      path: '/factors',
      icon: <DatabaseOutlined />,
      title: '因子库',
      description: '通过相关性检验后入库的正式定义，随时可以复评。',
      stat: factors.data ? `${factors.data.length} 个已入库因子` : '读取中…',
    },
    {
      path: '/pipelines',
      icon: <ApartmentOutlined />,
      title: '流水线模板',
      description: '编排评价方法的执行顺序，公式定义在评价模块库。',
      stat: templates.data ? `${templates.data.length} 个模板` : '读取中…',
    },
    {
      path: '/models',
      icon: <NodeIndexOutlined />,
      title: '模型测试',
      description: '多因子组合的训练集与样本外测试集回测。',
      stat: '组合回测',
    },
  ]

  return (
    <div className="home">
      <header className="home-head">
        <Typography.Title level={1}>Factor Lab</Typography.Title>
        <Typography.Text type="secondary">
          因子评价平台
          {timeline.data ? ` · 行情数据截至 ${timeline.data.end_day}` : ''}
          {activeJobs ? ` · ${activeJobs} 个任务进行中` : ''}
        </Typography.Text>
      </header>

      <nav className="home-grid" aria-label="板块">
        {sections.map((section) => (
          <Link key={section.path} to={section.path} className="home-card">
            <span className="home-card-icon" aria-hidden="true">
              {section.icon}
            </span>
            <span className="home-card-title">{section.title}</span>
            <span className="home-card-desc">{section.description}</span>
            <span className="home-card-foot">
              <span className="home-card-stat">{section.stat}</span>
              <ArrowRightOutlined aria-hidden="true" />
            </span>
          </Link>
        ))}
        <Link to="/mining" className="home-card home-card-wide">
          <span className="home-card-icon" aria-hidden="true">
            <BranchesOutlined />
          </span>
          <span className="home-card-wide-body">
            <span className="home-card-title">遗传规划挖掘</span>
            <span className="home-card-desc">
              在训练区间搜索因子表达式，通过测试筛选的候选自动送入因子库检验。独立进程，不占用评价队列。
            </span>
          </span>
          <span className="home-card-foot">
            <span className="home-card-stat">
              {campaigns.data
                ? runningCampaigns
                  ? `${runningCampaigns} 个任务运行中`
                  : `${campaigns.data.length} 个历史任务`
                : '读取中…'}
            </span>
            <ArrowRightOutlined aria-hidden="true" />
          </span>
        </Link>
      </nav>

      <div className="home-links">
        <Link to="/queue">任务队列</Link>
        <Link to="/methods">评价模块库</Link>
        <Link to="/test-factors/new">新增测试因子</Link>
      </div>
    </div>
  )
}
