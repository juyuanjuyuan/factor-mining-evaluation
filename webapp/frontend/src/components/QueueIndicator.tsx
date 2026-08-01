import { LoadingOutlined, ThunderboltOutlined } from '@ant-design/icons'
import { useQuery } from '@tanstack/react-query'
import { Badge, Button, Empty, Popover, Progress, Tooltip, Typography } from 'antd'
import { useState } from 'react'
import { Link } from 'react-router-dom'
import { api, Job } from '../api/client'
import { shortTime } from '../lib/metrics'

const ACTIVE = ['queued', 'running', 'cancelling']

/**
 * 头部常驻的队列入口：只呈现正在跑和排队中的任务。
 * 已完成的任务不进这里，避免头部变成第二个队列页；完整历史仍在 /queue。
 */
export function QueueIndicator() {
  const [open, setOpen] = useState(false)
  // 有活跃任务时 1.8 秒刷新；空闲时降到 12 秒，与队列页保持同一节奏
  const jobs = useQuery({
    queryKey: ['jobs'],
    queryFn: api.jobs,
    refetchInterval: (query) =>
      (query.state.data || []).some((job) => ACTIVE.includes(job.status)) ? 1800 : 12000,
  })
  const active = (jobs.data || []).filter((job) => ACTIVE.includes(job.status))
  const running = active.filter((job) => job.status !== 'queued').length
  const queued = active.length - running
  const remainingRuns = active.reduce((total, job) => total + Math.max(job.total_runs - job.finished_runs, 0), 0)

  const content = (
    <div className="queue-pop">
      <div className="queue-pop-head">
        <Typography.Text strong>任务队列</Typography.Text>
        <Typography.Text type="secondary">
          {active.length ? `执行中 ${running} · 等待中 ${queued} · 待完成 ${remainingRuns} 个运行` : '单 worker 串行执行'}
        </Typography.Text>
      </div>
      {active.length ? (
        <div className="queue-pop-list">
          {active.map((job) => (
            <JobRow key={job.id} job={job} onNavigate={() => setOpen(false)} />
          ))}
        </div>
      ) : (
        <Empty
          image={Empty.PRESENTED_IMAGE_SIMPLE}
          description="暂无进行中的任务"
          className="queue-pop-empty"
        />
      )}
      <div className="queue-pop-foot">
        <Link to="/queue" onClick={() => setOpen(false)}>
          <Button type="link" size="small">
            查看全部任务与历史
          </Button>
        </Link>
      </div>
    </div>
  )

  return (
    <Popover
      open={open}
      onOpenChange={setOpen}
      trigger="click"
      placement="bottomRight"
      arrow={false}
      content={content}
      overlayClassName="queue-pop-overlay"
    >
      <Tooltip title={active.length ? `${active.length} 个任务进行中` : '任务队列'} placement="bottom">
        <Badge count={active.length} size="small" offset={[-2, 3]} color="#9a4f32">
          <Button
            type="text"
            className="queue-indicator"
            aria-label={active.length ? `任务队列，${active.length} 个进行中` : '任务队列'}
            icon={running ? <LoadingOutlined /> : <ThunderboltOutlined />}
          />
        </Badge>
      </Tooltip>
    </Popover>
  )
}

function JobRow({ job, onNavigate }: { job: Job; onNavigate: () => void }) {
  const percent = job.total_runs ? Math.round((job.finished_runs / job.total_runs) * 100) : 0
  return (
    <Link to="/queue" className="queue-pop-item" onClick={onNavigate}>
      <div className="queue-pop-item-head">
        <Typography.Text ellipsis className="queue-pop-item-title">
          {job.title}
        </Typography.Text>
        <Typography.Text type="secondary" className="queue-pop-item-count">
          {job.finished_runs}/{job.total_runs}
        </Typography.Text>
      </div>
      <Progress
        percent={percent}
        size="small"
        showInfo={false}
        status={job.status === 'queued' ? 'normal' : 'active'}
      />
      <Typography.Text type="secondary" className="queue-pop-item-meta">
        {job.status === 'queued' ? '排队中' : job.status === 'cancelling' ? '正在停止' : '运行中'}
        {' · '}
        {job.kind === 'funnel' ? '漏斗' : '自由组合'}
        {' · '}
        {shortTime(job.created_at)}
      </Typography.Text>
    </Link>
  )
}
