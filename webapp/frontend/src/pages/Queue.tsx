import { DeleteOutlined } from '@ant-design/icons'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import {
  Button,
  Card,
  Collapse,
  Empty,
  Flex,
  Popconfirm,
  Progress,
  Segmented,
  Space,
  Table,
  Tag,
  Tooltip,
  Typography,
  message,
} from 'antd'
import { useState } from 'react'
import { Link } from 'react-router-dom'
import { api, Job } from '../api/client'
import { StatusTag } from '../components/StatusTag'
import { durationText, formatMetric, shortTime, stageLabel } from '../lib/metrics'

function JobRuns({ id }: { id: string }) {
  const job = useQuery({
    queryKey: ['job', id],
    queryFn: () => api.job(id),
    refetchInterval: (query) =>
      ['queued', 'running', 'cancelling'].includes(query.state.data?.status || '') ? 1800 : false,
  })
  const [gateFilter, setGateFilter] = useState<'all' | 'passed' | 'not_passed'>('all')
  const runs = job.data?.runs || []
  const passedCount = runs.filter((run) => run.gate_outcome === 'passed').length
  const notPassedCount = runs.filter((run) => run.gate_outcome === 'not_passed').length
  const hasGate = passedCount + notPassedCount > 0
  const filtered =
    hasGate && gateFilter !== 'all'
      ? runs.filter((run) => run.gate_outcome === gateFilter)
      : runs
  return (
    <>
      {hasGate && (
        <div className="job-gate-filter">
          <Typography.Text type="secondary">按门槛筛选</Typography.Text>
          <Segmented
            size="small"
            value={gateFilter}
            onChange={(value) => setGateFilter(value as 'all' | 'passed' | 'not_passed')}
            options={[
              { label: `全部 ${runs.length}`, value: 'all' },
              { label: `过关 ${passedCount}`, value: 'passed' },
              { label: `未过关 ${notPassedCount}`, value: 'not_passed' },
            ]}
          />
        </div>
      )}
      <Table
        size="small"
        rowKey="id"
        dataSource={filtered}
        pagination={filtered.length > 20 ? { pageSize: 20 } : false}
        columns={[
          {
            title: '因子',
            dataIndex: 'factor_name',
            render: (value, run) => <Link to={`/runs/${run.id}`}>{value}</Link>,
          },
          { title: '阶段', dataIndex: 'stage', width: 150, render: (value) => stageLabel(value) },
          { title: '状态', width: 100, render: (_, run) => <StatusTag status={run.status} /> },
          {
            title: '门槛',
            width: 100,
            render: (_, run) =>
              run.gate_outcome ? (
                <Tooltip title={run.gate_explanation}>
                  <span>
                    <StatusTag status={run.gate_outcome} />
                  </span>
                </Tooltip>
              ) : (
                <StatusTag status={run.gate_outcome} />
              ),
          },
          {
            title: 'IC 均值',
            width: 110,
            render: (_, run) =>
              typeof run.result?.ic_mean === 'number' ? formatMetric('ic_mean', run.result.ic_mean) : '暂无',
          },
          {
            title: '耗时',
            width: 110,
            render: (_, run) => durationText(run.started_at, run.finished_at),
          },
          {
            title: '备注',
            ellipsis: true,
            render: (_, run) =>
              run.error ? (
                <Tooltip title={<pre className="error-text">{run.error}</pre>} overlayStyle={{ maxWidth: 560 }}>
                  <Typography.Text type="danger" ellipsis>
                    {run.error.split('\n').filter(Boolean).pop()}
                  </Typography.Text>
                </Tooltip>
              ) : (
                '暂无'
              ),
          },
          {
            title: '',
            width: 70,
            render: (_, run) => <Link to={`/runs/${run.id}`}>详情</Link>,
          },
        ]}
      />
    </>
  )
}

export default function Queue() {
  const queryClient = useQueryClient()
  // 有活跃任务时 1.8 秒刷新；空闲时降到 12 秒，减少无意义轮询
  const jobs = useQuery({
    queryKey: ['jobs'],
    queryFn: api.jobs,
    refetchInterval: (query) =>
      (query.state.data || []).some((job) => ['queued', 'running', 'cancelling'].includes(job.status)) ? 1800 : 12000,
  })
  // 展开状态在会话内记住：点进结果详情再返回时不回到默认开合
  const [expanded, setExpanded] = useState<string[] | null>(() => {
    try {
      const raw = sessionStorage.getItem('queue:expanded')
      return raw ? (JSON.parse(raw) as string[]) : null
    } catch {
      return null
    }
  })
  const [deletingId, setDeletingId] = useState<string | null>(null)
  const handleExpandedChange = (keys: string | string[]) => {
    const next = Array.isArray(keys) ? keys : [keys]
    setExpanded(next)
    try {
      sessionStorage.setItem('queue:expanded', JSON.stringify(next))
    } catch {
      // 存储不可用时静默降级为内存状态
    }
  }
  const cancel = async (job: Job, force: boolean) => {
    try {
      await api.cancelJob(job.id, force)
      message.success(force ? '已强制停止当前任务' : '已取消所有排队项')
      jobs.refetch()
    } catch (error) {
      message.error((error as Error).message)
    }
  }
  const remove = async (job: Job) => {
    setDeletingId(job.id)
    try {
      await api.deleteJob(job.id)
      queryClient.removeQueries({ queryKey: ['job', job.id] })
      setExpanded((current) => current?.filter((id) => id !== job.id) ?? null)
      message.success('任务记录已删除；磁盘回测产物仍保留')
      await jobs.refetch()
    } catch (error) {
      message.error((error as Error).message)
    } finally {
      setDeletingId(null)
    }
  }
  const activeIds = (jobs.data || [])
    .filter((job) => ['queued', 'running', 'cancelling'].includes(job.status))
    .map((job) => job.id)
  const queueOverview = (jobs.data || []).reduce(
    (summary, job) => {
      if (job.status === 'running' || job.status === 'cancelling') summary.running += 1
      if (job.status === 'queued') summary.queued += 1
      if (job.status === 'succeeded') summary.completed += 1
      summary.remainingRuns += Math.max(job.total_runs - job.finished_runs, 0)
      return summary
    },
    { running: 0, queued: 0, completed: 0, remainingRuns: 0 },
  )
  // 默认展开活跃任务；用户手动开合后尊重用户选择
  const activeKeys = expanded ?? activeIds.slice(0, 1)

  return (
    <>
      <div className="page-heading">
        <div className="page-heading-copy">
          <Typography.Title level={2}>任务队列</Typography.Title>
          <Typography.Text type="secondary">
            单 worker 串行执行{activeIds.length ? `，当前 ${activeIds.length} 个活跃任务，每 1.8 秒自动刷新` : '，暂无活跃任务'}
          </Typography.Text>
        </div>
      </div>
      <div className="queue-overview" aria-label="队列概览">
        <div className="queue-overview-item">
          <span>执行中</span>
          <strong>{queueOverview.running}</strong>
        </div>
        <div className="queue-overview-item">
          <span>等待中</span>
          <strong>{queueOverview.queued}</strong>
        </div>
        <div className="queue-overview-item">
          <span>待完成运行</span>
          <strong>{queueOverview.remainingRuns}</strong>
        </div>
        <div className="queue-overview-item">
          <span>已完成任务</span>
          <strong>{queueOverview.completed}</strong>
        </div>
      </div>
      <Card className="surface-card">
        {(jobs.data || []).length === 0 && !jobs.isLoading ? (
          <Empty description="还没有任务。先从测试库选择因子并配置评价。">
            <Link to="/test-factors">
              <Button type="primary">前往测试库</Button>
            </Link>
          </Empty>
        ) : (
          <Collapse
            className="queue-list"
            bordered={false}
            activeKey={activeKeys}
            onChange={handleExpandedChange}
            items={(jobs.data || []).map((job) => ({
              key: job.id,
              label: (
                <Flex align="center" gap={14} wrap>
                  <StatusTag status={job.status} />
                  <Tag color={job.kind === 'funnel' ? 'purple' : 'blue'}>
                    {job.kind === 'funnel' ? '漏斗' : '自由组合'}
                  </Tag>
                  <Typography.Text strong>{job.title}</Typography.Text>
                  <Progress
                    percent={job.total_runs ? Math.round((job.finished_runs / job.total_runs) * 100) : 0}
                    size="small"
                    showInfo={false}
                    style={{ width: 200 }}
                    status={job.status === 'failed' ? 'exception' : undefined}
                  />
                  <Typography.Text type="secondary">
                    {job.finished_runs}/{job.total_runs}
                  </Typography.Text>
                  <Typography.Text type="secondary" style={{ fontSize: 12 }}>
                    {shortTime(job.created_at)}
                  </Typography.Text>
                </Flex>
              ),
              extra: (
                <Space onClick={(event) => event.stopPropagation()}>
                  {['queued', 'running', 'cancelling'].includes(job.status) ? (
                    <>
                      <Button size="small" onClick={() => cancel(job, false)}>
                        取消排队
                      </Button>
                      <Popconfirm
                        title="强制停止会中断正在计算的评价"
                        description="worker 将被重启，下一个任务需要重新加载行情数据"
                        onConfirm={() => cancel(job, true)}
                        okText="确认停止"
                        cancelText="再想想"
                      >
                        <Button danger size="small">
                          强制停止
                        </Button>
                      </Popconfirm>
                    </>
                  ) : (
                    <Popconfirm
                      title="删除这条任务记录？"
                      description="该任务下的运行记录会一并删除，磁盘上的回测产物不会删除。"
                      onConfirm={() => remove(job)}
                      okText="确认删除"
                      cancelText="取消"
                    >
                      <Button danger type="text" size="small" icon={<DeleteOutlined />} loading={deletingId === job.id}>
                        删除记录
                      </Button>
                    </Popconfirm>
                  )}
                </Space>
              ),
              children: <JobRuns id={job.id} />,
            }))}
          />
        )}
      </Card>
    </>
  )
}
