import { LeftOutlined, RightOutlined } from '@ant-design/icons'
import { useQuery } from '@tanstack/react-query'
import { Button, Empty, Skeleton, Tag, Typography } from 'antd'
import { useNavigate, useParams } from 'react-router-dom'
import { api } from '../api/client'
import { BackButton } from '../components/BackButton'
import { RunResultView } from '../components/RunResultView'
import { StatusTag } from '../components/StatusTag'
import { durationText, shortTime, stageLabel } from '../lib/metrics'

export default function RunDetail() {
  const { runId = '' } = useParams()
  const navigate = useNavigate()
  const run = useQuery({
    queryKey: ['run', runId],
    queryFn: () => api.run(runId),
    refetchInterval: (query) =>
      ['queued', 'running'].includes(query.state.data?.status || '') ? 1500 : false,
  })
  const jobId = run.data?.job_id
  const job = useQuery({
    queryKey: ['job', jobId || ''],
    queryFn: () => api.job(jobId!),
    enabled: Boolean(jobId),
    refetchInterval: (query) =>
      ['queued', 'running'].includes(query.state.data?.status || '') ? 1800 : false,
  })
  if (run.isLoading) return <Skeleton active />
  if (!run.data) return <Empty description="结果不存在" />
  const data = run.data
  const jobRuns = job.data?.runs || []
  const queueIndex = jobRuns.findIndex((item) => item.id === data.id)
  const previousRun = queueIndex > 0 ? jobRuns[queueIndex - 1] : null
  const nextRun = queueIndex >= 0 && queueIndex < jobRuns.length - 1 ? jobRuns[queueIndex + 1] : null
  const showQueueNavigation = queueIndex >= 0 && jobRuns.length > 1

  return (
    <>
      <div className="page-heading page-heading-row">
        <div className="page-heading-copy">
          <BackButton fallback="/queue" />
          <Typography.Title level={2}>{data.factor_name}</Typography.Title>
          <span className="heading-tags">
            <StatusTag status={data.status} />
            {data.stage && <Tag color="geekblue">{stageLabel(data.stage)}</Tag>}
            {data.gate_outcome && <StatusTag status={data.gate_outcome} />}
            <Typography.Text type="secondary">
              {shortTime(data.started_at)} · 耗时 {durationText(data.started_at, data.finished_at)}
            </Typography.Text>
          </span>
        </div>
        {showQueueNavigation && (
          <div className="run-sequence-nav">
            <Typography.Text type="secondary">
              队列 {queueIndex + 1}/{jobRuns.length}
            </Typography.Text>
            <Button icon={<LeftOutlined />} disabled={!previousRun} onClick={() => navigate(`/runs/${previousRun?.id}`)}>
              上一个
            </Button>
            <Button type="primary" disabled={!nextRun} onClick={() => navigate(`/runs/${nextRun?.id}`)}>
              下一个 <RightOutlined />
            </Button>
          </div>
        )}
      </div>
      <RunResultView runId={runId} />
    </>
  )
}
