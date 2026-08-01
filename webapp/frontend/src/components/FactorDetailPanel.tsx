import {
  CheckCircleOutlined,
  DeleteOutlined,
  EditOutlined,
  RocketOutlined,
  TagsOutlined,
} from '@ant-design/icons'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Alert, Button, Collapse, Descriptions, Empty, Popconfirm, Skeleton, Space, Tag, Tooltip, Typography, message } from 'antd'
import { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { api, Factor } from '../api/client'
import { ProjectEditorModal } from './ProjectEditorModal'
import { RunConfigDrawer } from './RunConfigDrawer'
import { RunResultView } from './RunResultView'
import { StatusTag } from './StatusTag'
import { TagEditorModal } from './TagEditorModal'
import { durationText, shortTime, stageLabel } from '../lib/metrics'

export function FactorDetailEmpty({ library }: { library: 'test' | 'factor' }) {
  return (
    <div className="factor-workspace-empty">
      <Empty
        image={Empty.PRESENTED_IMAGE_SIMPLE}
        description={`从左侧选择一个${library === 'test' ? '测试' : ''}因子，这里会显示它最新一次的评价结果`}
      />
    </div>
  )
}

export function FactorDetailPanel({
  batchId,
  name,
  library,
}: {
  batchId: string
  name: string
  library: 'test' | 'factor'
}) {
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const factor = useQuery({ queryKey: ['factor', batchId, name], queryFn: () => api.factor(batchId, name) })
  const [runOpen, setRunOpen] = useState(false)
  const [tagEditorOpen, setTagEditorOpen] = useState(false)
  const [projectEditorOpen, setProjectEditorOpen] = useState(false)
  const submit = useMutation({
    mutationFn: (item: Factor) => api.submitFactor(item.batch_id, item.factor_name),
    onSuccess: (submitted) => {
      const replaced = submitted.admission_decision?.replaced_factor_names || []
      message.success(
        replaced.length
          ? `已提交到因子库；${replaced.join('、')} 因相关性高且表现较弱已降级至测试库`
          : '已提交到因子库',
      )
      queryClient.invalidateQueries({ queryKey: ['test-factors'] })
      queryClient.invalidateQueries({ queryKey: ['factors'] })
      navigate(`/factors/${submitted.batch_id}/${submitted.factor_name}`)
    },
    onError: (error) => message.error(error.message),
  })
  const removeFromLibrary = useMutation({
    mutationFn: (item: Factor) => api.removeFromFactorLibrary(item.batch_id, item.factor_name),
    onSuccess: () => {
      message.success('已移出因子库；源测试因子和历史评价均已保留')
      queryClient.invalidateQueries({ queryKey: ['factors'] })
      queryClient.invalidateQueries({ queryKey: ['test-factors'] })
      queryClient.invalidateQueries({ queryKey: ['factor-correlation'] })
      queryClient.invalidateQueries({ queryKey: ['factor-tags', 'factor'] })
      navigate('/factors')
    },
    onError: (error) => message.error(error.message),
  })

  if (factor.isLoading) {
    return (
      <div className="factor-workspace-body">
        <Skeleton active paragraph={{ rows: 10 }} />
      </div>
    )
  }
  if (factor.isError) {
    return (
      <div className="factor-workspace-body">
        <Alert
          type="error"
          showIcon
          message="无法读取该因子"
          description={factor.error.message}
          action={
            <Button size="small" onClick={() => factor.refetch()}>
              重试
            </Button>
          }
        />
      </div>
    )
  }
  if (!factor.data) return <Empty description="因子不存在" />

  const item: Factor = factor.data
  const inFactorLibrary = item.library_scope === 'factor'
  // 因子详情接口返回的是 runs（created_at 倒序），列表接口返回的是 latest_run；两种形状都接
  const latest = item.latest_run ?? item.runs?.[0]

  return (
    <div className="factor-workspace-body">
      <div className="factor-workspace-head">
        <div className="factor-workspace-title">
          <Typography.Title level={2}>{item.factor_name}</Typography.Title>
          <Space size={[6, 6]} wrap>
            {latest && <StatusTag status={latest.status} />}
            {latest?.gate_outcome && <StatusTag status={latest.gate_outcome} />}
            <Tag>{item.project}</Tag>
            {item.uses_proxy && <Tag color="gold">代理口径</Tag>}
            {!inFactorLibrary && item.submitted && <Tag color="green">已提交因子库</Tag>}
            {latest && (
              <Typography.Text type="secondary" className="factor-latest-meta">
                {shortTime(latest.finished_at || latest.created_at)} · {stageLabel(latest.stage)} · 耗时{' '}
                {durationText(latest.started_at, latest.finished_at)}
              </Typography.Text>
            )}
          </Space>
        </div>
        <Space wrap>
          {latest && <Link to={`/runs/${latest.id}`}>单独打开结果页</Link>}
          {!inFactorLibrary && (
            <Tooltip title="提交时先做相关性检验；若与库内因子高度相关，仅在 60 日 Sharpe 中位数（同分看 Fitness）严格更优时替换旧因子">
              <Button
                icon={<CheckCircleOutlined />}
                disabled={item.submitted || submit.isPending}
                loading={submit.isPending}
                onClick={() => submit.mutate(item)}
              >
                {item.submitted ? '已提交到因子库' : '提交到因子库'}
              </Button>
            </Tooltip>
          )}
          {inFactorLibrary && (
            <Popconfirm
              title="移出因子库？"
              description="仅移除正式库副本；源测试因子、标签和历史评价会保留。"
              okText="确认移出"
              cancelText="取消"
              okButtonProps={{ danger: true }}
              onConfirm={() => removeFromLibrary.mutate(item)}
            >
              <Button danger icon={<DeleteOutlined />} loading={removeFromLibrary.isPending}>
                移出因子库
              </Button>
            </Popconfirm>
          )}
          <Button type="primary" icon={<RocketOutlined />} onClick={() => setRunOpen(true)}>
            提交测试
          </Button>
        </Space>
      </div>

      <Collapse
        className="collapse-card factor-definition-collapse"
        items={[
          {
            key: 'definition',
            label: (
              <div className="factor-definition-label">
                <span>因子定义</span>
                <Typography.Text code ellipsis className="factor-definition-peek">
                  {item.expression}
                </Typography.Text>
              </div>
            ),
            children: (
              <Descriptions column={2} size="small">
                <Descriptions.Item label="批次">{item.batch_name}</Descriptions.Item>
                <Descriptions.Item label="项目">
                  <Space size={4}>
                    <span>{item.project}</span>
                    <Button size="small" type="text" icon={<EditOutlined />} onClick={() => setProjectEditorOpen(true)}>
                      修改
                    </Button>
                  </Space>
                </Descriptions.Item>
                <Descriptions.Item label="归类">{item.factor_category || '未分类'}</Descriptions.Item>
                <Descriptions.Item label="实现口径">{item.implementation_set}</Descriptions.Item>
                <Descriptions.Item label="库状态">
                  {inFactorLibrary ? '因子库' : item.submitted ? '测试库 · 已提交' : '测试库'}
                </Descriptions.Item>
                <Descriptions.Item label="代理口径">{item.uses_proxy ? '是' : '否'}</Descriptions.Item>
                <Descriptions.Item label="数据符号" span={2}>
                  {item.required_symbols.map((symbol) => (
                    <Tag key={symbol}>{symbol}</Tag>
                  ))}
                </Descriptions.Item>
                <Descriptions.Item label="研究标签" span={2}>
                  <Space wrap size={[4, 4]}>
                    {item.tags.length ? (
                      item.tags.map((tag) => <Tag key={tag}>{tag}</Tag>)
                    ) : (
                      <Typography.Text type="secondary">暂未设置</Typography.Text>
                    )}
                    <Button size="small" type="text" icon={<TagsOutlined />} onClick={() => setTagEditorOpen(true)}>
                      管理标签
                    </Button>
                  </Space>
                </Descriptions.Item>
                <Descriptions.Item label="表达式" span={2}>
                  <Typography.Text code copyable>
                    {item.expression}
                  </Typography.Text>
                </Descriptions.Item>
              </Descriptions>
            ),
          },
        ]}
      />

      {latest ? (
        <RunResultView runId={latest.id} />
      ) : (
        <Alert
          className="factor-no-run-note"
          type="info"
          showIcon
          message="这个因子还没有评价结果"
          description="点击右上角「提交测试」选择流水线与期间，任务完成后结果会出现在这里。"
        />
      )}

      <RunConfigDrawer open={runOpen} factors={[item]} onClose={() => setRunOpen(false)} />
      <TagEditorModal
        factor={item}
        library={item.library_scope || library}
        open={tagEditorOpen}
        onClose={() => setTagEditorOpen(false)}
      />
      <ProjectEditorModal
        factor={item}
        library={item.library_scope || library}
        open={projectEditorOpen}
        onClose={() => setProjectEditorOpen(false)}
      />
    </div>
  )
}
