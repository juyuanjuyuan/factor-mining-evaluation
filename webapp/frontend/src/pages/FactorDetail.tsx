import { CheckCircleOutlined, DeleteOutlined, EditOutlined, RocketOutlined, TagsOutlined } from '@ant-design/icons'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Button, Card, Descriptions, Empty, Popconfirm, Space, Table, Tag, Typography, message } from 'antd'
import { useState } from 'react'
import { Link, useLocation, useNavigate, useParams } from 'react-router-dom'
import { api, Factor } from '../api/client'
import { BackButton } from '../components/BackButton'
import { RunConfigDrawer } from '../components/RunConfigDrawer'
import { ProjectEditorModal } from '../components/ProjectEditorModal'
import { StatusTag } from '../components/StatusTag'
import { TagEditorModal } from '../components/TagEditorModal'
import { durationText, formatMetric, methodLabel, shortTime, stageLabel } from '../lib/metrics'

export default function FactorDetail() {
  const { batchId = '', name = '' } = useParams()
  const navigate = useNavigate()
  const location = useLocation()
  const queryClient = useQueryClient()
  const fromTestLibrary = location.pathname.startsWith('/test-factors')
  const factor = useQuery({ queryKey: ['factor', batchId, name], queryFn: () => api.factor(batchId, name) })
  const [runOpen, setRunOpen] = useState(false)
  const [tagEditorOpen, setTagEditorOpen] = useState(false)
  const [projectEditorOpen, setProjectEditorOpen] = useState(false)
  const submit = useMutation({
    mutationFn: (item: Factor) => api.submitFactor(item.batch_id, item.factor_name),
    onSuccess: (submitted) => {
      message.success('已提交到因子库')
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
  if (factor.isLoading) return <Card loading />
  if (!factor.data) return <Empty description="因子不存在" />
  const item: Factor = factor.data
  return (
    <>
      <div className="page-heading">
        <div>
          <BackButton
            fallback={fromTestLibrary ? '/test-factors' : '/factors'}
            label={`返回${fromTestLibrary ? '测试库' : '因子库'}`}
          />
          <Typography.Title level={2}>{item.factor_name}</Typography.Title>
        </div>
        <Space>
          {fromTestLibrary && item.library_scope !== 'factor' && (
            <Button
              icon={<CheckCircleOutlined />}
              disabled={item.submitted || submit.isPending}
              onClick={() => submit.mutate(item)}
            >
              {item.submitted ? '已提交到因子库' : '提交到因子库'}
            </Button>
          )}
          {item.library_scope === 'factor' && (
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
            运行评价
          </Button>
        </Space>
      </div>
      <Card className="surface-card" title="因子定义">
        <Descriptions column={2}>
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
          <Descriptions.Item label="库状态">{item.library_scope === 'factor' ? '因子库' : item.submitted ? '测试库 · 已提交' : '测试库'}</Descriptions.Item>
          <Descriptions.Item label="代理口径">{item.uses_proxy ? '是' : '否'}</Descriptions.Item>
          <Descriptions.Item label="数据符号" span={2}>
            {item.required_symbols.map((symbol) => (
              <Tag key={symbol}>{symbol}</Tag>
            ))}
          </Descriptions.Item>
          <Descriptions.Item label="研究标签" span={2}>
            <Space wrap>
              {item.tags.length ? item.tags.map((tag) => <Tag key={tag}>{tag}</Tag>) : <Typography.Text type="secondary">暂未设置</Typography.Text>}
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
      </Card>
      <Card className="surface-card section-card" title="运行历史">
        <Table
          rowKey="id"
          dataSource={item.runs || []}
          pagination={{ pageSize: 10 }}
          locale={{ emptyText: <Empty description="尚未评价，点击右上角「运行评价」" /> }}
          columns={[
            {
              title: '时间',
              width: 150,
              render: (_, run) => shortTime(run.created_at),
            },
            { title: '流程', width: 140, render: (_, run) => stageLabel(run.stage) },
            { title: '状态', width: 100, render: (_, run) => <StatusTag status={run.status} /> },
            {
              title: '方法',
              render: (_, run) => run.methods.map((method) => <Tag key={method}>{methodLabel(method)}</Tag>),
            },
            {
              title: 'IC 均值',
              width: 110,
              render: (_, run) =>
                typeof run.result?.ic_mean === 'number' ? (
                  <span className={`metric-number ${(run.result.ic_mean as number) > 0 ? 'good' : 'bad'}`}>
                    {formatMetric('ic_mean', run.result.ic_mean)}
                  </span>
                ) : (
                  '暂无'
                ),
            },
            {
              title: 'ICIR',
              width: 110,
              render: (_, run) =>
                typeof run.result?.ir === 'number' ? formatMetric('ir', run.result.ir) : '暂无',
            },
            {
              title: '耗时',
              width: 100,
              render: (_, run) => durationText(run.started_at, run.finished_at),
            },
            {
              title: '',
              width: 70,
              render: (_, run) => <Link to={`/runs/${run.id}`}>查看</Link>,
            },
          ]}
        />
      </Card>
      <RunConfigDrawer open={runOpen} factors={[item]} onClose={() => setRunOpen(false)} />
      <TagEditorModal
        factor={item}
        library={item.library_scope || (fromTestLibrary ? 'test' : 'factor')}
        open={tagEditorOpen}
        onClose={() => setTagEditorOpen(false)}
      />
      <ProjectEditorModal
        factor={item}
        library={item.library_scope || (fromTestLibrary ? 'test' : 'factor')}
        open={projectEditorOpen}
        onClose={() => setProjectEditorOpen(false)}
      />
    </>
  )
}
