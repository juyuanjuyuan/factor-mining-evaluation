import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  Alert,
  Button,
  Card,
  Form,
  Input,
  InputNumber,
  Modal,
  Radio,
  Skeleton,
  Space,
  Tabs,
  Tag,
  Typography,
  message,
} from 'antd'
import { useState } from 'react'
import { api, FunnelStage, Template } from '../api/client'
import { PipelineBuilder } from '../components/PipelineBuilder'
import { SectionTabs } from '../components/SectionTabs'
import { STAGE_INFO, methodLabel } from '../lib/metrics'

type SaveTemplate = {
  id?: number
  payload: Partial<Template>
}

type StoredFunnelStage = Pick<FunnelStage, 'name' | 'methods'>

function templateStages(template?: Template): StoredFunnelStage[] {
  const value = template?.params?.stages
  if (!Array.isArray(value)) return []
  return value.filter(
    (stage): stage is StoredFunnelStage =>
      typeof stage === 'object'
      && stage !== null
      && typeof (stage as StoredFunnelStage).name === 'string'
      && Array.isArray((stage as StoredFunnelStage).methods),
  )
}

export default function PipelineTemplates() {
  const queryClient = useQueryClient()
  const stages = useQuery({ queryKey: ['funnel-stages'], queryFn: api.funnelStages })
  const templates = useQuery({ queryKey: ['templates'], queryFn: api.templates })
  const [open, setOpen] = useState(false)
  const [editing, setEditing] = useState<Template | null>(null)
  const [kind, setKind] = useState<'methods' | 'funnel'>('methods')
  const [pipelineMethods, setPipelineMethods] = useState<string[]>(['rank_ic', 'rank_icir'])
  const [funnelPipelines, setFunnelPipelines] = useState<Record<string, string[]>>({})
  const [form] = Form.useForm()

  const save = useMutation({
    mutationFn: ({ id, payload }: SaveTemplate) =>
      id ? api.updateTemplate(id, payload) : api.createTemplate(payload),
    onSuccess: () => {
      message.success(editing ? '模板已更新' : '模板已保存')
      closeEditor()
      queryClient.invalidateQueries({ queryKey: ['templates'] })
    },
    onError: (error) => message.error(error.message),
  })

  const closeEditor = () => {
    setOpen(false)
    setEditing(null)
    form.resetFields()
  }

  const initializeFunnelPipelines = (template?: Template) => {
    const stored = new Map(templateStages(template).map((stage) => [stage.name, stage.methods]))
    const initialized = Object.fromEntries(
      (stages.data || []).map((stage) => [
        stage.name,
        [...(stored.get(stage.name) || stage.methods)],
      ]),
    )
    setFunnelPipelines(initialized)
  }

  const openEditor = (template?: Template, clone = false) => {
    const nextKind = template?.kind || 'methods'
    setEditing(clone ? null : template || null)
    setKind(nextKind)
    setPipelineMethods(template?.methods?.length ? template.methods : ['rank_ic', 'rank_icir'])
    initializeFunnelPipelines(template)
    form.setFieldsValue({
      name: template ? `${template.name}${clone ? ' 副本' : ''}` : '',
      horizon: Number(template?.params?.horizon ?? 1),
      n_quantiles: Number(template?.params?.n_quantiles ?? 10),
      decay: Number(template?.params?.decay ?? 1),
      significance_level: Number(template?.params?.significance_level ?? 0.05),
    })
    setOpen(true)
  }

  const remove = async (id: number) => {
    try {
      await api.deleteTemplate(id)
      message.success('模板已删除')
      queryClient.invalidateQueries({ queryKey: ['templates'] })
    } catch (error) {
      message.error((error as Error).message)
    }
  }

  const submitTemplate = (values: Record<string, number | string>) => {
    if (kind === 'methods' && pipelineMethods.length === 0) {
      message.warning('请至少添加一个评价方法')
      return
    }
    const params =
      kind === 'funnel'
        ? {
            horizon: values.horizon,
            n_quantiles: values.n_quantiles,
            decay: values.decay,
            significance_level: values.significance_level,
            stages: (stages.data || []).map((stage) => ({
              name: stage.name,
              methods: funnelPipelines[stage.name] || [],
            })),
          }
        : {}
    save.mutate({
      id: editing?.id,
      payload: {
        name: String(values.name),
        kind,
        methods: kind === 'methods' ? pipelineMethods : [],
        params,
      },
    })
  }

  return (
    <>
      <div className="page-heading page-heading-row">
        <div className="page-heading-copy">
          <Typography.Title level={2}>流水线模板</Typography.Title>
          <Typography.Text type="secondary">
            管理评价方法的执行顺序；完整数学定义和方法说明集中在评价模块库
          </Typography.Text>
        </div>
        <Space wrap>
          <SectionTabs
            tabs={[
              { label: '流水线模板', path: '/pipelines' },
              { label: '评价模块库', path: '/methods' },
            ]}
          />
          <Button type="primary" onClick={() => openEditor()}>
            新建模板
          </Button>
        </Space>
      </div>

      <Card className="surface-card">
        {templates.isLoading ? (
          <Skeleton active />
        ) : (
          <div className="template-list">
            {(templates.data || []).map((template) => (
              <div key={template.id} className="template-row">
                <div className="template-row-body">
                  <Space wrap>
                    <Typography.Text strong>{template.name}</Typography.Text>
                    {template.is_builtin && <Tag color="blue">内置</Tag>}
                    <Tag>{template.kind === 'funnel' ? '漏斗' : '方法组合'}</Tag>
                  </Space>
                  <div className="template-row-desc">
                    {template.kind === 'funnel'
                      ? `标准四阶段 · H${Number(template.params?.horizon ?? 1)} · ${Number(template.params?.n_quantiles ?? 10)} 组 · Decay ${Number(template.params?.decay ?? 1)} · 显著性水平 ${Number(template.params?.significance_level ?? 0.05)}`
                      : (template.methods || []).map(methodLabel).join(' → ')}
                  </div>
                </div>
                <Space className="template-row-actions">
                  <Button type="link" onClick={() => openEditor(template)}>
                    编辑
                  </Button>
                  {template.is_builtin ? (
                    <Button type="link" onClick={() => openEditor(template, true)}>
                      复制为新模板
                    </Button>
                  ) : (
                    <Button danger type="link" onClick={() => remove(template.id)}>
                      删除
                    </Button>
                  )}
                </Space>
              </div>
            ))}
          </div>
        )}
      </Card>

      <Modal
        title={editing ? '编辑流水线模板' : '新建流水线模板'}
        open={open}
        width={980}
        onCancel={closeEditor}
        onOk={() => form.submit()}
        confirmLoading={save.isPending}
      >
        <Form form={form} layout="vertical" onFinish={submitTemplate}>
          <Form.Item name="name" label="模板名称" rules={[{ required: true, whitespace: true }]}>
            <Input disabled={Boolean(editing?.is_builtin)} />
          </Form.Item>
          <Form.Item label="类型">
            <Radio.Group
              value={kind}
              onChange={(event) => {
                const nextKind = event.target.value as 'methods' | 'funnel'
                setKind(nextKind)
                if (nextKind === 'funnel' && Object.keys(funnelPipelines).length === 0) {
                  initializeFunnelPipelines()
                }
              }}
              disabled={Boolean(editing)}
            >
              <Radio.Button value="methods">方法组合</Radio.Button>
              <Radio.Button value="funnel">完整漏斗</Radio.Button>
            </Radio.Group>
          </Form.Item>

          {kind === 'methods' ? (
            <Form.Item label="评价流水线（按顺序执行，可拖拽调整、可重复添加）">
              <PipelineBuilder value={pipelineMethods} onChange={setPipelineMethods} />
            </Form.Item>
          ) : (
            <>
              <Alert
                type="info"
                showIcon
                title="标准四阶段漏斗"
                description="阶段顺序和门槛逻辑固定；每个阶段的方法顺序、重复执行和扩展诊断均可编辑。门槛所需核心方法必须保留。"
                style={{ marginBottom: 16 }}
              />
              <Space size="large" wrap>
                <Form.Item
                  name="horizon"
                  label="持有期（天）"
                  rules={[{ required: true }]}
                >
                  <InputNumber min={1} max={60} />
                </Form.Item>
                <Form.Item
                  name="n_quantiles"
                  label="分位组数"
                  rules={[{ required: true }]}
                >
                  <InputNumber min={2} max={20} />
                </Form.Item>
                <Form.Item
                  name="decay"
                  label="Decay（天）"
                  rules={[
                    { required: true },
                    { type: 'number', min: 1, message: 'Decay 必须是正整数' },
                  ]}
                  tooltip="n>1 时在交易前对因子值按 n,n−1,…,1 做线性衰减；n 必须为正整数，1 不额外平滑。"
                >
                  <InputNumber min={1} precision={0} />
                </Form.Item>
                <Form.Item
                  name="significance_level"
                  label="IC 显著性水平"
                  rules={[{ required: true }]}
                  tooltip="原始 IC 与市值中性化 IC 阶段的 Newey-West p 值阈值"
                >
                  <InputNumber min={0.001} max={0.5} step={0.01} />
                </Form.Item>
              </Space>
              <Card size="small" title="编辑四阶段方法" className="funnel-stage-preview">
                <Tabs
                  items={(stages.data || []).map((stage) => ({
                    key: stage.name,
                    label: STAGE_INFO[stage.name]?.label || stage.name,
                    children: (
                      <div>
                        <Alert
                          type="warning"
                          showIcon
                          title={STAGE_INFO[stage.name]?.hint}
                          description={`必须按顺序保留：${stage.required_methods.map(methodLabel).join(' → ')}`}
                          style={{ marginBottom: 12 }}
                        />
                        <PipelineBuilder
                          value={funnelPipelines[stage.name] || stage.methods}
                          onChange={(next) =>
                            setFunnelPipelines((previous) => ({
                              ...previous,
                              [stage.name]: next,
                            }))
                          }
                        />
                      </div>
                    ),
                  }))}
                />
              </Card>
            </>
          )}
        </Form>
      </Modal>
    </>
  )
}
