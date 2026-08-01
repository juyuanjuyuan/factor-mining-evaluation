import { HistoryOutlined } from '@ant-design/icons'
import { useQuery } from '@tanstack/react-query'
import {
  Alert,
  Button,
  Drawer,
  Form,
  Input,
  InputNumber,
  Space,
  Steps,
  Tag,
  Tooltip,
  Typography,
  message,
} from 'antd'
import { useEffect, useRef, useState } from 'react'
import { api, Factor, FunnelStage, GateSpec, Template } from '../api/client'
import { STAGE_INFO, methodLabel } from '../lib/metrics'
import { PipelineBuilder } from './PipelineBuilder'
import { GateBuilder } from './GateBuilder'

const CUSTOM = -1
const LAST_CONFIG_KEY = 'run-config:last:v1'

const EMPTY_GATE: GateSpec = { conditions: [], match: 'all' }

type StoredConfig = {
  templateId: number
  kind: 'evaluate' | 'funnel'
  methods: string[]
  horizon: number
  n_quantiles: number
  decay: number
  significance_level: number
  signal_start?: string
  signal_end?: string
  gate: GateSpec
}

/** 读取上次提交的配置；结构对不上就当作没有，回落到默认模板。 */
function readLastConfig(): StoredConfig | null {
  try {
    const raw = localStorage.getItem(LAST_CONFIG_KEY)
    if (!raw) return null
    const parsed = JSON.parse(raw) as StoredConfig
    if (!Array.isArray(parsed.methods) || typeof parsed.horizon !== 'number') return null
    return {
      ...parsed,
      kind: parsed.kind === 'funnel' ? 'funnel' : 'evaluate',
      decay: typeof parsed.decay === 'number' ? parsed.decay : 1,
      gate:
        parsed.gate && Array.isArray(parsed.gate.conditions)
          ? { conditions: parsed.gate.conditions, match: parsed.gate.match === 'any' ? 'any' : 'all' }
          : EMPTY_GATE,
    }
  } catch {
    return null
  }
}

/** Read a saved gate off a template's params, ignoring malformed shapes. */
function templateGate(template: Template): GateSpec | null {
  const raw = template.params?.gate as GateSpec | undefined
  if (!raw || !Array.isArray(raw.conditions) || !raw.conditions.length) return null
  return { conditions: raw.conditions, match: raw.match === 'any' ? 'any' : 'all' }
}

type StoredFunnelStage = Pick<FunnelStage, 'name' | 'methods'>

type TagSelection = {
  tags: string[]
  match: 'any' | 'all'
  library: 'test' | 'factor'
}

function templateStages(template: Template): StoredFunnelStage[] {
  const value = template.params?.stages
  if (!Array.isArray(value)) return []
  return value.filter(
    (stage): stage is StoredFunnelStage =>
      typeof stage === 'object'
      && stage !== null
      && typeof (stage as StoredFunnelStage).name === 'string'
      && Array.isArray((stage as StoredFunnelStage).methods),
  )
}

function Section({
  title,
  hint,
  extra,
  children,
}: {
  title: string
  hint?: string
  extra?: React.ReactNode
  children: React.ReactNode
}) {
  return (
    <section className="run-config-section">
      <div className="run-config-section-head">
        <div>
          <Typography.Text strong className="run-config-section-title">
            {title}
          </Typography.Text>
          {hint && (
            <Typography.Text type="secondary" className="run-config-section-hint">
              {hint}
            </Typography.Text>
          )}
        </div>
        {extra}
      </div>
      {children}
    </section>
  )
}

export function RunConfigDrawer({
  open,
  factors,
  tagSelection,
  onClose,
}: {
  open: boolean
  factors: Factor[]
  tagSelection?: TagSelection
  onClose: () => void
}) {
  const [kind, setKind] = useState<'evaluate' | 'funnel'>('evaluate')
  const [methods, setMethods] = useState<string[]>([])
  const [templateId, setTemplateId] = useState<number | typeof CUSTOM>(CUSTOM)
  const [funnelPipelines, setFunnelPipelines] = useState<Record<string, string[]>>({})
  const [gate, setGate] = useState<GateSpec>(EMPTY_GATE)
  const [submitting, setSubmitting] = useState(false)
  const [pipelineValid, setPipelineValid] = useState(true)
  const [reusedLast, setReusedLast] = useState(false)
  const methodCatalog = useQuery({ queryKey: ['methods'], queryFn: api.methods })
  const funnelStageCatalog = useQuery({ queryKey: ['funnel-stages'], queryFn: api.funnelStages })
  const templates = useQuery({ queryKey: ['templates'], queryFn: api.templates })
  const timeline = useQuery({
    queryKey: ['market-timeline'],
    queryFn: api.marketTimeline,
    enabled: open,
  })
  const [form] = Form.useForm()
  const horizon = Form.useWatch('horizon', form) ?? 1
  const nQuantiles = Form.useWatch('n_quantiles', form) ?? 10
  const decay = Form.useWatch('decay', form) ?? 1
  const signalStart = Form.useWatch('signal_start', form)
  const signalEnd = Form.useWatch('signal_end', form)

  const validateWindow = () => {
    const start = form.getFieldValue('signal_start') as string | undefined
    const end = form.getFieldValue('signal_end') as string | undefined
    if (Boolean(start) !== Boolean(end)) {
      return Promise.reject(new Error('请同时填写开始日和结束日'))
    }
    if (start && end && start > end) {
      return Promise.reject(new Error('开始日不能晚于结束日'))
    }
    return Promise.resolve()
  }

  const applyTemplate = (template: Template) => {
    setTemplateId(template.id)
    setKind(template.kind === 'funnel' ? 'funnel' : 'evaluate')
    if (template.kind === 'methods') setMethods(template.methods)
    setGate(template.kind === 'methods' ? templateGate(template) ?? EMPTY_GATE : EMPTY_GATE)
    if (template.kind === 'funnel') {
      setFunnelPipelines(
        Object.fromEntries(templateStages(template).map((stage) => [stage.name, stage.methods])),
      )
    }
    const templateHorizon = template.params?.horizon
    if (typeof templateHorizon === 'number') form.setFieldValue('horizon', templateHorizon)
    const templateQuantiles = template.params?.n_quantiles
    if (typeof templateQuantiles === 'number') form.setFieldValue('n_quantiles', templateQuantiles)
    const templateDecay = template.params?.decay
    form.setFieldValue('decay', typeof templateDecay === 'number' ? templateDecay : 1)
    const significance = template.params?.significance_level
    if (typeof significance === 'number') form.setFieldValue('significance_level', significance)
  }

  const applyDefaultTemplate = () => {
    const first = templates.data?.[0]
    if (!first) return
    applyTemplate(first)
    form.setFieldsValue({ signal_start: undefined, signal_end: undefined })
    setReusedLast(false)
  }

  // 每次打开只初始化一次：优先沿用上次提交的配置，否则落到第一个内置模板
  const initialized = useRef(false)
  useEffect(() => {
    if (!open) {
      initialized.current = false
      return
    }
    if (initialized.current || !templates.isFetched) return
    initialized.current = true
    const last = readLastConfig()
    if (last) {
      setTemplateId(last.templateId)
      setKind(last.kind)
      setMethods(last.methods)
      setGate(last.gate)
      form.setFieldsValue({
        horizon: last.horizon,
        n_quantiles: last.n_quantiles,
        decay: last.decay,
        significance_level: last.significance_level,
        signal_start: last.signal_start,
        signal_end: last.signal_end,
      })
      setReusedLast(true)
      return
    }
    applyDefaultTemplate()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, templates.isFetched])

  const chooseCustom = () => {
    setTemplateId(CUSTOM)
    setKind('evaluate')
    if (!methods.length && methodCatalog.data) {
      setMethods(methodCatalog.data.filter((item) => item.is_default).map((item) => item.name))
    }
  }

  const submit = async () => {
    const values = await form.validateFields()
    if (kind === 'evaluate' && !methods.length) {
      message.warning('请至少选择一个评价方法')
      return
    }
    if (kind === 'evaluate' && !pipelineValid) {
      message.warning('请先补全流水线中缺失的前置依赖')
      return
    }
    const gateConditions = gate.conditions.filter((condition) => condition.metric)
    setSubmitting(true)
    try {
      await api.createJob({
        kind,
        template_id: templateId === CUSTOM ? undefined : templateId,
        ...(tagSelection
          ? {
              tags: tagSelection.tags,
              tag_match: tagSelection.match,
              library: tagSelection.library,
            }
          : {
              factors: factors.map((factor) => ({
                factor_name: factor.factor_name,
                batch_id: factor.batch_id,
              })),
            }),
        methods,
        horizon: values.horizon,
        n_quantiles: values.n_quantiles,
        decay: values.decay,
        significance_level: values.significance_level,
        signal_start: values.signal_start || undefined,
        signal_end: values.signal_end || undefined,
        gate:
          kind === 'evaluate' && gateConditions.length
            ? { conditions: gateConditions, match: gate.match }
            : undefined,
      })
      try {
        localStorage.setItem(
          LAST_CONFIG_KEY,
          JSON.stringify({
            templateId,
            kind,
            methods,
            horizon: values.horizon,
            n_quantiles: values.n_quantiles,
            decay: values.decay,
            significance_level: values.significance_level,
            signal_start: values.signal_start || undefined,
            signal_end: values.signal_end || undefined,
            gate: { conditions: gateConditions, match: gate.match },
          } satisfies StoredConfig),
        )
      } catch {
        // 存储不可用时不影响提交本身
      }
      message.success('任务已提交，进度可在右上角队列图标查看')
      onClose()
    } catch (error) {
      message.error((error as Error).message)
    } finally {
      setSubmitting(false)
    }
  }

  const hasProxyFactor = factors.some((factor) => factor.uses_proxy)
  const supportsNetReturns = methodCatalog.data?.some((method) => method.name === 'quantile_net_returns')
  const usesNetReturns = kind === 'evaluate' && methods.includes('quantile_net_returns')
  const usesGrossReturns = kind === 'evaluate' && methods.includes('quantile_returns') && !usesNetReturns
  const canSubmit = kind === 'funnel' || (methods.length > 0 && pipelineValid)
  const activeTemplate = templates.data?.find((template) => template.id === templateId)
  const funnelStepCount = (funnelStageCatalog.data || []).reduce(
    (total, stage) => total + (funnelPipelines[stage.name] || stage.methods).length,
    0,
  )
  const stepCount = kind === 'funnel' ? funnelStepCount : methods.length
  const windowText = signalStart && signalEnd ? `${signalStart} ~ ${signalEnd}` : '全部历史'

  return (
    <Drawer
      title="提交测试"
      className="run-config-drawer"
      size="min(1120px, 100vw)"
      open={open}
      onClose={onClose}
      footer={
        <div className="run-config-footer">
          <Typography.Text type="secondary">
            {tagSelection
              ? `按标签重新匹配 · 当前命中 ${factors.length} 个因子`
              : `${factors.length} 个因子 · ${stepCount} 个评价步骤`}
          </Typography.Text>
          <Space>
            <Button onClick={onClose}>取消</Button>
            <Tooltip title={canSubmit ? '' : '请先补全评价流水线'}>
              <Button type="primary" loading={submitting} disabled={!canSubmit} onClick={submit}>
                提交任务
              </Button>
            </Tooltip>
          </Space>
        </div>
      }
    >
      <div className="run-config-body">
        <div className="run-config-main">
          {reusedLast && (
            <Alert
              className="run-config-reuse"
              type="info"
              showIcon
              icon={<HistoryOutlined />}
              message="已沿用上次提交的配置"
              action={
                <Button size="small" type="link" onClick={applyDefaultTemplate}>
                  改回默认模板
                </Button>
              }
            />
          )}

          <Section
            title="评价对象"
            hint={
              tagSelection
                ? `标签${tagSelection.match === 'all' ? '同时包含' : '包含任一'}，提交时重新匹配，不受分页影响`
                : '来自库中的当前选择'
            }
          >
            <div className="run-config-targets">
              {tagSelection
                ? tagSelection.tags.map((tag) => (
                    <Tag key={tag} color="processing">
                      {tag}
                    </Tag>
                  ))
                : factors.slice(0, 16).map((factor) => (
                    <Tag key={`${factor.batch_id}/${factor.factor_name}`}>{factor.factor_name}</Tag>
                  ))}
              {!tagSelection && factors.length > 16 && <Tag>… 共 {factors.length} 个</Tag>}
            </div>
          </Section>

          <Section title="评价方案" hint="选一个模板作为起点，再按需改动下面的流水线">
            <div className="template-choice-list">
              {(templates.data || []).map((template) => (
                <button
                  type="button"
                  key={template.id}
                  className={`template-choice${templateId === template.id ? ' active' : ''}`}
                  onClick={() => applyTemplate(template)}
                >
                  <span className="template-choice-name">{template.name}</span>
                  <span className="template-choice-meta">
                    {template.kind === 'funnel' ? '四阶段漏斗' : `${template.methods.length} 个步骤`}
                  </span>
                </button>
              ))}
              <button
                type="button"
                className={`template-choice${templateId === CUSTOM ? ' active' : ''}`}
                onClick={chooseCustom}
              >
                <span className="template-choice-name">自定义</span>
                <span className="template-choice-meta">从默认方法开始搭</span>
              </button>
            </div>
            {kind === 'funnel' ? (
              <>
                <Typography.Paragraph type="secondary" className="run-config-note">
                  漏斗按以下四个阶段依次执行，任一阶段未通过即淘汰，不再进入后续阶段：
                </Typography.Paragraph>
                <Steps
                  orientation="vertical"
                  size="small"
                  current={-1}
                  items={(funnelStageCatalog.data || []).map((stage) => ({
                    title: STAGE_INFO[stage.name]?.label || stage.name,
                    description: (
                      <div>
                        <div>{STAGE_INFO[stage.name]?.hint}</div>
                        <div className="funnel-stage-methods">
                          {(funnelPipelines[stage.name] || stage.methods).map((method, index) => (
                            <Tag key={`${method}-${index}`}>{methodLabel(method)}</Tag>
                          ))}
                        </div>
                      </div>
                    ),
                  }))}
                />
              </>
            ) : (
              <>
                <Typography.Paragraph type="secondary" className="run-config-note">
                  按顺序执行，可拖拽调整、可重复添加同一方法。
                </Typography.Paragraph>
                <PipelineBuilder
                  value={methods}
                  onValidityChange={setPipelineValid}
                  onChange={(next) => {
                    setMethods(next)
                    setTemplateId(CUSTOM)
                  }}
                />
              </>
            )}
          </Section>

          <Form
            form={form}
            layout="vertical"
            initialValues={{ horizon: 1, n_quantiles: 10, decay: 1, significance_level: 0.05 }}
          >
            <Section title="参数与评价期间">
              <Space size="large" wrap>
                <Form.Item
                  label="持有期（天）"
                  name="horizon"
                  rules={[{ required: true }]}
                  tooltip="开盘买入后持有的交易日数，收益口径为 open[t+1+H]/open[t+1]-1"
                >
                  <InputNumber min={1} max={60} />
                </Form.Item>
                <Form.Item label="分位组数" name="n_quantiles" rules={[{ required: true }]}>
                  <InputNumber min={2} max={20} />
                </Form.Item>
                <Form.Item
                  label="Decay（天）"
                  name="decay"
                  rules={[
                    { required: true },
                    { type: 'number', min: 1, message: 'Decay 必须是正整数' },
                  ]}
                  tooltip="对因子值做线性衰减：n>1 时按今天至前 n−1 日的 n,n−1,…,1 权重平均。n 必须为正整数；默认 1 与历史回测一致。"
                >
                  <InputNumber min={1} precision={0} />
                </Form.Item>
                {kind === 'funnel' && (
                  <Form.Item
                    label="显著性水平"
                    name="significance_level"
                    tooltip="IC 阶段的 Newey-West p 值淘汰阈值"
                  >
                    <InputNumber min={0.001} max={0.5} step={0.01} />
                  </Form.Item>
                )}
              </Space>
              <div className="evaluation-window-section">
                <div className="evaluation-window-heading">
                  <div>
                    <Typography.Text strong>评价期间（信号日）</Typography.Text>
                    <Typography.Paragraph type="secondary">
                      留空表示使用全部历史。因子仍先用完整历史计算滚动值，再截取该期间；末尾会按持有期裁去无法在期间内平仓的信号日。
                    </Typography.Paragraph>
                  </div>
                  {(signalStart || signalEnd) && (
                    <Button
                      type="link"
                      onClick={() => form.setFieldsValue({ signal_start: undefined, signal_end: undefined })}
                    >
                      清空（全历史）
                    </Button>
                  )}
                </div>
                <div className="evaluation-window-fields">
                  <Form.Item
                    label="开始日"
                    name="signal_start"
                    dependencies={['signal_end']}
                    rules={[{ validator: validateWindow }]}
                  >
                    <Input
                      type="date"
                      min={timeline.data?.start_day}
                      max={timeline.data?.end_day}
                      placeholder={timeline.data?.start_day}
                    />
                  </Form.Item>
                  <span className="evaluation-window-separator">至</span>
                  <Form.Item
                    label="结束日"
                    name="signal_end"
                    dependencies={['signal_start']}
                    rules={[{ validator: validateWindow }]}
                  >
                    <Input
                      type="date"
                      min={timeline.data?.start_day}
                      max={timeline.data?.end_day}
                      placeholder={timeline.data?.end_day}
                    />
                  </Form.Item>
                </div>
                <Typography.Text type="secondary" className="evaluation-window-current">
                  本次选择：{windowText}
                  {timeline.data ? ` · 数据覆盖 ${timeline.data.start_day} ~ ${timeline.data.end_day}` : ''}
                </Typography.Text>
              </div>
            </Section>

            {kind === 'evaluate' && (
              <Section
                title="结果门槛（可选）"
                hint="任务完成后自动标记「通过 / 未过关」，只影响标记，不会跳过任何计算"
                extra={
                  gate.conditions.length > 0 ? (
                    <Tag className="gate-count-tag">
                      {gate.conditions.length} 个条件 · {gate.match === 'all' ? '全部满足' : '满足任一'}
                    </Tag>
                  ) : undefined
                }
              >
                <GateBuilder value={gate} onChange={setGate} />
              </Section>
            )}
          </Form>
        </div>

        <aside className="run-config-aside" aria-label="本次提交摘要">
          <div className="run-config-aside-inner">
            <div className="run-config-summary-card">
              <Typography.Text strong>本次提交</Typography.Text>
              <dl className="run-config-summary-list">
                <div>
                  <dt>评价对象</dt>
                  <dd>{tagSelection ? `标签匹配 ${factors.length} 个` : `${factors.length} 个因子`}</dd>
                </div>
                <div>
                  <dt>方案</dt>
                  <dd>{activeTemplate ? activeTemplate.name : '自定义'}</dd>
                </div>
                <div>
                  <dt>类型</dt>
                  <dd>{kind === 'funnel' ? '四阶段漏斗' : '自由组合'}</dd>
                </div>
                <div>
                  <dt>评价步骤</dt>
                  <dd>{stepCount} 个</dd>
                </div>
                <div>
                  <dt>持有期</dt>
                  <dd>{horizon} 天</dd>
                </div>
                <div>
                  <dt>分位组</dt>
                  <dd>{nQuantiles} 组</dd>
                </div>
                <div>
                  <dt>Decay</dt>
                  <dd>{decay} 天</dd>
                </div>
                <div>
                  <dt>评价期间</dt>
                  <dd>{windowText}</dd>
                </div>
                <div>
                  <dt>结果门槛</dt>
                  <dd>{gate.conditions.length ? `${gate.conditions.length} 个条件` : '未设置'}</dd>
                </div>
              </dl>
            </div>

            {hasProxyFactor && (
              <Alert
                className="run-config-aside-note"
                type="warning"
                showIcon
                message="包含代理数据口径因子"
                description="结果会保留该因子的代理说明。与真实输入因子比较前请先确认数据口径一致。"
              />
            )}
            {usesGrossReturns && (
              <Alert
                className="run-config-aside-note"
                type="info"
                showIcon
                message="组合收益口径：毛收益"
                description={
                  supportsNetReturns
                    ? '当前流水线使用「分组收益」。需要扣除显性交易费率时，请换成「分组净收益」，再运行累计收益和图表步骤。'
                    : '当前流水线使用未扣交易成本的分组收益。'
                }
              />
            )}
            {usesNetReturns && (
              <Alert
                className="run-config-aside-note"
                type="info"
                showIcon
                message="组合收益口径：净收益"
                description="会扣除已配置的显性交易费率；滑点、买卖价差和市场冲击不在此口径内。"
              />
            )}
            {kind === 'funnel' && (
              <Alert
                className="run-config-aside-note"
                type="info"
                showIcon
                message="漏斗的组合收益阶段使用毛收益"
                description="第 3 阶段要求分组收益作为组合检验输入。提交前可在流水线模板页核对每个阶段的执行顺序。"
              />
            )}
          </div>
        </aside>
      </div>
    </Drawer>
  )
}
