import {
  DeleteOutlined,
  ExperimentOutlined,
  PlayCircleOutlined,
  PlusOutlined,
  SaveOutlined,
} from '@ant-design/icons'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import {
  Alert,
  Button,
  Card,
  Divider,
  Empty,
  Flex,
  Input,
  InputNumber,
  Popconfirm,
  Select,
  Slider,
  Space,
  Tag,
  Tooltip,
  Typography,
  message,
} from 'antd'
import { useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { api, ModelTerm, ModelTest, ModelTestInput, Run, Template } from '../api/client'
import { PipelineBuilder } from '../components/PipelineBuilder'
import { methodLabel, shortTime } from '../lib/metrics'

type Draft = ModelTestInput

const CUSTOM_TEMPLATE = -1
const DEFAULT_TRAIN_END = '2021-12-13'

const newDraft = (methods: string[] = []): Draft => ({
  model_name: '',
  terms: [],
  train_start: '',
  train_end: '',
  test_start: '',
  test_end: '',
  horizon: 1,
  n_quantiles: 10,
  methods,
  training_method: 'winsorized_zscore_ridge',
  training_params: {},
})

const statusColor = (run?: Run | null) => {
  if (!run) return undefined
  if (run.status === 'succeeded') return 'green'
  if (run.status === 'failed' || run.status === 'cancelled') return 'red'
  if (run.status === 'running') return 'blue'
  return 'gold'
}

function dayPosition(days: string[], value: string) {
  if (!value) return Math.max(0, Math.floor(days.length / 2) - 1)
  const exact = days.indexOf(value)
  if (exact >= 0) return exact
  const next = days.findIndex((day) => day >= value)
  if (next === -1) return days.length - 1
  return Math.max(0, next - 1)
}

function TrainTestSplitSlider({
  days,
  min,
  max,
  split,
  disabled,
  onChange,
}: {
  days: string[]
  min: number
  max: number
  split: number
  disabled: boolean
  onChange: (value: number) => void
}) {
  const safeSplit = Math.max(min, Math.min(split, max))
  const trainingDays = safeSplit + 1
  const testingDays = days.length - safeSplit - 1
  return (
    <div className="model-split-slider">
      <div className="model-split-slider-head">
        <div className="model-split-window training">
          <Typography.Text strong>训练集</Typography.Text>
          <Typography.Text type="secondary">{days[0]} 至 {days[safeSplit]}</Typography.Text>
          <Tag color="green">{trainingDays} 个交易日</Tag>
        </div>
        <div className="model-split-boundary">
          <Typography.Text type="secondary">分界日</Typography.Text>
          <Tag>{days[safeSplit]}</Tag>
        </div>
        <div className="model-split-window testing">
          <Typography.Text strong>测试集</Typography.Text>
          <Typography.Text type="secondary">{days[safeSplit + 1]} 至 {days[days.length - 1]}</Typography.Text>
          <Tag color="volcano">{testingDays} 个交易日</Tag>
        </div>
      </div>
      <Slider
        min={min}
        max={max}
        step={1}
        value={safeSplit}
        disabled={disabled}
        tooltip={{ formatter: (position) => (typeof position === 'number' ? days[position] : '') }}
        onChange={(next: number) => onChange(next)}
      />
      <Typography.Text type="secondary" className="model-split-instruction">
        拖动这个分界点：左侧全部用于训练，右侧全部用于样本外测试。
      </Typography.Text>
    </div>
  )
}

function RunPhase({ label, run }: { label: string; run?: Run | null }) {
  if (!run) {
    return (
      <div className="model-phase empty">
        <Typography.Text strong>{label}</Typography.Text>
        <Typography.Text type="secondary">尚未运行</Typography.Text>
      </div>
    )
  }
  return (
    <div className="model-phase">
      <Flex justify="space-between" align="center" gap={8}>
        <Typography.Text strong>{label}</Typography.Text>
        <Tag color={statusColor(run)}>{run.status === 'succeeded' ? '已完成' : run.status}</Tag>
      </Flex>
      <Typography.Text type="secondary">
        {run.result?.sample_start_day && run.result?.sample_end_day
          ? `${String(run.result.sample_start_day)} ~ ${String(run.result.sample_end_day)}`
          : '正在等待或计算所选样本'}
      </Typography.Text>
      <Link to={`/runs/${run.id}`}>查看本次评价结果</Link>
    </div>
  )
}

export default function ModelTests() {
  const queryClient = useQueryClient()
  const models = useQuery({
    queryKey: ['models'],
    queryFn: api.models,
    refetchInterval: 2500,
  })
  const factors = useQuery({ queryKey: ['factors'], queryFn: api.factors })
  const timeline = useQuery({
    queryKey: ['market-timeline'],
    queryFn: api.marketTimeline,
    staleTime: Infinity,
  })
  const methodCatalog = useQuery({ queryKey: ['methods'], queryFn: api.methods })
  const templates = useQuery({
    queryKey: ['templates'],
    queryFn: api.templates,
    staleTime: Infinity,
  })
  const trainingMethods = useQuery({
    queryKey: ['model-training-methods'],
    queryFn: api.modelTrainingMethods,
    staleTime: Infinity,
  })
  const [activeId, setActiveId] = useState<string | null>(null)
  const [draft, setDraft] = useState<Draft>(() => newDraft())
  const [templateId, setTemplateId] = useState<number>(CUSTOM_TEMPLATE)
  const [pipelineValid, setPipelineValid] = useState(false)
  const [saving, setSaving] = useState(false)
  const [deletingId, setDeletingId] = useState<string | null>(null)

  const active = useMemo(
    () => (models.data || []).find((model) => model.id === activeId) || null,
    [activeId, models.data],
  )
  const defaultMethods = useMemo(
    () => (methodCatalog.data || []).filter((method) => method.is_default).map((method) => method.name),
    [methodCatalog.data],
  )
  const methodTemplates = useMemo(
    () => (templates.data || []).filter((template) => template.kind === 'methods'),
    [templates.data],
  )
  const locked = Boolean(active?.locked)
  const trainingPending = Boolean(
    active?.training_run && ['queued', 'running'].includes(active.training_run.status),
  )
  const configurationDisabled = locked || trainingPending
  const selectedTrainingMethod = (trainingMethods.data || []).find(
    (method) => method.name === draft.training_method,
  )
  const factorOptions = useMemo(
    () =>
      (factors.data || []).map((factor) => ({
        value: `${factor.batch_id}/${factor.factor_name}`,
        label: `${factor.factor_name} · ${factor.batch_name}`,
        batch_id: factor.batch_id,
        factor_name: factor.factor_name,
      })),
    [factors.data],
  )
  const tradingDays = timeline.data?.trading_days || []
  const splitPosition = useMemo(() => {
    if (!tradingDays.length) return null
    const minimum = draft.horizon + 1
    const maximum = tradingDays.length - draft.horizon - 3
    return Math.max(minimum, Math.min(dayPosition(tradingDays, draft.train_end), maximum))
  }, [draft.horizon, draft.train_end, tradingDays])

  useEffect(() => {
    if (!activeId && !draft.methods.length && defaultMethods.length) {
      setDraft((previous) => ({ ...previous, methods: defaultMethods }))
    }
  }, [activeId, defaultMethods, draft.methods.length])

  useEffect(() => {
    if (activeId || !tradingDays.length || draft.train_start || draft.train_end || draft.test_start || draft.test_end) return
    const minimum = draft.horizon + 1
    const maximum = tradingDays.length - draft.horizon - 3
    const split = Math.max(minimum, Math.min(maximum, dayPosition(tradingDays, DEFAULT_TRAIN_END)))
    setDraft((previous) => ({
      ...previous,
      train_start: tradingDays[0],
      train_end: tradingDays[split],
      test_start: tradingDays[split + 1],
      test_end: tradingDays[tradingDays.length - 1],
    }))
  }, [activeId, draft.horizon, draft.test_end, draft.test_start, draft.train_end, draft.train_start, tradingDays])

  const beginNew = () => {
    setActiveId(null)
    setDraft(newDraft(defaultMethods))
    setTemplateId(CUSTOM_TEMPLATE)
    setPipelineValid(defaultMethods.length > 0)
  }

  const loadModel = (model: ModelTest) => {
    setActiveId(model.id)
    // A model persists the resolved sequence, not a mutable template reference.
    setTemplateId(CUSTOM_TEMPLATE)
    setDraft({
      model_name: model.model_name,
      terms: model.terms.map((term) => ({
        batch_id: term.batch_id,
        factor_name: term.factor_name,
        weight: term.weight,
      })),
      train_start: model.train_start,
      train_end: model.train_end,
      test_start: model.test_start,
      test_end: model.test_end,
      horizon: model.horizon,
      n_quantiles: model.n_quantiles,
      methods: model.methods,
      training_method: model.training_method,
      training_params: model.training_params,
    })
  }

  useEffect(() => {
    if (!active?.fit_result?.expression) return
    setDraft((previous) => ({
      ...previous,
      terms: active.terms.map((term) => ({
        batch_id: term.batch_id,
        factor_name: term.factor_name,
        weight: term.weight,
      })),
    }))
  }, [active?.fit_result?.expression])

  const changeTerm = (index: number, patch: Partial<ModelTerm>) => {
    setDraft((previous) => ({
      ...previous,
      terms: previous.terms.map((term, termIndex) => (termIndex === index ? { ...term, ...patch } : term)),
    }))
  }

  const setSplitPosition = (split: number) => {
    setDraft((previous) => ({
      ...previous,
      train_start: tradingDays[0],
      train_end: tradingDays[split],
      test_start: tradingDays[split + 1],
      test_end: tradingDays[tradingDays.length - 1],
    }))
  }

  const applyTemplate = (template: Template) => {
    setTemplateId(template.id)
    setDraft((previous) => ({ ...previous, methods: template.methods }))
    // Method templates are validated and dependency-complete before they are saved.
    setPipelineValid(template.methods.length > 0)
  }

  const addTerm = () => {
    const present = new Set(draft.terms.map((term) => `${term.batch_id}/${term.factor_name}`))
    const candidate = factorOptions.find((option) => !present.has(option.value))
    if (!candidate) {
      message.warning('因子库中没有更多可加入的已提交因子')
      return
    }
    setDraft((previous) => ({
      ...previous,
      terms: [...previous.terms, { batch_id: candidate.batch_id, factor_name: candidate.factor_name, weight: 1 }],
    }))
  }

  const payload = (): ModelTestInput => ({
    ...draft,
    model_name: draft.model_name.trim(),
    terms: draft.terms.map((term) => ({
      batch_id: term.batch_id,
      factor_name: term.factor_name,
      weight: Number(term.weight),
    })),
  })

  const persist = async () => {
    if (!draft.terms.length) throw new Error('请至少加入一个因子库因子')
    if (!pipelineValid) throw new Error('请先补全评价流水线中缺失的前置依赖')
    setSaving(true)
    try {
      const saved = activeId ? await api.updateModel(activeId, payload()) : await api.createModel(payload())
      setActiveId(saved.id)
      setDraft((previous) => ({ ...previous, methods: saved.methods }))
      await queryClient.invalidateQueries({ queryKey: ['models'] })
      return saved
    } finally {
      setSaving(false)
    }
  }

  const saveDraft = async () => {
    try {
      await persist()
      message.success('模型配置已保存；可继续调整训练方法或模型参数')
    } catch (error) {
      message.error((error as Error).message)
    }
  }

  const runTraining = async () => {
    try {
      const saved = await persist()
      await api.trainModel(saved.id)
      await queryClient.invalidateQueries({ queryKey: ['models'] })
      message.success(
        saved.training_method === 'manual_weights'
          ? '训练集评价已提交；完成后仍可调整手动权重'
          : '训练集拟合与评价已提交；完成后会显示学习到的权重',
      )
    } catch (error) {
      message.error((error as Error).message)
    }
  }

  const runTesting = async () => {
    if (!active) return
    try {
      await api.testModel(active.id)
      await queryClient.invalidateQueries({ queryKey: ['models'] })
      message.success('样本外测试已提交，模型配置现已锁定')
    } catch (error) {
      message.error((error as Error).message)
    }
  }

  const removeModel = async (model: ModelTest) => {
    setDeletingId(model.id)
    try {
      await api.deleteModel(model.id)
      if (activeId === model.id) beginNew()
      await queryClient.invalidateQueries({ queryKey: ['models'] })
      message.success('模型会话已删除；关联的评价任务仍保留在任务队列')
    } catch (error) {
      message.error((error as Error).message)
    } finally {
      setDeletingId(null)
    }
  }

  const selectableFactorOptions = factorOptions.map(({ value, label }) => ({ value, label }))
  const readyToSave = Boolean(
    tradingDays.length
    && splitPosition !== null
    && draft.model_name.trim()
    && draft.terms.length
    && draft.methods.length
    && pipelineValid,
  )

  return (
    <>
      <Flex justify="space-between" align="center" className="page-heading" wrap gap={16}>
        <div className="page-heading-copy">
          <Typography.Title level={2}>模型测试</Typography.Title>
          <Typography.Text type="secondary">
            用因子库的已提交因子组成线性模型，在训练集确定参数后，以冻结配置进行一次样本外验证。
          </Typography.Text>
        </div>
        <Button icon={<PlusOutlined />} onClick={beginNew}>新建线性模型</Button>
      </Flex>

      <Alert
        className="model-leakage-note"
        type="info"
        showIcon
        title="训练与测试严格隔离"
        description="模型只允许在训练阶段调整。提交样本外测试后，因子定义、权重、日期区间和评价流水线都会锁定；测试窗口末尾会按持有期裁去无法在窗口内平仓的信号日。"
      />

      <div className="model-testing-layout">
        <Card className="surface-card model-session-list" title="模型会话">
          <Button block type={activeId ? 'default' : 'primary'} icon={<PlusOutlined />} onClick={beginNew}>
            新建草稿
          </Button>
          <Divider />
          {models.isLoading ? (
            <Typography.Text type="secondary">正在读取模型会话…</Typography.Text>
          ) : !(models.data || []).length ? (
            <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="尚无模型会话" />
          ) : (
            <div className="model-session-items" role="listbox" aria-label="模型测试会话">
              {(models.data || []).map((model) => (
                <div
                  key={model.id}
                  className={`model-session-item${model.id === activeId ? ' active' : ''}`}
                >
                  <button type="button" className="model-session-select" onClick={() => loadModel(model)}>
                    <Typography.Text strong>{model.model_name}</Typography.Text>
                    <span>{model.locked ? '已锁定样本外测试' : model.training_run ? '已有训练结果' : '训练草稿'}</span>
                    <small>{shortTime(model.updated_at)}</small>
                  </button>
                  <Popconfirm
                    title="删除这个模型会话？"
                    description="模型配置会被删除，已经提交的训练和测试任务仍保留在任务队列。"
                    onConfirm={() => removeModel(model)}
                    okText="确认删除"
                    cancelText="取消"
                  >
                    <Tooltip title="删除模型会话">
                      <Button
                        aria-label={`删除模型会话 ${model.model_name}`}
                        className="model-session-delete"
                        danger
                        type="text"
                        icon={<DeleteOutlined />}
                        loading={deletingId === model.id}
                      />
                    </Tooltip>
                  </Popconfirm>
                </div>
              ))}
            </div>
          )}
        </Card>

        <div className="model-testing-main">
          {locked && (
            <Alert
              className="model-locked-note"
              type="warning"
              showIcon
              title="此模型已经提交样本外测试，配置已锁定"
              description="若要研究新的权重，请新建一个模型会话并重新使用训练集；不要将已见的测试表现用于当前模型调参。"
            />
          )}
          <Card
            className="surface-card"
            title={active ? `线性模型配置 · ${active.model_name}` : '新建线性模型'}
            extra={
              <Space>
                {active && <Tag color={locked ? 'gold' : 'blue'}>{locked ? '已锁定' : '可训练调参'}</Tag>}
                <Button icon={<SaveOutlined />} loading={saving} disabled={configurationDisabled || !readyToSave} onClick={saveDraft}>
                  保存草稿
                </Button>
                <Button type="primary" icon={<ExperimentOutlined />} loading={saving} disabled={configurationDisabled || !readyToSave} onClick={runTraining}>
                  {selectedTrainingMethod?.requires_fitting ? '拟合并评价训练集' : '运行训练集'}
                </Button>
              </Space>
            }
          >
            <div className="model-form-grid">
              <label>
                <span>模型名称</span>
                <Input value={draft.model_name} disabled={configurationDisabled} placeholder="例如：价值质量线性模型 v1" onChange={(event) => setDraft((previous) => ({ ...previous, model_name: event.target.value }))} />
              </label>
              <label>
                <span>模型形式</span>
                <Input value={`线性加权 · ${selectedTrainingMethod?.label || '加载训练方式中'}`} disabled />
              </label>
            </div>

            <Divider titlePlacement="start">训练集与测试集（信号日）</Divider>
            {timeline.isLoading ? (
              <Typography.Text type="secondary">正在读取行情数据的交易日范围…</Typography.Text>
            ) : timeline.isError ? (
              <Alert type="error" showIcon title="无法读取可用交易日范围" description={timeline.error.message} />
            ) : splitPosition !== null && timeline.data ? (
              <div className="model-timeline">
                <div className="model-timeline-summary">
                  <Typography.Text strong>可用行情时间轴</Typography.Text>
                  <Typography.Text type="secondary">
                    {timeline.data.start_day} 至 {timeline.data.end_day} · {timeline.data.count.toLocaleString()} 个交易日
                  </Typography.Text>
                </div>
                <TrainTestSplitSlider
                  days={tradingDays}
                  split={splitPosition}
                  min={draft.horizon + 1}
                  max={tradingDays.length - draft.horizon - 3}
                  disabled={configurationDisabled}
                  onChange={setSplitPosition}
                />
              </div>
            ) : (
              <Alert type="warning" showIcon title="当前行情数据没有可用于时间切片的交易日" />
            )}
            <Typography.Paragraph type="secondary" className="model-field-hint">
              这条时间轴只使用真实交易日。拖动分界点会同时更新两个样本的日期；每段至少保留“持有期 + 2”个交易日，系统会在末尾再裁去无法在本段内完成开平仓的信号日。
            </Typography.Paragraph>
            <div className="model-parameter-grid">
              <label>
                <span>持有期（天）</span>
                <InputNumber
                  min={1}
                  max={60}
                  value={draft.horizon}
                  disabled={configurationDisabled}
                  onChange={(value) => setDraft((previous) => ({ ...previous, horizon: Number(value ?? 1) }))}
                />
              </label>
              <label>
                <span>分位组数</span>
                <InputNumber
                  min={2}
                  max={20}
                  value={draft.n_quantiles}
                  disabled={configurationDisabled}
                  onChange={(value) => setDraft((previous) => ({ ...previous, n_quantiles: Number(value ?? 10) }))}
                />
              </label>
            </div>

            <Divider titlePlacement="start">因子库线性分量</Divider>
            <div className="model-training-method-grid">
              <label>
                <span>训练方式</span>
                <Select
                  value={draft.training_method}
                  disabled={configurationDisabled}
                  loading={trainingMethods.isLoading}
                  options={(trainingMethods.data || []).map((method) => ({
                    value: method.name,
                    label: method.label,
                  }))}
                  onChange={(name) => {
                    const method = (trainingMethods.data || []).find((candidate) => candidate.name === name)
                    setDraft((previous) => ({
                      ...previous,
                      training_method: name,
                      training_params: Object.fromEntries(
                        (method?.parameters || []).map((parameter) => [parameter.name, parameter.default]),
                      ),
                    }))
                  }}
                />
              </label>
              {(selectedTrainingMethod?.parameters || []).map((parameter) => (
                <label key={parameter.name}>
                  <span>{parameter.label}</span>
                  <InputNumber
                    min={parameter.minimum}
                    max={parameter.maximum}
                    step={parameter.step}
                    value={draft.training_params[parameter.name] ?? parameter.default}
                    disabled={configurationDisabled}
                    onChange={(value) => setDraft((previous) => ({
                      ...previous,
                      training_params: {
                        ...previous.training_params,
                        [parameter.name]: Number(value ?? parameter.default),
                      },
                    }))}
                  />
                  <small>{parameter.description}</small>
                </label>
              ))}
            </div>
            {selectedTrainingMethod && (
              <Typography.Paragraph type="secondary" className="model-field-hint">
                {selectedTrainingMethod.description}
              </Typography.Paragraph>
            )}
            <div className="model-term-list">
              {draft.terms.map((term, index) => (
                <div className="model-term-row" key={`${term.batch_id}/${term.factor_name}-${index}`}>
                  <span className="model-term-index">{index + 1}</span>
                  <Select
                    showSearch
                    optionFilterProp="label"
                    value={`${term.batch_id}/${term.factor_name}`}
                    disabled={configurationDisabled}
                    options={selectableFactorOptions}
                    placeholder="选择因子库因子"
                    onChange={(value) => {
                      const option = factorOptions.find((candidate) => candidate.value === value)
                      if (option) changeTerm(index, { batch_id: option.batch_id, factor_name: option.factor_name })
                    }}
                  />
                  <InputNumber
                    value={term.weight}
                    disabled={configurationDisabled || !selectedTrainingMethod?.term_weight_editable}
                    step={0.1}
                    precision={6}
                    addonBefore={selectedTrainingMethod?.term_weight_editable ? '权重' : '拟合权重'}
                    onChange={(value) => changeTerm(index, { weight: Number(value ?? 0) })}
                  />
                  <Tooltip title="移除该分量">
                    <Button disabled={configurationDisabled} type="text" icon={<DeleteOutlined />} onClick={() => setDraft((previous) => ({ ...previous, terms: previous.terms.filter((_, termIndex) => termIndex !== index) }))} />
                  </Tooltip>
                </div>
              ))}
              {!draft.terms.length && <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="先从因子库加入一个已提交因子" />}
              <Button disabled={configurationDisabled || !factorOptions.length} icon={<PlusOutlined />} onClick={addTerm}>加入因子</Button>
            </div>
            <Typography.Paragraph type="secondary" className="model-field-hint">
              {draft.training_method === 'winsorized_zscore_ridge'
                ? '每个因子每天先按分位数去极值，再做横截面 z-score，从而保留受控后的原始幅度信息；测试阶段复用冻结权重。'
                : '每个因子先在每个交易日转换为横截面分位秩，再进行线性相加。负权重表示反向使用该因子；自动训练方式只读取训练窗口，测试阶段复用冻结权重。'}
            </Typography.Paragraph>
            {active?.fit_result && (
              <div className="model-fit-summary">
                <Flex justify="space-between" align="center" gap={8} wrap>
                  <Typography.Text strong>最近一次训练结果</Typography.Text>
                  <Tag color="green">
                    {(trainingMethods.data || []).find((method) => method.name === active.fit_result?.method)?.label || active.fit_result.method}
                  </Tag>
                </Flex>
                <div className="model-fit-weights">
                  {active.fit_result.terms.map((term) => (
                    <span key={`${term.batch_id}/${term.factor_name}`}>
                      <Typography.Text type="secondary">{term.factor_name}</Typography.Text>
                      <Typography.Text code>{Number(term.weight).toPrecision(6)}</Typography.Text>
                    </span>
                  ))}
                </div>
                {active.fit_result.diagnostics.trading_day_count != null && (
                  <Typography.Text type="secondary">
                    {String(active.fit_result.diagnostics.sample_start_day)} 至 {String(active.fit_result.diagnostics.sample_end_day)} · {' '}
                    {Number(active.fit_result.diagnostics.trading_day_count).toLocaleString()} 个交易日 · {' '}
                    {Number(active.fit_result.diagnostics.observation_count).toLocaleString()} 个股票日观测
                  </Typography.Text>
                )}
              </div>
            )}

            <Divider titlePlacement="start">评价方式</Divider>
            <label className="model-template-picker">
              <span>评价模板</span>
              <Select
                value={templateId}
                disabled={configurationDisabled}
                loading={templates.isLoading}
                options={[
                  ...methodTemplates.map((template) => ({ value: template.id, label: template.name })),
                  { value: CUSTOM_TEMPLATE, label: '自定义流水线' },
                ]}
                onChange={(nextTemplateId) => {
                  if (nextTemplateId === CUSTOM_TEMPLATE) {
                    setTemplateId(CUSTOM_TEMPLATE)
                    return
                  }
                  const template = methodTemplates.find((item) => item.id === nextTemplateId)
                  if (template) applyTemplate(template)
                }}
              />
            </label>
            <Typography.Paragraph type="secondary" className="model-field-hint">
              仅显示“方法组合”模板。选中后会复制其中的有序评价方法；后续拖拽或增删方法会切换为自定义。模型保存和训练/测试执行的始终是当前展开后的方法顺序。
              {' '}<Link to="/pipelines">管理流水线模板</Link>
            </Typography.Paragraph>
            <div className={configurationDisabled ? 'model-pipeline-locked' : undefined}>
              <PipelineBuilder
                value={draft.methods}
                onValidityChange={setPipelineValid}
                onChange={(methods) => {
                  setTemplateId(CUSTOM_TEMPLATE)
                  setDraft((previous) => ({ ...previous, methods }))
                }}
              />
            </div>
            {draft.methods.length > 0 && (
              <Typography.Paragraph type="secondary" className="model-field-hint">
                当前顺序：{draft.methods.map(methodLabel).join(' → ')}
              </Typography.Paragraph>
            )}
          </Card>

          <Card
            className="surface-card section-card"
            title="两阶段结果"
            extra={
              <Button
                type="primary"
                icon={<PlayCircleOutlined />}
                disabled={!active?.can_run_testing}
                onClick={runTesting}
              >
                执行样本外测试
              </Button>
            }
          >
            {!active ? (
              <Typography.Text type="secondary">先保存并运行训练集；训练成功后才可提交样本外测试。</Typography.Text>
            ) : (
              <div className="model-phase-grid">
                <RunPhase label="训练集：调参与评价" run={active.training_run} />
                <RunPhase label="测试集：冻结后的样本外验证" run={active.testing_run} />
              </div>
            )}
          </Card>
        </div>
      </div>
    </>
  )
}
