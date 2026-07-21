import { useQuery } from '@tanstack/react-query'
import {
  Alert,
  Button,
  Divider,
  Drawer,
  Form,
  InputNumber,
  Segmented,
  Space,
  Steps,
  Tag,
  Typography,
  message,
} from 'antd'
import { useEffect, useState } from 'react'
import { api, Factor, FunnelStage, GateSpec, Template } from '../api/client'
import { STAGE_INFO, methodLabel } from '../lib/metrics'
import { PipelineBuilder } from './PipelineBuilder'
import { GateBuilder } from './GateBuilder'

const CUSTOM = -1

const EMPTY_GATE: GateSpec = { conditions: [], match: 'all' }

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
  const methodCatalog = useQuery({ queryKey: ['methods'], queryFn: api.methods })
  const funnelStageCatalog = useQuery({ queryKey: ['funnel-stages'], queryFn: api.funnelStages })
  const templates = useQuery({ queryKey: ['templates'], queryFn: api.templates })
  const [form] = Form.useForm()
  const horizon = Form.useWatch('horizon', form) ?? 1
  const nQuantiles = Form.useWatch('n_quantiles', form) ?? 10

  // 打开时默认选中第一个内置模板（默认流水线）
  useEffect(() => {
    if (open && templates.data?.length && templateId === CUSTOM && !methods.length) {
      applyTemplate(templates.data[0])
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, templates.data])

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
    const horizon = template.params?.horizon
    if (typeof horizon === 'number') {
      form.setFieldValue('horizon', horizon)
    }
    const nQuantiles = template.params?.n_quantiles
    if (typeof nQuantiles === 'number') {
      form.setFieldValue('n_quantiles', nQuantiles)
    }
    const significance = template.params?.significance_level
    if (typeof significance === 'number') {
      form.setFieldValue('significance_level', significance)
    }
  }

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
        significance_level: values.significance_level,
        gate:
          kind === 'evaluate' && gateConditions.length
            ? { conditions: gateConditions, match: gate.match }
            : undefined,
      })
      message.success('任务已提交，可继续提交其他任务，或前往任务队列查看进度')
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

  return (
    <Drawer
      title={
        tagSelection
          ? `回测配置：${tagSelection.tags.join('、')}`
          : factors.length === 1
          ? `回测配置 · ${factors[0]?.factor_name}`
          : `回测配置 · ${factors.length} 个因子`
      }
      size="min(780px, 100vw)"
      open={open}
      onClose={onClose}
      extra={
        <Button type="primary" loading={submitting} disabled={!canSubmit} onClick={submit}>
          提交任务
        </Button>
      }
    >
      {tagSelection && (
        <Typography.Paragraph type="secondary" className="tag-run-drawer-note">
          将按标签{tagSelection.match === 'all' ? '同时包含' : '包含任一'}的规则，在提交时重新匹配因子，避免分页或页面刷新造成遗漏。
        </Typography.Paragraph>
      )}
      {factors.length > 1 && (
        <div className="drawer-factor-list">
          {factors.slice(0, 12).map((factor) => (
            <Tag key={`${factor.batch_id}/${factor.factor_name}`}>{factor.factor_name}</Tag>
          ))}
          {factors.length > 12 && <Tag>… 共 {factors.length} 个</Tag>}
        </div>
      )}
      <div className="run-config-summary" aria-label="本次评价设置概览">
        <div>
          <span>评价对象</span>
          <strong>{factors.length} 个因子</strong>
        </div>
        <div>
          <span>持有期</span>
          <strong>{horizon} 天</strong>
        </div>
        <div>
          <span>分位组</span>
          <strong>{nQuantiles} 组</strong>
        </div>
        <div>
          <span>任务类型</span>
          <strong>{kind === 'funnel' ? '四阶段漏斗' : '自定义评价'}</strong>
        </div>
      </div>
      {hasProxyFactor && (
        <Alert
          className="drawer-callout"
          type="warning"
          showIcon
          message="本次包含代理数据口径因子"
          description="结果会保留该因子的代理说明。与真实输入因子的结果比较时，请先确认数据口径一致。"
        />
      )}
      <Typography.Text type="secondary">流程模板</Typography.Text>
      <div className="template-picker">
        <Segmented
          block
          value={templateId}
          onChange={(next) => {
            if (next === CUSTOM) {
              chooseCustom()
              return
            }
            const template = templates.data?.find((item) => item.id === next)
            if (template) applyTemplate(template)
          }}
          options={[
            ...(templates.data || []).map((template) => ({
              label: template.name,
              value: template.id,
            })),
            { label: '自定义', value: CUSTOM },
          ]}
        />
      </div>
      <Divider className="drawer-divider" />
      <Form
        form={form}
        layout="vertical"
        initialValues={{ horizon: 1, n_quantiles: 10, significance_level: 0.05 }}
      >
        <Space size="large">
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
        {kind === 'funnel' ? (
          <>
            <Typography.Paragraph type="secondary">
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
          <Form.Item label="评价流水线（按顺序执行，可拖拽调整、可重复添加同一方法）">
            <PipelineBuilder
              value={methods}
              onValidityChange={setPipelineValid}
              onChange={(next) => {
                setMethods(next)
                setTemplateId(CUSTOM)
              }}
            />
          </Form.Item>
        )}
        {kind === 'evaluate' && (
          <div className="gate-section">
            <div className="gate-section-head">
              <Typography.Text strong>结果门槛（可选）</Typography.Text>
              {gate.conditions.length > 0 && (
                <Tag className="gate-count-tag">
                  {gate.conditions.length} 个条件 · {gate.match === 'all' ? '全部满足' : '满足任一'}
                </Tag>
              )}
            </div>
            <Typography.Paragraph type="secondary" className="gate-section-hint">
              为本次回测设定过关标准。任务完成后，每个因子会被自动标记「通过 / 未过关」，方便在一批因子里快速筛出达标的。门槛只影响标记，不会跳过任何计算。
            </Typography.Paragraph>
            <GateBuilder value={gate} onChange={setGate} />
          </div>
        )}
        {usesGrossReturns && (
          <Alert
            className="return-basis-note"
            type="info"
            showIcon
            message="组合收益口径：毛收益"
            description={
              supportsNetReturns
                ? '当前流水线使用“分组收益”。若需要在分位组收益中扣除显性交易费率，请将其替换为“分组净收益”，再运行累计收益和图表步骤。'
                : '当前流水线使用未扣交易成本的分组收益。'
            }
          />
        )}
        {usesNetReturns && (
          <Alert
            className="return-basis-note"
            type="info"
            showIcon
            message="组合收益口径：净收益"
            description="当前流水线会扣除已配置的显性交易费率；滑点、买卖价差和市场冲击不在此口径内。"
          />
        )}
        {kind === 'funnel' && (
          <Alert
            className="return-basis-note"
            type="info"
            showIcon
            message="漏斗的组合收益阶段使用毛收益"
            description="第 3 阶段要求分组收益作为组合检验输入。提交前可在模板页核对每个阶段的执行顺序。"
          />
        )}
      </Form>
    </Drawer>
  )
}
