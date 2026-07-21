/**
 * Screening-gate helpers shared by the config builder and the result views.
 *
 * A gate is a set of threshold conditions on produced metrics (e.g.
 * 最高组分段回撤(60日)最差值 ≥ -0.40). The catalog of gate-able metrics is
 * derived from the same metric dictionary the rest of the app uses, so labels,
 * units and formatting stay consistent everywhere.
 */
import type { GateCondition, GateOp } from '../api/client'
import { METRIC_SPECS, MetricFormat, formatMetric, metricSpec } from './metrics'

export type GateMetricOption = {
  value: string
  label: string
  group: string
  format: MetricFormat
}

export const GATE_OPERATORS: { value: GateOp; symbol: string; label: string }[] = [
  { value: 'gte', symbol: '≥', label: '大于等于' },
  { value: 'lte', symbol: '≤', label: '小于等于' },
  { value: 'gt', symbol: '>', label: '大于' },
  { value: 'lt', symbol: '<', label: '小于' },
  { value: 'between', symbol: '区间', label: '介于两者之间' },
]

const ROLLING_WINDOWS = ['20', '60', '252']
// Only the statistics the engine emits per family (see rolling_sharpe.py and rolling_drawdown.py):
// drawdown → worst/median, sharpe → pos_share/min/median. Anything else would
// always read as "指标缺失" in the gate.
const ROLLING_STATS_BY_KIND: Record<string, string[]> = {
  drawdown: ['worst', 'median'],
  sharpe: ['pos_share', 'min', 'median'],
}

// The metric picker leads with the sections a portfolio screen cares about.
const GATE_GROUP_ORDER = ['滚动风险', '头部组合', '分位组收益', 'IC 分析', '显著性检验', '数据质量']

function rollingMetricKeys(): string[] {
  const keys: string[] = []
  for (const [kind, stats] of Object.entries(ROLLING_STATS_BY_KIND)) {
    for (const window of ROLLING_WINDOWS) {
      for (const stat of stats) {
        keys.push(`gn_rolling_${kind}_${window}_${stat}`)
      }
    }
  }
  return keys
}

let cachedCatalog: GateMetricOption[] | null = null

/** Every metric a gate condition may target, with its display format. */
export function gateMetricCatalog(): GateMetricOption[] {
  if (cachedCatalog) return cachedCatalog
  const keys = new Set<string>()
  for (const [key, spec] of Object.entries(METRIC_SPECS)) {
    if (spec.format === 'text') continue
    if (spec.group === '运行信息') continue
    if (spec.label.startsWith('旧版')) continue // deprecated net-cost fields
    keys.add(key)
  }
  for (const key of rollingMetricKeys()) keys.add(key)
  cachedCatalog = [...keys].map((key) => {
    const spec = metricSpec(key)
    return { value: key, label: spec.label, group: spec.group, format: spec.format }
  })
  return cachedCatalog
}

/** Grouped option list for an antd Select (search by label). */
export function gateMetricSelectGroups(): { label: string; options: { value: string; label: string }[] }[] {
  const byGroup = new Map<string, GateMetricOption[]>()
  for (const option of gateMetricCatalog()) {
    if (!byGroup.has(option.group)) byGroup.set(option.group, [])
    byGroup.get(option.group)!.push(option)
  }
  const groups = [...byGroup.keys()].sort((a, b) => {
    const ia = GATE_GROUP_ORDER.indexOf(a)
    const ib = GATE_GROUP_ORDER.indexOf(b)
    return (ia === -1 ? 99 : ia) - (ib === -1 ? 99 : ib)
  })
  return groups.map((group) => ({
    label: group,
    options: byGroup
      .get(group)!
      .sort((a, b) => a.label.localeCompare(b.label, 'zh'))
      .map((option) => ({ value: option.value, label: option.label })),
  }))
}

export function gateMetricFormat(metric: string): MetricFormat {
  return metricSpec(metric).format
}

/**
 * Drawdowns are stored as ≤0 fractions but read as positive magnitudes ("回撤
 * 20%"). For these metrics the gate takes a positive threshold with 不超过/至少,
 * which we translate to the correct signed comparison so the backend stays
 * generic. "不超过 M" ⇔ raw ≥ -M (op gte); "至少 M" ⇔ raw ≤ -M (op lte).
 */
export function isDrawdownMagnitude(metric: string): boolean {
  return gateMetricFormat(metric) === 'drawdown_pct'
}

export const DRAWDOWN_GATE_OPERATORS: { value: GateOp; label: string }[] = [
  { value: 'gte', label: '不超过' },
  { value: 'lte', label: '至少' },
]

export function operatorSymbol(op: string): string {
  return GATE_OPERATORS.find((item) => item.value === op)?.symbol ?? op
}

/** Convert a raw stored threshold into the unit the input box shows. */
export function thresholdToInput(format: MetricFormat, raw: number | null | undefined): number | null {
  if (raw === null || raw === undefined || Number.isNaN(raw)) return null
  if (format === 'percent') return Number((raw * 100).toPrecision(12))
  if (format === 'drawdown_pct') return Number((Math.abs(raw) * 100).toPrecision(12))
  return raw
}

/** Convert what the user typed back into the raw metric unit. */
export function thresholdFromInput(format: MetricFormat, input: number | null | undefined): number | null {
  if (input === null || input === undefined || Number.isNaN(input)) return null
  if (format === 'percent') return input / 100
  // Positive magnitude percent → the ≤0 fraction the metric is stored in.
  if (format === 'drawdown_pct') return -(Math.abs(input) / 100)
  return input
}

export function thresholdStep(format: MetricFormat): number {
  switch (format) {
    case 'percent':
    case 'drawdown_pct':
      return 1
    case 'pvalue':
      return 0.01
    case 'int':
      return 1
    default:
      return 0.01
  }
}

export function thresholdSuffix(format: MetricFormat): string {
  return format === 'percent' || format === 'drawdown_pct' ? '%' : ''
}

/** One-line human description, e.g. "ICIR ≥ 0.0500" or "最高组分段回撤(60日)最差值 不超过 40.00%". */
export function describeGateCondition(condition: GateCondition): string {
  const label = metricSpec(condition.metric).label
  if (isDrawdownMagnitude(condition.metric) && condition.op !== 'between') {
    const relation = condition.op === 'lte' ? '至少' : '不超过'
    return `${label} ${relation} ${formatMetric(condition.metric, condition.value)}`
  }
  if (condition.op === 'between') {
    const low = formatMetric(condition.metric, condition.value)
    const high = formatMetric(condition.metric, condition.value2 ?? condition.value)
    return `${label} ∈ [${low}, ${high}]`
  }
  return `${label} ${operatorSymbol(condition.op)} ${formatMetric(condition.metric, condition.value)}`
}
