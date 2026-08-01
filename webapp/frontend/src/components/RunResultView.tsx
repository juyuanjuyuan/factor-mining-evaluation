import { CheckCircleFilled, CloseCircleFilled } from '@ant-design/icons'
import { useQuery } from '@tanstack/react-query'
import {
  Alert,
  Anchor,
  Button,
  Card,
  Collapse,
  Descriptions,
  Empty,
  Segmented,
  Skeleton,
  Spin,
  Table,
  Tag,
  Tooltip,
  Typography,
} from 'antd'
import { useState } from 'react'
import { Link } from 'react-router-dom'
import { api, Detail, GateConditionResult, MarketCycleBackground } from '../api/client'
import { HoldingAuditPanel } from './HoldingAuditPanel'
import { LazyRender } from './LazyRender'
import { DetailChart, SummaryBar } from './charts/SeriesChart'
import { describeGateCondition } from '../lib/gate'
import {
  KEY_METRICS,
  METRIC_GROUPS,
  PARAM_KEYS,
  detailTitle,
  detailColumnTitle,
  formatMetric,
  metricSpec,
  methodLabel,
  shortTime,
  sortDetails,
  splitVersion,
} from '../lib/metrics'

type DetailDisplayKind = 'series' | 'summary' | 'table'

function detailDisplayKind(name: string): DetailDisplayKind | null {
  const base = splitVersion(name).base
  if (base === 'group_returns') return 'summary'
  if (base === 'yearly_fitness') return 'table'
  if (
    base === 'ic' ||
    base === 'ic_horizon_decay' ||
    base === 'ic_peak_decay' ||
    base === 'ic_trend_filter' ||
    base === 'ic_trend_filter_mean_10' ||
    base === 'cumulative_returns' ||
    base.startsWith('rolling_') ||
    base.includes('autocovariances')
  ) return 'series'
  return null
}

function formatYearlyFitnessValue(column: string, value: unknown): string {
  if (typeof value !== 'number' || !Number.isFinite(value)) return '暂无'
  if (column === 'trading_days') return String(Math.round(value))
  if (['cumulative_net_return', 'annualized_net_return', 'mean_daily_one_way_turnover', 'annual_max_drawdown'].includes(column)) {
    return `${(value * 100).toFixed(2)}%`
  }
  return value.toFixed(4)
}

function YearlyFitnessTable({ detail }: { detail: Detail }) {
  const visibleColumns = detail.columns.filter(
    (column) => !['drawdown_penalty_lambda', 'fitness_radicand'].includes(column),
  )
  const rows = detail.index.map((year, rowIndex) => ({
    key: year,
    year,
    ...Object.fromEntries(
      detail.columns.map((column, columnIndex) => [column, detail.data[rowIndex]?.[columnIndex] ?? null]),
    ),
  }))
  const columns = [
    { title: '自然年', dataIndex: 'year', key: 'year', width: 96 },
    ...visibleColumns.map((column) => ({
      title: detailColumnTitle(column),
      dataIndex: column,
      key: column,
      align: 'right' as const,
      render: (value: unknown) => formatYearlyFitnessValue(column, value),
    })),
  ]
  return (
    <>
      <Typography.Paragraph type="secondary" style={{ marginBottom: 12 }}>
        最高分位组的净收益、单边换手与年度最大回撤按信号日期所属自然年汇总；首尾不足一年的收益已按实际交易日数年化。回撤惩罚在根号外扣除。
      </Typography.Paragraph>
      <Table
        size="small"
        columns={columns}
        dataSource={rows}
        pagination={false}
        scroll={{ x: 'max-content' }}
      />
    </>
  )
}

function DetailPanel({
  runId,
  name,
  cycleBackgrounds,
}: {
  runId: string
  name: string
  cycleBackgrounds?: MarketCycleBackground[]
}) {
  const detail = useQuery({ queryKey: ['detail', runId, name], queryFn: () => api.detail(runId, name) })
  if (detail.isLoading)
    return (
      <Card className="section-card">
        <Skeleton active />
      </Card>
    )
  if (!detail.data) return null
  const displayKind = detailDisplayKind(name)
  if (!displayKind) return null
  return (
    <Card className="surface-card section-card" title={detailTitle(name)}>
      {displayKind === 'series' ? (
        <DetailChart detail={detail.data} cycleBackgrounds={cycleBackgrounds} />
      ) : displayKind === 'table' ? (
        <YearlyFitnessTable detail={detail.data} />
      ) : (
        <SummaryBar detail={detail.data} />
      )}
    </Card>
  )
}

function InnerDetail({
  runId,
  name,
  cycleBackgrounds,
}: {
  runId: string
  name: string
  cycleBackgrounds?: MarketCycleBackground[]
}) {
  const detail = useQuery({ queryKey: ['detail', runId, name], queryFn: () => api.detail(runId, name) })
  if (detail.isLoading) return <Skeleton active />
  if (!detail.data) return null
  return <DetailChart detail={detail.data} cycleBackgrounds={cycleBackgrounds} />
}

function RollingPanel({
  runId,
  family,
  names,
  cycleBackgrounds,
}: {
  runId: string
  family: 'rolling_sharpe' | 'rolling_drawdown'
  names: string[]
  cycleBackgrounds?: MarketCycleBackground[]
}) {
  const windows = names
    .filter((name) => name.startsWith(`${family}_`))
    .map((name) => name.slice(family.length + 1))
    .sort((a, b) => parseInt(a, 10) - parseInt(b, 10) || a.localeCompare(b))
  const [window, setWindow] = useState(windows.includes('60') ? '60' : windows[0])
  if (!window) return null
  const windowDays = Number.parseInt(window, 10)
  return (
    <Card
      className="surface-card section-card"
      title={
        family === 'rolling_sharpe'
          ? `分段年化夏普（${window} 日非重叠窗口）`
          : `分段最大回撤（${window} 日非重叠窗口）`
      }
      extra={
        <Segmented
          value={window}
          options={windows.map((value) => {
            const { base, version } = splitVersion(value)
            return { value, label: version > 1 ? `${base} 日（第${version}次）` : `${value} 日` }
          })}
          onChange={(value) => setWindow(String(value))}
        />
      }
    >
      <Typography.Paragraph type="secondary" style={{ marginBottom: 12 }}>
        每个点按窗口结束日标记，使用该日及其前 {windowDays - 1} 个有效交易日的收益
        （共 {windowDays} 日，含结束日）；下一点从随后的非重叠区块开始。
      </Typography.Paragraph>
      <InnerDetail runId={runId} name={`${family}_${window}`} cycleBackgrounds={cycleBackgrounds} />
    </Card>
  )
}

// 关键指标较上次的变化量：percent 格式以百分点表示，其余沿用指标精度
function formatDelta(key: string, delta: number): string {
  const { format } = metricSpec(key)
  const sign = delta > 0 ? '+' : ''
  if (format === 'percent') return `${sign}${(delta * 100).toFixed(2)}pp`
  if (Math.abs(delta) >= 1000) {
    return `${sign}${delta.toLocaleString(undefined, { maximumFractionDigits: 2 })}`
  }
  return `${sign}${delta.toFixed(4)}`
}

// 按指标方向判断变化好坏：higher 越大越好、lower 越小越好、abs 绝对值越大越好
function deltaTone(key: string, current: number, prior: number): '' | 'good' | 'bad' {
  if (current === prior) return ''
  const { direction } = metricSpec(key)
  if (direction === 'higher') return current > prior ? 'good' : 'bad'
  if (direction === 'lower') return current < prior ? 'good' : 'bad'
  if (direction === 'abs') {
    if (Math.abs(current) === Math.abs(prior)) return ''
    return Math.abs(current) > Math.abs(prior) ? 'good' : 'bad'
  }
  return ''
}

function GateResultCard({
  outcome,
  explanation,
  conditions,
}: {
  outcome?: string
  explanation?: string
  conditions: GateConditionResult[]
}) {
  const passed = outcome === 'passed'
  const passedCount = conditions.filter((item) => item.passed).length
  return (
    <Card className="surface-card section-card gate-result-card" title="结果门槛">
      <div className={`gate-verdict ${passed ? 'pass' : 'fail'}`}>
        {passed ? <CheckCircleFilled /> : <CloseCircleFilled />}
        <div>
          <div className="gate-verdict-title">{passed ? '通过' : '未过关'}</div>
          <div className="gate-verdict-sub">
            {explanation || `${passedCount}/${conditions.length} 项达标`}
          </div>
        </div>
      </div>
      <div className="gate-result-list">
        {conditions.map((condition, index) => (
          <div key={index} className={`gate-result-row ${condition.passed ? 'pass' : 'fail'}`}>
            <span className="gate-result-icon">
              {condition.passed ? <CheckCircleFilled /> : <CloseCircleFilled />}
            </span>
            <span className="gate-result-cond">{describeGateCondition(condition)}</span>
            <span className="gate-result-actual">
              实际：
              {condition.missing ? (
                <Typography.Text type="warning">指标缺失</Typography.Text>
              ) : (
                <strong>{formatMetric(condition.metric, condition.actual)}</strong>
              )}
            </span>
          </div>
        ))}
      </div>
    </Card>
  )
}

/**
 * 一次运行的完整结果主体（不含页面标题与队列翻页）。
 * 结果详情页和因子库右侧面板共用同一份渲染，避免两处指标口径漂移。
 */
export function RunResultView({ runId, showAnchors = true }: { runId: string; showAnchors?: boolean }) {
  const run = useQuery({
    queryKey: ['run', runId],
    queryFn: () => api.run(runId),
    refetchInterval: (query) =>
      ['queued', 'running'].includes(query.state.data?.status || '') ? 1500 : false,
  })
  // 同因子的历史成功运行：给关键指标提供「较上次」对比基准
  const factorName = run.data?.factor_name
  const priorRuns = useQuery({
    queryKey: ['factor-runs', factorName || ''],
    queryFn: () => api.runs('succeeded', factorName!),
    enabled: Boolean(factorName),
  })
  if (run.isLoading) return <Skeleton active paragraph={{ rows: 10 }} />
  if (!run.data) return <Empty description="结果不存在" />
  const data = run.data
  const result = data.result || {}
  const configuredDecay = Number(result.decay ?? data.run_params?.decay ?? 1)
  const decay = Number.isInteger(configuredDecay) && configuredDecay >= 1 ? configuredDecay : 1

  const heroMetrics = KEY_METRICS.filter(
    (key) => typeof result[key] === 'number' || typeof result[key] === 'boolean',
  )
  const grouped = new Map<string, Array<[string, unknown]>>()
  for (const [key, value] of Object.entries(result)) {
    if (PARAM_KEYS.has(key)) continue
    if (!['number', 'boolean'].includes(typeof value)) continue
    const group = metricSpec(key).group
    if (!grouped.has(group)) grouped.set(group, [])
    grouped.get(group)!.push([key, value])
  }
  const orderedDetails = sortDetails(
    (data.detail_names || []).filter(
      (name) =>
        !name.startsWith('rolling_sharpe_') &&
        !name.startsWith('rolling_drawdown_') &&
        detailDisplayKind(name) !== null,
    ),
  )
  const active = ['queued', 'running'].includes(data.status)
  const gateConditions = Array.isArray(data.gate_value) ? data.gate_value : null
  const returnBasis = data.methods.includes('quantile_net_returns')
    ? {
        label: '净收益',
        tooltip: '已扣除配置中的显性交易费率，不包含滑点、买卖价差和市场冲击。',
      }
    : data.methods.includes('quantile_returns')
      ? {
          label: '毛收益',
          tooltip: '分位组收益未扣除交易成本。',
        }
      : { label: '尚未计算组合收益', tooltip: '当前运行未包含分位组收益步骤。' }
  const metricCount = [...grouped.values()].reduce((sum, list) => sum + list.length, 0)
  // 「较上次」基准：同因子、同持有期、同分组数、同 Decay 的最近一次更早的成功运行；
  // 收益口径（return_definition）不一致的运行不做对比
  const priorRun = (priorRuns.data || []).find(
    (item) =>
      item.id !== data.id &&
      item.horizon === data.horizon &&
      item.n_quantiles === data.n_quantiles &&
      Number(item.result?.decay ?? item.run_params?.decay ?? 1) === decay &&
      Boolean(item.finished_at) &&
      (!data.finished_at || item.finished_at! < data.finished_at) &&
      (!item.result?.return_definition ||
        !result.return_definition ||
        item.result.return_definition === result.return_definition),
  )
  const detailNames = data.detail_names || []
  const hasRollingSharpe = detailNames.some((name) => name.startsWith('rolling_sharpe_'))
  const hasRollingDrawdown = detailNames.some((name) => name.startsWith('rolling_drawdown_'))
  const hasHoldingAudit = data.status === 'succeeded' && data.methods.includes('holding_audit')
  // 页面很长：右侧锚点目录直达任一区块
  const anchorItems = [
    ...(gateConditions ? [{ key: 'gate', href: '#section-gate', title: '结果门槛' }] : []),
    ...(heroMetrics.length > 0 ? [{ key: 'hero', href: '#section-hero', title: '关键指标' }] : []),
    { key: 'context', href: '#section-context', title: '运行设置' },
    ...(grouped.size > 0 ? [{ key: 'metrics', href: '#section-metrics', title: '全部指标' }] : []),
    ...(hasHoldingAudit ? [{ key: 'holding-audit', href: '#section-holding-audit', title: '历史回测持仓' }] : []),
    ...orderedDetails.map((name) => ({ key: name, href: `#detail-${name}`, title: detailTitle(name) })),
    ...(hasRollingSharpe
      ? [{ key: 'rolling-sharpe', href: '#section-rolling-sharpe', title: '分段年化夏普' }]
      : []),
    ...(hasRollingDrawdown
      ? [{ key: 'rolling-drawdown', href: '#section-rolling-drawdown', title: '分段最大回撤' }]
      : []),
  ]

  return (
    <div className="run-detail-layout">
      <div className="run-detail-main">
        {active && (
          <Alert
            type="info"
            showIcon
            icon={<Spin size="small" />}
            title={data.status === 'running' ? '正在计算，页面会自动刷新…' : '排队等待中，页面会自动刷新…'}
            className="section-card"
          />
        )}
        {data.error && (
          <Alert
            type="error"
            showIcon
            title="运行失败"
            description={<pre className="error-text">{data.error}</pre>}
            className="section-card"
          />
        )}

        {gateConditions && (
          <div id="section-gate">
            <GateResultCard
              outcome={data.gate_outcome}
              explanation={data.gate_explanation}
              conditions={gateConditions}
            />
          </div>
        )}

        {heroMetrics.length > 0 && (
          <div id="section-hero">
            {priorRun && (
              <Typography.Text type="secondary" className="hero-compare-note">
                指标下方的变化量对比 {shortTime(priorRun.finished_at)} 的上一次成功运行
                （同持有期与分组数；样本区间与评价方法可能不同，收益类差值仅供趋势参考）
              </Typography.Text>
            )}
            <div className="hero-metrics">
              {heroMetrics.map((key) => {
                const value = result[key]
                const numeric = typeof value === 'number' ? value : value ? 1 : 0
                const direction = metricSpec(key).direction
                const tone =
                  key === 'nw_ic_p_value'
                    ? numeric < 0.05
                      ? 'good'
                      : 'bad'
                    : typeof value === 'boolean'
                      ? value
                        ? 'good'
                        : 'bad'
                      : direction === 'lower'
                        ? ''
                        : numeric > 0
                          ? 'good'
                          : numeric < 0
                            ? 'bad'
                            : ''
                const priorValue = priorRun?.result?.[key]
                const hasDelta = typeof value === 'number' && typeof priorValue === 'number'
                return (
                  <Tooltip key={key} title={metricSpec(key).tooltip}>
                    <div className="hero-metric">
                      <div className="hero-metric-label">{metricSpec(key).label}</div>
                      <div className={`hero-metric-value ${tone}`}>{formatMetric(key, value)}</div>
                      {hasDelta && (
                        <div className={`hero-metric-delta ${deltaTone(key, value, priorValue)}`}>
                          {formatDelta(key, value - priorValue)} vs 上次
                        </div>
                      )}
                    </div>
                  </Tooltip>
                )
              })}
            </div>
          </div>
        )}

        <section className="run-context section-card" aria-label="本次运行设置" id="section-context">
          <Descriptions column={3} size="small">
            <Descriptions.Item label="持有期">{data.horizon} 天</Descriptions.Item>
            <Descriptions.Item label="分位组数">{data.n_quantiles}</Descriptions.Item>
            <Descriptions.Item label="Decay">{decay} 天</Descriptions.Item>
            <Descriptions.Item label="组合收益">
              <Tooltip title={returnBasis.tooltip}>
                <Tag className={`return-basis-tag ${returnBasis.label === '净收益' ? 'net' : ''}`}>
                  {returnBasis.label}
                </Tag>
              </Tooltip>
            </Descriptions.Item>
            <Descriptions.Item label="请求的评价期间" span={3}>
              {data.run_params?.signal_start && data.run_params?.signal_end
                ? `${String(data.run_params.signal_start)} ~ ${String(data.run_params.signal_end)}`
                : '全部历史'}
            </Descriptions.Item>
            <Descriptions.Item label="实际信号样本" span={3}>
              {result.sample_start_day && result.sample_end_day
                ? `${String(result.sample_start_day)} ~ ${String(result.sample_end_day)}`
                : result.start_day && result.end_day
                  ? `${String(result.start_day)} ~ ${String(result.end_day)}`
                  : '暂无'}
            </Descriptions.Item>
            <Descriptions.Item label="评价方法" span={3}>
              {data.methods.map((method, index) => (
                <Tooltip key={`${method}-${index}`} title={method}>
                  <Tag>{methodLabel(method)}</Tag>
                </Tooltip>
              ))}
            </Descriptions.Item>
            <Descriptions.Item label="表达式" span={3}>
              <Typography.Text code copyable>
                {data.expression}
              </Typography.Text>
            </Descriptions.Item>
          </Descriptions>
        </section>

        {hasHoldingAudit && (
          <div id="section-holding-audit">
            <HoldingAuditPanel runId={runId} />
          </div>
        )}

        {grouped.size > 0 && (
          <div id="section-metrics">
            <Collapse
              className="collapse-card section-card"
              items={[
                {
                  key: 'metrics',
                  label: `全部指标（${metricCount} 项 · ${grouped.size} 组）`,
                  children: METRIC_GROUPS.filter((group) => grouped.has(group)).map((group) => (
                    <div key={group} className="metric-group">
                      <Typography.Text type="secondary" className="metric-group-title">
                        {group}
                      </Typography.Text>
                      <div className="metric-grid">
                        {grouped.get(group)!.map(([key, value]) => (
                          <Tooltip key={key} title={metricSpec(key).tooltip}>
                            <div className="metric-cell">
                              <div className="metric-cell-label">{metricSpec(key).label}</div>
                              <div className="metric-cell-value">{formatMetric(key, value)}</div>
                            </div>
                          </Tooltip>
                        ))}
                      </div>
                    </div>
                  )),
                },
              ]}
            />
          </div>
        )}

        {orderedDetails.map((name) => (
          <div key={name} id={`detail-${name}`}>
            <LazyRender>
              <DetailPanel runId={runId} name={name} cycleBackgrounds={data.market_cycle_backgrounds} />
            </LazyRender>
          </div>
        ))}
        {hasRollingSharpe && (
          <div id="section-rolling-sharpe">
            <LazyRender>
              <RollingPanel
                runId={runId}
                family="rolling_sharpe"
                names={detailNames}
                cycleBackgrounds={data.market_cycle_backgrounds}
              />
            </LazyRender>
          </div>
        )}
        {hasRollingDrawdown && (
          <div id="section-rolling-drawdown">
            <LazyRender>
              <RollingPanel
                runId={runId}
                family="rolling_drawdown"
                names={detailNames}
                cycleBackgrounds={data.market_cycle_backgrounds}
              />
            </LazyRender>
          </div>
        )}

        {data.status === 'succeeded' && (
          <div className="section-card">
            <Link to={`/compare?runs=${data.id}`}>
              <Button>加入多因子对比</Button>
            </Link>
          </div>
        )}
      </div>
      {showAnchors && anchorItems.length > 1 && (
        <div className="run-anchor-rail" aria-label="页面目录">
          <Anchor
            affix={false}
            targetOffset={92}
            // 直接定位而不用平滑滚动动画：长页跳转更干脆，也避免部分
            // WebView 挂起 rAF 时动画根本不执行
            onClick={(event, link) => {
              event.preventDefault()
              const target = document.getElementById(link.href.replace('#', ''))
              if (target) {
                const top = target.getBoundingClientRect().top + window.scrollY - 92
                window.scrollTo(0, Math.max(top, 0))
              }
            }}
            items={anchorItems}
          />
        </div>
      )}
    </div>
  )
}
