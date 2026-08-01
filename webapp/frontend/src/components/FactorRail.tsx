import { RocketOutlined, SwapOutlined } from '@ant-design/icons'
import { Button, Checkbox, Empty, Select, Skeleton, Space, Tooltip, Typography } from 'antd'
import { ReactNode, useMemo } from 'react'
import { Factor } from '../api/client'
import { formatMetric } from '../lib/metrics'

export type RailSort = 'name' | 'return' | 'sharpe' | 'fitness' | 'recent'

const SORT_OPTIONS: Array<{ value: RailSort; label: string }> = [
  { value: 'name', label: '按名称' },
  { value: 'return', label: '按年化收益率' },
  { value: 'sharpe', label: '按 60日 Sharpe' },
  { value: 'fitness', label: '按 Fitness' },
  { value: 'recent', label: '按最近评价' },
]

const SORT_METRIC: Record<Exclude<RailSort, 'name' | 'recent'>, string> = {
  return: 'top_group_annualized_return',
  sharpe: 'gn_rolling_sharpe_60_median',
  fitness: 'fitness',
}

export const factorKey = (factor: Factor) => `${factor.batch_id}/${factor.factor_name}`

const metricOf = (factor: Factor, key: string): number | null => {
  const value = factor.latest_run?.result?.[key]
  return typeof value === 'number' ? value : null
}

/**
 * 列表的显示顺序：先按项目名，组内按当前排序字段。
 * 页面自动选中「第一个因子」时也用它，保证选中的就是看到的第一行。
 */
export function orderFactors(factors: Factor[], sort: RailSort): Array<readonly [string, Factor[]]> {
  const byProject = new Map<string, Factor[]>()
  for (const factor of factors) {
    const rows = byProject.get(factor.project) || []
    rows.push(factor)
    byProject.set(factor.project, rows)
  }
  const compare = (left: Factor, right: Factor) => {
    if (sort === 'name') return left.factor_name.localeCompare(right.factor_name, 'zh-CN')
    if (sort === 'recent') {
      const la = left.latest_run?.finished_at || left.latest_run?.created_at || ''
      const ra = right.latest_run?.finished_at || right.latest_run?.created_at || ''
      return ra.localeCompare(la)
    }
    const key = SORT_METRIC[sort]
    const lv = metricOf(left, key)
    const rv = metricOf(right, key)
    if (lv === null && rv === null) return left.factor_name.localeCompare(right.factor_name, 'zh-CN')
    if (lv === null) return 1
    if (rv === null) return -1
    return rv - lv
  }
  return [...byProject.entries()]
    .sort(([left], [right]) => left.localeCompare(right, 'zh-CN'))
    .map(([project, rows]) => [project, [...rows].sort(compare)] as const)
}

/** 一行一个因子：名字 + 最近一次评价的状态与主指标，够用来决定点开哪一个。 */
function FactorRow({
  factor,
  sort,
  active,
  checked,
  onActivate,
  onToggle,
}: {
  factor: Factor
  sort: RailSort
  active: boolean
  checked: boolean
  onActivate: () => void
  onToggle: (next: boolean) => void
}) {
  const status = factor.latest_run?.status
  const metricKey = sort === 'name' || sort === 'recent' ? 'top_group_annualized_return' : SORT_METRIC[sort]
  const value = metricOf(factor, metricKey)
  return (
    <div className={`factor-row${active ? ' active' : ''}`}>
      <Checkbox
        checked={checked}
        onChange={(event) => onToggle(event.target.checked)}
        aria-label={`选择 ${factor.factor_name}`}
      />
      <button type="button" className="factor-row-select" onClick={onActivate} aria-current={active}>
        <span className="factor-row-name">
          <span className={`factor-row-dot ${status || 'none'}`} aria-hidden="true" />
          <span className="factor-row-label">{factor.factor_name}</span>
        </span>
        <span className="factor-row-meta">
          <span className="factor-row-category">{factor.factor_category || factor.implementation_set}</span>
          <span className={`factor-row-metric ${value === null ? '' : value > 0 ? 'good' : value < 0 ? 'bad' : ''}`}>
            {value === null ? (factor.latest_run ? '无该指标' : '未评价') : formatMetric(metricKey, value)}
          </span>
        </span>
      </button>
    </div>
  )
}

export function FactorRail({
  factors,
  loading,
  sort,
  onSortChange,
  activeKey,
  onActivate,
  checkedKeys,
  onCheckedChange,
  onRunSelected,
  onCompareSelected,
  comparableCount,
  header,
  footer,
}: {
  factors: Factor[]
  loading: boolean
  sort: RailSort
  onSortChange: (next: RailSort) => void
  activeKey: string | null
  onActivate: (factor: Factor) => void
  checkedKeys: string[]
  onCheckedChange: (next: string[]) => void
  onRunSelected: () => void
  onCompareSelected: () => void
  comparableCount: number
  header?: ReactNode
  footer?: ReactNode
}) {
  const groups = useMemo(() => orderFactors(factors, sort), [factors, sort])
  const checkedSet = new Set(checkedKeys)
  const toggle = (factor: Factor, next: boolean) => {
    const key = factorKey(factor)
    onCheckedChange(next ? [...checkedKeys, key] : checkedKeys.filter((item) => item !== key))
  }

  return (
    <aside className="factor-rail" aria-label="因子列表">
      {header}
      <div className="factor-rail-meta">
        <Typography.Text type="secondary">{factors.length} 个因子</Typography.Text>
        <Select
          size="small"
          variant="borderless"
          value={sort}
          onChange={onSortChange}
          options={SORT_OPTIONS}
          popupMatchSelectWidth={200}
          className="factor-rail-sort"
        />
      </div>
      {checkedKeys.length > 0 && (
        <div className="factor-rail-selection" role="status">
          <Typography.Text strong>已选 {checkedKeys.length} 个</Typography.Text>
          <Space size={6}>
            <Button size="small" type="primary" icon={<RocketOutlined />} onClick={onRunSelected}>
              提交测试
            </Button>
            <Tooltip title={comparableCount >= 2 ? '对比所选因子的最近一次成功结果' : '至少需要 2 个已有成功结果的因子'}>
              <Button size="small" icon={<SwapOutlined />} disabled={comparableCount < 2} onClick={onCompareSelected}>
                对比
              </Button>
            </Tooltip>
            <Button size="small" type="text" onClick={() => onCheckedChange([])}>
              清空
            </Button>
          </Space>
        </div>
      )}
      <div className="factor-rail-list">
        {loading ? (
          <Skeleton active title={false} paragraph={{ rows: 12 }} className="factor-rail-skeleton" />
        ) : groups.length ? (
          groups.map(([project, rows]) => (
            <section className="factor-rail-group" key={project}>
              <header className="factor-rail-group-head">
                <Typography.Text strong>{project}</Typography.Text>
                <Typography.Text type="secondary">{rows.length}</Typography.Text>
              </header>
              {rows.map((factor) => {
                const key = factorKey(factor)
                return (
                  <FactorRow
                    key={key}
                    factor={factor}
                    sort={sort}
                    active={activeKey === key}
                    checked={checkedSet.has(key)}
                    onActivate={() => onActivate(factor)}
                    onToggle={(next) => toggle(factor, next)}
                  />
                )
              })}
            </section>
          ))
        ) : (
          <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="没有符合筛选条件的因子" />
        )}
      </div>
      {footer && <div className="factor-rail-foot">{footer}</div>}
    </aside>
  )
}
