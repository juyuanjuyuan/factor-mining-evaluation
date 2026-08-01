import { useQuery } from '@tanstack/react-query'
import { Alert, Button, Card, Empty, Select, Space, Table, Tooltip, Typography } from 'antd'
import { useMemo, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { api } from '../api/client'
import { OverlayChart } from '../components/charts/SeriesChart'
import { METRIC_GROUPS, bestValue, formatMetric, metricSpec, shortTime } from '../lib/metrics'

export default function Compare() {
  // 选中的运行放在 URL 上：可直接分享对比链接，返回本页时选择不丢失
  const [searchParams, setSearchParams] = useSearchParams()
  const selected = useMemo(
    () => searchParams.get('runs')?.split(',').filter(Boolean) ?? [],
    [searchParams],
  )
  const setSelected = (ids: string[]) => {
    setSearchParams(
      (current) => {
        const next = new URLSearchParams(current)
        if (ids.length) next.set('runs', ids.join(','))
        else next.delete('runs')
        return next
      },
      { replace: true },
    )
  }
  const runs = useQuery({ queryKey: ['successful-runs'], queryFn: () => api.runs('succeeded') })
  // 带着 2 个以上运行进入本页（列表页跳转或分享链接）时直接生成对比；
  // 用查询而不是 mutation：结果进缓存，离开再返回时对比结果直接复原
  const [submittedIds, setSubmittedIds] = useState<string[]>(() =>
    selected.length >= 2 ? selected : [],
  )
  const comparison = useQuery({
    queryKey: ['comparison', submittedIds],
    queryFn: () => api.compare(submittedIds),
    enabled: submittedIds.length >= 2,
  })

  const metricRows = (comparison.data?.metric_rows || [])
    .map((row) => ({ ...row, spec: metricSpec(row.metric) }))
    .sort((a, b) => {
      const ga = METRIC_GROUPS.indexOf(a.spec.group as (typeof METRIC_GROUPS)[number])
      const gb = METRIC_GROUPS.indexOf(b.spec.group as (typeof METRIC_GROUPS)[number])
      return (ga === -1 ? 99 : ga) - (gb === -1 ? 99 : gb)
    })

  return (
    <>
      <div className="page-heading">
        <div>
          <Typography.Title level={2}>多因子对比</Typography.Title>
          <Typography.Text type="secondary">
            选择多个成功的运行结果，横向对比指标与曲线（建议持有期、分组数、Decay 一致）
          </Typography.Text>
        </div>
      </div>
      <Card className="surface-card">
        <Space.Compact style={{ width: '100%' }}>
          <Select
            mode="multiple"
            maxCount={20}
            style={{ flex: 1 }}
            placeholder="搜索并选择成功的运行结果（可多选）"
            value={selected}
            onChange={setSelected}
            optionFilterProp="label"
            options={(runs.data || []).map((run) => ({
              value: run.id,
              label: `${run.factor_name} · H${run.horizon} · D${Number(run.result?.decay ?? run.run_params?.decay ?? 1)} · ${run.methods.length}方法 · ${shortTime(run.finished_at)}`,
            }))}
          />
          <Button
            type="primary"
            disabled={selected.length < 2}
            loading={comparison.isFetching}
            onClick={() => setSubmittedIds(selected)}
          >
            生成对比
          </Button>
        </Space.Compact>
        {selected.length === 1 && (
          <Typography.Text type="secondary" className="hint-line">
            再选至少 1 个运行结果即可对比
          </Typography.Text>
        )}
      </Card>
      {comparison.isError && (
        <Alert
          type="error"
          showIcon
          className="section-card"
          title="生成对比失败"
          description={comparison.error.message}
        />
      )}
      {!comparison.data && !comparison.isFetching && !comparison.isError && (
        <Card className="surface-card section-card">
          <Empty description="选择 2 个以上运行结果，点击「生成对比」" />
        </Card>
      )}
      {comparison.data && (
        <>
          <Card
            className="surface-card section-card"
            title="指标对照"
            extra={<Typography.Text type="secondary">高亮 = 该指标下表现最优（已按指标方向判断）</Typography.Text>}
          >
            <Table
              rowKey="metric"
              size="small"
              pagination={false}
              scroll={{ x: true, y: 560 }}
              dataSource={metricRows}
              columns={[
                {
                  title: '指标',
                  fixed: 'left',
                  width: 240,
                  render: (_, row) => (
                    <Tooltip title={row.spec.tooltip}>
                      <span>
                        <Typography.Text>{row.spec.label}</Typography.Text>
                        <span className="metric-group-chip">{row.spec.group}</span>
                      </span>
                    </Tooltip>
                  ),
                },
                ...comparison.data.runs.map((run) => ({
                  title: run.factor_name,
                  render: (_: unknown, row: (typeof metricRows)[number]) => {
                    const value = row.values[run.id]
                    const numeric = Object.values(row.values).filter(
                      (item): item is number => typeof item === 'number',
                    )
                    const best =
                      typeof value === 'number' && bestValue(row.metric, numeric) === value
                    return (
                      <Typography.Text className={best ? 'best-value' : ''}>
                        {formatMetric(row.metric, value)}
                      </Typography.Text>
                    )
                  },
                })),
              ]}
            />
          </Card>
          {comparison.data.curves.top_quantile.length > 0 && (
            <Card className="surface-card section-card">
              <OverlayChart curves={comparison.data.curves.top_quantile} title="最高分位组累计收益" />
            </Card>
          )}
          {comparison.data.curves.cumulative_ic.length > 0 && (
            <Card className="surface-card section-card">
              <OverlayChart curves={comparison.data.curves.cumulative_ic} title="累计 IC" />
            </Card>
          )}
        </>
      )}
    </>
  )
}
