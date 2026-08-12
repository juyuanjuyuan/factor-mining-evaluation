import { useQuery } from '@tanstack/react-query'
import { Alert, Card, Descriptions, Input, Skeleton, Table, Typography } from 'antd'
import { useEffect, useState } from 'react'
import { api, HoldingAuditPosition } from '../api/client'

function percent(value: number | null | undefined, digits = 2): string {
  return typeof value === 'number' && Number.isFinite(value) ? `${(value * 100).toFixed(digits)}%` : '—'
}

function number(value: number | null | undefined, digits = 3): string {
  return typeof value === 'number' && Number.isFinite(value) ? value.toLocaleString(undefined, { maximumFractionDigits: digits }) : '—'
}

function factorValue(value: number): string {
  if (!Number.isFinite(value)) return '—'
  return Math.abs(value) >= 10_000 ? value.toExponential(3) : value.toFixed(4)
}

export function HoldingAuditPanel({ runId }: { runId: string }) {
  const dates = useQuery({ queryKey: ['holding-audit-dates', runId], queryFn: () => api.holdingAuditDates(runId) })
  const [signalDay, setSignalDay] = useState('')
  const [page, setPage] = useState(1)
  const availableDays = dates.data?.dates || []
  const firstDay = availableDays[0]?.signal_day
  const lastDay = availableDays[availableDays.length - 1]?.signal_day

  useEffect(() => {
    if (!signalDay && lastDay) setSignalDay(lastDay)
  }, [lastDay, signalDay])

  const holding = useQuery({
    queryKey: ['holding-audit', runId, signalDay, page],
    queryFn: () => api.holdingAudit(runId, signalDay, page),
    enabled: Boolean(signalDay),
  })

  const columns = [
    { title: '公司名称', dataIndex: 'security_name', key: 'security_name', width: 132, fixed: 'left' as const, render: (value: string | null) => value || '—' },
    { title: '证券代码', dataIndex: 'security_code', key: 'security_code', width: 105 },
    { title: '组内排名', dataIndex: 'rank_in_top_group', key: 'rank_in_top_group', align: 'right' as const, width: 92 },
    { title: '目标权重', dataIndex: 'target_weight', key: 'target_weight', align: 'right' as const, width: 105, render: (value: number) => percent(value, 4) },
    { title: '中性化后因子', dataIndex: 'factor_value', key: 'factor_value', align: 'right' as const, width: 138, render: (value: number) => factorValue(value) },
    { title: '一级行业', dataIndex: 'industry_l1_name', key: 'industry_l1_name', width: 132, render: (value: string | null) => value || '—' },
    { title: '行业代码', dataIndex: 'industry_l1_code', key: 'industry_l1_code', width: 112, render: (value: string | null) => value || '—' },
    { title: '市值（亿元）', dataIndex: 'market_cap_yi', key: 'market_cap_yi', align: 'right' as const, width: 126, render: (value: number | null) => number(value, 1) },
    { title: '买入开盘价', dataIndex: 'entry_open', key: 'entry_open', align: 'right' as const, width: 110, render: (value: number | null) => number(value, 3) },
    { title: '卖出开盘价', dataIndex: 'exit_open', key: 'exit_open', align: 'right' as const, width: 110, render: (value: number | null) => number(value, 3) },
    { title: '开盘收益', dataIndex: 'forward_open_return', key: 'forward_open_return', align: 'right' as const, width: 105, render: (value: number) => percent(value) },
  ]

  return (
    <Card className="surface-card section-card" title="历史回测持仓（最高分位）">
      <Typography.Paragraph type="secondary" style={{ marginBottom: 12 }}>
        展示运行中已启用的 <code>holding_audit</code> 产物：先完成中性化、信号日与买入日特殊状态/买入日涨跌停过滤和净分组收益，再按信号日读取最高分位的等权目标权重。仅加载所选日期的一页仓位，不预加载全样本。
      </Typography.Paragraph>
      {dates.isLoading ? (
        <Skeleton active />
      ) : dates.isError ? (
        <Alert type="error" showIcon message="仓单日期索引无法读取" description={dates.error.message} />
      ) : !availableDays.length ? (
        <Alert type="warning" showIcon message="没有可审计的最高分位持仓" />
      ) : (
        <>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap', marginBottom: 12 }}>
            <Typography.Text strong>信号日</Typography.Text>
            <Input
              type="date"
              value={signalDay}
              min={firstDay}
              max={lastDay}
              style={{ width: 164 }}
              onChange={(event) => {
                setSignalDay(event.target.value)
                setPage(1)
              }}
            />
            <Typography.Text type="secondary">可查区间：{firstDay} 至 {lastDay}（只接受产生分组收益的交易日）</Typography.Text>
          </div>
          {holding.isLoading ? (
            <Skeleton active />
          ) : holding.isError ? (
            <Alert type="warning" showIcon message="该日期没有可审计持仓" description={holding.error.message} />
          ) : holding.data ? (
            <>
              <Alert
                type="info"
                showIcon
                style={{ marginBottom: 12 }}
                message={`公司名称为${holding.data.reference_data.security_name_basis}${
                  holding.data.reference_data.security_name_observed_at
                    ? `（主表截至 ${holding.data.reference_data.security_name_observed_at}）`
                    : ''
                }；行业名称按信号日的${holding.data.reference_data.industry_classification || '一级行业分类'}代码翻译。原始代码同时保留。`}
              />
              <Descriptions size="small" column={4} style={{ marginBottom: 12 }}>
                <Descriptions.Item label="分组">{holding.data.top_group}</Descriptions.Item>
                <Descriptions.Item label="买入日">{holding.data.entry_day}</Descriptions.Item>
                <Descriptions.Item label="卖出日">{holding.data.exit_day}</Descriptions.Item>
                <Descriptions.Item label="持仓数">{holding.data.total}</Descriptions.Item>
                <Descriptions.Item label="组合毛收益">{percent(holding.data.summary.top_group_gross_return)}</Descriptions.Item>
                <Descriptions.Item label="交易成本">{percent(holding.data.summary.top_group_transaction_cost, 3)}</Descriptions.Item>
                <Descriptions.Item label="组合净收益">{percent(holding.data.summary.top_group_net_return)}</Descriptions.Item>
                <Descriptions.Item label="等权目标权重">{percent(holding.data.summary.top_group_weight, 4)}</Descriptions.Item>
              </Descriptions>
              <Table<HoldingAuditPosition>
                rowKey={(row) => `${row.signal_day}-${row.security_code}`}
                size="small"
                columns={columns}
                dataSource={holding.data.rows}
                scroll={{ x: 'max-content' }}
                pagination={{
                  current: holding.data.page,
                  total: holding.data.total,
                  pageSize: holding.data.page_size,
                  showSizeChanger: false,
                  onChange: (nextPage) => setPage(nextPage),
                }}
              />
            </>
          ) : null}
        </>
      )}
    </Card>
  )
}
