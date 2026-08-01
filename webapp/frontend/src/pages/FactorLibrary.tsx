import {
  BranchesOutlined,
  CheckCircleOutlined,
  DeleteOutlined,
  DotChartOutlined,
  FilterOutlined,
  PlusOutlined,
  RocketOutlined,
  SwapOutlined,
  TableOutlined,
  TagsOutlined,
} from '@ant-design/icons'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Alert, Badge, Button, Card, Collapse, Empty, Flex, Input, Modal, Popconfirm, Popover, Segmented, Select, Skeleton, Space, Statistic, Table, Tag, Tooltip, Typography, message } from 'antd'
import { useEffect, useMemo, useState } from 'react'
import { Link, useNavigate, useParams, useSearchParams } from 'react-router-dom'
import { api, Factor } from '../api/client'
import { CorrelationWindowChart } from '../components/charts/SeriesChart'
import { FactorDetailEmpty, FactorDetailPanel } from '../components/FactorDetailPanel'
import { FactorRail, RailSort, factorKey, orderFactors } from '../components/FactorRail'
import { RunConfigDrawer } from '../components/RunConfigDrawer'
import { StatusTag } from '../components/StatusTag'
import { TagEditorModal } from '../components/TagEditorModal'
import { formatMetric, shortTime } from '../lib/metrics'

const latestMetric = (factor: Factor, key: string): number | null => {
  const value = factor.latest_run?.result?.[key]
  return typeof value === 'number' ? value : null
}

function MetricValue({ factor, metric }: { factor: Factor; metric: string }) {
  const value = latestMetric(factor, metric)
  if (value === null) return <Typography.Text type="secondary">暂无</Typography.Text>
  const tone = value > 0 ? 'good' : value < 0 ? 'bad' : ''
  return <span className={`metric-number ${tone}`}>{formatMetric(metric, value)}</span>
}

const sorter = (metric: string) => (a: Factor, b: Factor) => {
  const va = latestMetric(a, metric)
  const vb = latestMetric(b, metric)
  if (va === null && vb === null) return 0
  if (va === null) return -1
  if (vb === null) return 1
  return va - vb
}

function matchesTags(factor: Factor, selectedTags: string[], match: 'any' | 'all') {
  if (!selectedTags.length) return true
  const available = new Set(factor.tags.map((tag) => tag.toLocaleLowerCase()))
  const selected = selectedTags.map((tag) => tag.toLocaleLowerCase())
  return match === 'all' ? selected.every((tag) => available.has(tag)) : selected.some((tag) => available.has(tag))
}

type FactorLibraryProps = {
  mode: 'test' | 'factor'
}

type CorrelationRow = {
  key: string
  factor_name: string
  [factorName: string]: string | number
}

type CorrelationPairSelection = {
  factorA: string
  factorB: string
}

export default function FactorLibrary({ mode }: FactorLibraryProps) {
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const { batchId: routeBatchId, name: routeName } = useParams()
  const isTestLibrary = mode === 'test'
  const base = isTestLibrary ? '/test-factors' : '/factors'
  const [correlationPair, setCorrelationPair] = useState<CorrelationPairSelection | null>(null)
  const [overviewOpen, setOverviewOpen] = useState(false)
  const [correlationOpen, setCorrelationOpen] = useState(false)
  const factors = useQuery({
    queryKey: [isTestLibrary ? 'test-factors' : 'factors'],
    queryFn: isTestLibrary ? api.testFactors : api.factors,
  })
  const correlation = useQuery({
    queryKey: ['factor-correlation'],
    queryFn: api.factorCorrelation,
    enabled: !isTestLibrary && correlationOpen,
  })
  const pairCorrelation = useQuery({
    queryKey: ['factor-correlation-pair', correlationPair?.factorA, correlationPair?.factorB],
    queryFn: () => api.factorCorrelationPair(correlationPair!.factorA, correlationPair!.factorB),
    enabled: !isTestLibrary && correlationPair !== null,
  })
  // 筛选条件放在 URL 上：切换因子、刷新或分享链接时都不丢失
  const [searchParams, setSearchParams] = useSearchParams()
  const search = searchParams.get('q') ?? ''
  const batch = searchParams.get('batch') ?? undefined
  const project = searchParams.get('project') ?? undefined
  const tagFilters = useMemo(
    () => searchParams.get('tags')?.split(',').filter(Boolean) ?? [],
    [searchParams],
  )
  const tagMatch: 'any' | 'all' = searchParams.get('match') === 'all' ? 'all' : 'any'
  const updateParams = (patch: Record<string, string | undefined>) => {
    setSearchParams(
      (current) => {
        const next = new URLSearchParams(current)
        for (const [key, value] of Object.entries(patch)) {
          if (value) next.set(key, value)
          else next.delete(key)
        }
        return next
      },
      { replace: true },
    )
  }
  const filterCount = (batch ? 1 : 0) + (project ? 1 : 0) + tagFilters.length
  const factorPath = (factor: Factor) => {
    const query = searchParams.toString()
    return `${base}/${factor.batch_id}/${factor.factor_name}${query ? `?${query}` : ''}`
  }
  const [selected, setSelected] = useState<string[]>([])
  const [drawerFactors, setDrawerFactors] = useState<Factor[]>([])
  const [tagRun, setTagRun] = useState(false)
  const [tagEditorFactor, setTagEditorFactor] = useState<Factor | null>(null)
  const [railSort, setRailSort] = useState<RailSort>(() => {
    const stored = sessionStorage.getItem(`factor-library:${mode}:sort`)
    return (stored as RailSort) || 'name'
  })
  const changeRailSort = (next: RailSort) => {
    setRailSort(next)
    try {
      sessionStorage.setItem(`factor-library:${mode}:sort`, next)
    } catch {
      // 存储不可用时静默降级为内存状态
    }
  }
  const tagCatalog = useQuery({
    queryKey: ['factor-tags', mode],
    queryFn: () => api.factorTags(mode),
  })
  const submit = useMutation({
    mutationFn: (factor: Factor) => api.submitFactor(factor.batch_id, factor.factor_name),
    onSuccess: (factor) => {
      const replaced = factor.admission_decision?.replaced_factor_names || []
      message.success(
        replaced.length
          ? `已提交到因子库；${replaced.join('、')} 因相关性高且表现较弱已降级至测试库`
          : '已提交到因子库',
      )
      queryClient.invalidateQueries({ queryKey: ['test-factors'] })
      queryClient.invalidateQueries({ queryKey: ['factors'] })
      navigate(`/factors/${factor.batch_id}/${factor.factor_name}`)
    },
    onError: (error) => message.error(error.message),
  })
  const removeFromLibrary = useMutation({
    mutationFn: (factor: Factor) => api.removeFromFactorLibrary(factor.batch_id, factor.factor_name),
    onSuccess: () => {
      message.success('已移出因子库；源测试因子和历史评价均已保留')
      setCorrelationPair(null)
      queryClient.invalidateQueries({ queryKey: ['factors'] })
      queryClient.invalidateQueries({ queryKey: ['test-factors'] })
      queryClient.invalidateQueries({ queryKey: ['factor-correlation'] })
      queryClient.invalidateQueries({ queryKey: ['factor-tags', 'factor'] })
    },
    onError: (error) => message.error(error.message),
  })

  const batches = useMemo(() => {
    const map = new Map<string, string>()
    for (const factor of factors.data || []) map.set(factor.batch_id, factor.batch_name)
    return [...map.entries()]
  }, [factors.data])
  const projects = useMemo(
    () => [...new Set((factors.data || []).map((factor) => factor.project))]
      .sort((left, right) => left.localeCompare(right, 'zh-CN')),
    [factors.data],
  )

  const filtered = useMemo(() => {
    let rows = factors.data || []
    if (batch) rows = rows.filter((factor) => factor.batch_id === batch)
    if (project) rows = rows.filter((factor) => factor.project === project)
    rows = rows.filter((factor) => matchesTags(factor, tagFilters, tagMatch))
    const term = search.trim().toLowerCase()
    if (term) {
      rows = rows.filter(
        (factor) =>
          factor.factor_name.toLowerCase().includes(term) ||
          factor.expression.toLowerCase().includes(term),
      )
    }
    return rows
  }, [factors.data, search, batch, project, tagFilters, tagMatch])

  const factorGroups = useMemo(() => orderFactors(filtered, railSort), [filtered, railSort])

  // 右侧不留白：没有指定因子就落到列表里显示的第一行
  useEffect(() => {
    if (routeName || !filtered.length) return
    const first = orderFactors(filtered, railSort)[0]?.[1][0]
    if (!first) return
    navigate(`${base}/${first.batch_id}/${first.factor_name}${searchParams.toString() ? `?${searchParams}` : ''}`, {
      replace: true,
    })
  }, [routeName, filtered, railSort, base, navigate, searchParams])

  const taggedFactors = useMemo(
    () => (factors.data || []).filter((factor) => matchesTags(factor, tagFilters, tagMatch)),
    [factors.data, tagFilters, tagMatch],
  )

  const selectedFactors = (factors.data || []).filter((factor) => selected.includes(factorKey(factor)))
  const comparableRuns = selectedFactors
    .map((factor) => (factor.latest_run?.status === 'succeeded' ? factor.latest_run.id : null))
    .filter((id): id is string => id !== null)
  const correlationRows = useMemo<CorrelationRow[]>(() => {
    const data = correlation.data
    if (!data) return []
    return data.factor_names.map((factorName, rowIndex) => ({
      key: factorName,
      factor_name: factorName,
      ...Object.fromEntries(
        data.factor_names.map((columnName, columnIndex) => [columnName, data.matrix[rowIndex]?.[columnIndex]]),
      ),
    }))
  }, [correlation.data])
  const correlationColumns = useMemo(
    () => [
      {
        title: '因子',
        dataIndex: 'factor_name',
        key: 'factor_name',
        width: 220,
        fixed: 'left' as const,
        render: (name: string) => <Typography.Text strong>{name}</Typography.Text>,
      },
      ...(correlation.data?.factor_names || []).map((factorName) => ({
        title: <Tooltip title={factorName}>{factorName}</Tooltip>,
        dataIndex: factorName,
        key: factorName,
        width: 138,
        align: 'right' as const,
        render: (value: number, row: CorrelationRow) => {
          const isDiagonal = row.factor_name === factorName
          const exceeds = !isDiagonal && Math.abs(value) > (correlation.data?.threshold || 0.75)
          if (isDiagonal) {
            return (
              <Typography.Text type="secondary">
                {Number.isFinite(value) ? value.toFixed(4) : '暂无'}
              </Typography.Text>
            )
          }
          return (
            <Tooltip title={`查看 ${row.factor_name} 与 ${factorName} 的 60 日分段相关性`}>
              <Button
                type="link"
                danger={exceeds}
                size="small"
                aria-label={`查看 ${row.factor_name} 与 ${factorName} 的分段相关性`}
                style={{ height: 'auto', padding: 0, fontVariantNumeric: 'tabular-nums' }}
                onClick={() => setCorrelationPair({ factorA: row.factor_name, factorB: factorName })}
              >
                {Number.isFinite(value) ? value.toFixed(4) : '暂无'}
              </Button>
            </Tooltip>
          )
        },
      })),
    ],
    [correlation.data],
  )

  const activeKey = routeBatchId && routeName ? `${routeBatchId}/${routeName}` : null

  const railHeader = (
    <div className="factor-rail-top">
      <Segmented
        block
        value={base}
        onChange={(value) => navigate(String(value))}
        options={[
          { label: '测试库', value: '/test-factors' },
          { label: '因子库', value: '/factors' },
        ]}
      />
      <div className="factor-rail-search">
        <Input.Search
          allowClear
          placeholder="搜索因子名或表达式"
          value={search}
          onChange={(event) => updateParams({ q: event.target.value || undefined })}
        />
        <Popover
          trigger="click"
          placement="bottomRight"
          title="筛选"
          content={
            <div className="factor-filter-popover">
              <label>
                <span>批次</span>
                <Select
                  allowClear
                  placeholder="全部批次"
                  value={batch}
                  onChange={(value) => updateParams({ batch: value })}
                  options={batches.map(([id, name]) => ({ value: id, label: name }))}
                />
              </label>
              <label>
                <span>项目</span>
                <Select
                  allowClear
                  placeholder="全部项目"
                  value={project}
                  onChange={(value) => updateParams({ project: value })}
                  options={projects.map((name) => ({ value: name, label: name }))}
                />
              </label>
              <label>
                <span>研究标签</span>
                <Select
                  allowClear
                  mode="multiple"
                  maxTagCount="responsive"
                  placeholder="全部标签"
                  value={tagFilters}
                  onChange={(values) => updateParams({ tags: values.length ? values.join(',') : undefined })}
                  options={(tagCatalog.data || []).map((item) => ({
                    value: item.tag,
                    label: `${item.tag}（${item.count}）`,
                  }))}
                />
              </label>
              {tagFilters.length > 1 && (
                <Segmented
                  block
                  value={tagMatch}
                  onChange={(value) => updateParams({ match: value === 'all' ? 'all' : undefined })}
                  options={[
                    { label: '匹配任一', value: 'any' },
                    { label: '同时包含', value: 'all' },
                  ]}
                />
              )}
              <Button
                size="small"
                type="text"
                disabled={!filterCount}
                onClick={() => updateParams({ batch: undefined, project: undefined, tags: undefined, match: undefined })}
              >
                清空筛选
              </Button>
            </div>
          }
        >
          <Badge count={filterCount} size="small" color="#9a4f32" offset={[-4, 2]}>
            <Button icon={<FilterOutlined />} aria-label="筛选因子" />
          </Badge>
        </Popover>
      </div>
    </div>
  )

  const railFooter = (
    <Space size={6} wrap>
      {isTestLibrary && (
        <Link to="/test-factors/new">
          <Button size="small" icon={<PlusOutlined />}>
            新增因子
          </Button>
        </Link>
      )}
      <Button size="small" icon={<TableOutlined />} onClick={() => setOverviewOpen(true)}>
        总览表
      </Button>
      {!isTestLibrary && (
        <>
          <Button size="small" icon={<DotChartOutlined />} onClick={() => setCorrelationOpen(true)}>
            相关性
          </Button>
          <Link to="/mining">
            <Button size="small" icon={<BranchesOutlined />}>
              遗传挖掘
            </Button>
          </Link>
        </>
      )}
    </Space>
  )

  return (
    <div className="library-shell">
      <FactorRail
        factors={filtered}
        loading={factors.isLoading}
        sort={railSort}
        onSortChange={changeRailSort}
        activeKey={activeKey}
        onActivate={(factor) => navigate(factorPath(factor))}
        checkedKeys={selected}
        onCheckedChange={setSelected}
        onRunSelected={() => setDrawerFactors(selectedFactors)}
        onCompareSelected={() => navigate(`/compare?runs=${comparableRuns.join(',')}`)}
        comparableCount={comparableRuns.length}
        header={railHeader}
        footer={railFooter}
      />

      <main className="library-main">
        {tagFilters.length > 0 && (
          <div className="tag-run-summary" role="status">
            <div>
              <Typography.Text strong>标签组合匹配 {taggedFactors.length} 个因子</Typography.Text>
              <Typography.Text type="secondary">
                提交时会按当前标签重新匹配，搜索不会缩小组合。
              </Typography.Text>
            </div>
            <Button
              type="primary"
              icon={<RocketOutlined />}
              disabled={!taggedFactors.length}
              onClick={() => setTagRun(true)}
            >
              评价标签组合
            </Button>
          </div>
        )}
        {activeKey && routeBatchId && routeName ? (
          <FactorDetailPanel key={activeKey} batchId={routeBatchId} name={routeName} library={mode} />
        ) : (
          <FactorDetailEmpty library={mode} />
        )}
      </main>

      <Modal
        open={overviewOpen}
        title={`${isTestLibrary ? '测试库' : '因子库'}总览表`}
        width="min(1400px, 94vw)"
        footer={null}
        onCancel={() => setOverviewOpen(false)}
      >
        {selectedFactors.length > 0 && (
          <div className="selection-summary" role="status">
            <div>
              <Typography.Text strong>已选择 {selectedFactors.length} 个因子</Typography.Text>
              <Typography.Text type="secondary">可使用同一套配置批量评价</Typography.Text>
            </div>
            <Space wrap>
              <Button icon={<RocketOutlined />} onClick={() => setDrawerFactors(selectedFactors)}>
                提交测试
              </Button>
              <Tooltip
                title={
                  comparableRuns.length >= 2
                    ? '对比所选因子的最近一次成功结果'
                    : '所选因子中至少需要 2 个已有成功运行结果'
                }
              >
                <Button
                  icon={<SwapOutlined />}
                  disabled={comparableRuns.length < 2}
                  onClick={() => navigate(`/compare?runs=${comparableRuns.join(',')}`)}
                >
                  对比结果{comparableRuns.length >= 2 ? ` (${comparableRuns.length})` : ''}
                </Button>
              </Tooltip>
            </Space>
          </div>
        )}
        {factorGroups.length ? (
          <Collapse
            defaultActiveKey={factorGroups.map(([name]) => name)}
            className="factor-project-groups"
            items={factorGroups.map(([projectName, projectFactors]) => ({
              key: projectName,
              label: (
                <Space>
                  <Typography.Text strong>{projectName}</Typography.Text>
                  <Tag>{projectFactors.length} 个因子</Tag>
                </Space>
              ),
              children: (
                <Table<Factor>
                  rowKey={(factor) => factorKey(factor)}
                  loading={factors.isLoading}
                  dataSource={[...projectFactors]}
                  scroll={{ x: 1240 }}
                  rowSelection={{
                    selectedRowKeys: selected.filter((key) =>
                      projectFactors.some((factor) => factorKey(factor) === key),
                    ),
                    onChange: (projectSelectedKeys) => {
                      const projectFactorKeys = new Set(projectFactors.map(factorKey))
                      setSelected((current) => [
                        ...current.filter((key) => !projectFactorKeys.has(key)),
                        ...projectSelectedKeys.map(String),
                      ])
                    },
                  }}
                  pagination={{ pageSize: 20, showSizeChanger: true, showTotal: (total) => `共 ${total} 个` }}
                  columns={[
                    {
                      title: '因子',
                      width: 220,
                      fixed: 'left',
                      render: (_, factor) => (
                        <div>
                          <Link to={factorPath(factor)} onClick={() => setOverviewOpen(false)}>
                            <Typography.Text strong>{factor.factor_name}</Typography.Text>
                          </Link>
                          <div>
                            <Typography.Text type="secondary" style={{ fontSize: 12 }}>
                              {factor.factor_category || factor.implementation_set}
                              {isTestLibrary && factor.submitted && (
                                <Tag color="green" style={{ marginLeft: 6 }}>
                                  已提交
                                </Tag>
                              )}
                            </Typography.Text>
                          </div>
                        </div>
                      ),
                    },
                    {
                      title: <Tooltip title="最高分位组按每年 252 个交易日几何年化的收益率">年化收益率</Tooltip>,
                      width: 118,
                      sorter: sorter('top_group_annualized_return'),
                      render: (_, factor) => <MetricValue factor={factor} metric="top_group_annualized_return" />,
                    },
                    {
                      title: <Tooltip title="最高分位组 60 日非重叠窗口年化 Sharpe 的中位数">60日窗口 Sharpe 中位数</Tooltip>,
                      width: 176,
                      sorter: sorter('gn_rolling_sharpe_60_median'),
                      render: (_, factor) => <MetricValue factor={factor} metric="gn_rolling_sharpe_60_median" />,
                    },
                    {
                      title: <Tooltip title="最高分位组各自然年 Fitness 的等权平均">Fitness</Tooltip>,
                      width: 96,
                      sorter: sorter('fitness'),
                      render: (_, factor) => <MetricValue factor={factor} metric="fitness" />,
                    },
                    {
                      title: '最近评价',
                      width: 156,
                      render: (_, factor) =>
                        factor.latest_run ? (
                          <Link to={`/runs/${factor.latest_run.id}`}>
                            <Space size={6}>
                              <StatusTag status={factor.latest_run.status} />
                              <Typography.Text type="secondary" style={{ fontSize: 12 }}>
                                {shortTime(factor.latest_run.finished_at || factor.latest_run.created_at)}
                              </Typography.Text>
                            </Space>
                          </Link>
                        ) : (
                          <Typography.Text type="secondary">未评价</Typography.Text>
                        ),
                    },
                    {
                      title: '研究标签',
                      width: 170,
                      render: (_, factor) =>
                        factor.tags.length ? (
                          <Space size={[4, 4]} wrap>
                            {factor.tags.map((tag) => (
                              <Tag
                                key={tag}
                                className="factor-tag"
                                onClick={() => updateParams({ tags: tag, match: undefined })}
                              >
                                {tag}
                              </Tag>
                            ))}
                          </Space>
                        ) : (
                          <Typography.Text type="secondary">未标记</Typography.Text>
                        ),
                    },
                    {
                      title: '表达式',
                      dataIndex: 'expression',
                      ellipsis: { showTitle: true },
                      render: (value: string) => <Typography.Text code>{value}</Typography.Text>,
                    },
                    {
                      title: '',
                      width: 150,
                      fixed: 'right',
                      render: (_, factor) => (
                        <Space size={6}>
                          <Tooltip title="管理研究标签">
                            <Button
                              size="small"
                              icon={<TagsOutlined />}
                              aria-label={`管理 ${factor.factor_name} 的研究标签`}
                              onClick={() => setTagEditorFactor(factor)}
                            />
                          </Tooltip>
                          {isTestLibrary && (
                            <Tooltip title={factor.submitted ? '已在因子库中' : '提交时先做相关性检验；若与库内因子高度相关，仅在 60 日 Sharpe 中位数（同分看 Fitness）严格更优时替换旧因子'}>
                              <Button
                                size="small"
                                icon={<CheckCircleOutlined />}
                                disabled={factor.submitted || submit.isPending}
                                onClick={() => submit.mutate(factor)}
                              />
                            </Tooltip>
                          )}
                          {!isTestLibrary && (
                            <Popconfirm
                              title="移出因子库？"
                              description="仅移除正式库副本；源测试因子、标签和历史评价会保留。"
                              okText="确认移出"
                              cancelText="取消"
                              okButtonProps={{ danger: true }}
                              onConfirm={() => removeFromLibrary.mutate(factor)}
                            >
                              <Tooltip title="移出因子库">
                                <Button
                                  danger
                                  size="small"
                                  icon={<DeleteOutlined />}
                                  loading={removeFromLibrary.isPending}
                                  aria-label={`将 ${factor.factor_name} 移出因子库`}
                                />
                              </Tooltip>
                            </Popconfirm>
                          )}
                          <Button type="primary" ghost size="small" onClick={() => setDrawerFactors([factor])}>
                            测试
                          </Button>
                        </Space>
                      ),
                    },
                  ]}
                />
              ),
            }))}
          />
        ) : (
          <Empty description="没有符合当前筛选条件的因子" />
        )}
      </Modal>

      <Modal
        open={correlationOpen}
        title="因子相关性矩阵"
        width="min(1200px, 94vw)"
        footer={null}
        onCancel={() => setCorrelationOpen(false)}
      >
        <Typography.Paragraph type="secondary">
          {correlation.data
            ? `${correlation.data.sample_definition}。非对角元素按绝对值不超过 ${correlation.data.threshold.toFixed(2)} 校验；对角线为因子自身相关性 1。点击任一非对角数字可查看 60 日非重叠分段相关性。`
            : '正在读取已保存的因子相关性矩阵…'}
        </Typography.Paragraph>
        {correlation.data && (
          <Tag color={correlation.data.passed ? 'green' : 'red'} style={{ marginBottom: 12 }}>
            {correlation.data.passed ? '符合阈值' : '存在超阈值因子对'}
          </Tag>
        )}
        {correlation.isError ? (
          <Alert type="error" showIcon message="无法读取因子相关性矩阵" description={correlation.error.message} />
        ) : correlation.data?.factor_names.length ? (
          <Table<CorrelationRow>
            size="small"
            pagination={false}
            dataSource={correlationRows}
            columns={correlationColumns}
            scroll={{ x: Math.max(560, 220 + correlation.data.factor_names.length * 138) }}
          />
        ) : correlation.isLoading ? (
          <Skeleton active />
        ) : (
          <Typography.Text type="secondary">因子库为空，提交首个因子后会在此生成 1 × 1 矩阵。</Typography.Text>
        )}
      </Modal>

      <RunConfigDrawer
        open={drawerFactors.length > 0}
        factors={drawerFactors}
        onClose={() => setDrawerFactors([])}
      />
      <RunConfigDrawer
        open={tagRun}
        factors={taggedFactors}
        tagSelection={{ tags: tagFilters, match: tagMatch, library: mode }}
        onClose={() => setTagRun(false)}
      />
      <TagEditorModal
        factor={tagEditorFactor}
        library={mode}
        open={tagEditorFactor !== null}
        onClose={() => setTagEditorFactor(null)}
      />
      <Modal
        open={correlationPair !== null}
        title={correlationPair ? `${correlationPair.factorA} / ${correlationPair.factorB}` : '分段相关性'}
        width={860}
        footer={null}
        onCancel={() => setCorrelationPair(null)}
      >
        {pairCorrelation.isError ? (
          <Alert
            type="error"
            showIcon
            message="无法读取分段相关性"
            description={pairCorrelation.error.message}
          />
        ) : (
          <>
            <Typography.Paragraph type="secondary">
              {pairCorrelation.data?.sample_definition || '正在计算 60 日非重叠分段相关性…'}
              {pairCorrelation.data
                ? `。是否超过 ${pairCorrelation.data.threshold.toFixed(2)} 仅按相关系数绝对值判断。`
                : ''}
            </Typography.Paragraph>
            <Flex gap={12} wrap style={{ marginBottom: 16 }}>
              <Card size="small" style={{ flex: '1 1 260px' }}>
                <Statistic
                  title="最大分段绝对相关性"
                  value={pairCorrelation.data?.max_abs_correlation ?? '暂无'}
                  precision={pairCorrelation.data?.max_abs_correlation == null ? undefined : 4}
                  valueStyle={
                    pairCorrelation.data?.max_abs_correlation != null &&
                    pairCorrelation.data.max_abs_correlation > pairCorrelation.data.threshold
                      ? { color: '#cf1322' }
                      : undefined
                  }
                />
              </Card>
              <Card size="small" style={{ flex: '1 1 260px' }}>
                <Statistic
                  title="超过阈值的窗口数量"
                  value={pairCorrelation.data?.violation_window_count ?? '暂无'}
                  valueStyle={pairCorrelation.data?.violation_window_count ? { color: '#cf1322' } : undefined}
                />
              </Card>
            </Flex>
            <Typography.Text strong>分段相关性走势</Typography.Text>
            <Typography.Text type="secondary" style={{ display: 'block', marginTop: 4, marginBottom: 8 }}>
              横轴为每个非重叠窗口的结束日期，红点表示相关性绝对值超过阈值。
            </Typography.Text>
            {pairCorrelation.isLoading ? (
              <Card size="small" style={{ minHeight: 330, padding: '28px 24px' }}>
                <Skeleton active title={false} paragraph={{ rows: 8 }} />
              </Card>
            ) : pairCorrelation.data?.windows.length ? (
              <CorrelationWindowChart
                windows={pairCorrelation.data.windows}
                threshold={pairCorrelation.data.threshold}
              />
            ) : (
              <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="没有完整的 60 日窗口" />
            )}
          </>
        )}
      </Modal>
    </div>
  )
}
