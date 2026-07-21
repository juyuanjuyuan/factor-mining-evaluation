import { CheckCircleOutlined, PlusOutlined, RocketOutlined, SwapOutlined, TagsOutlined } from '@ant-design/icons'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Alert, Button, Card, Collapse, Empty, Flex, Input, Modal, Segmented, Select, Skeleton, Space, Statistic, Table, Tag, Tooltip, Typography, message } from 'antd'
import { useEffect, useMemo, useRef, useState } from 'react'
import { Link, useNavigate, useSearchParams } from 'react-router-dom'
import { api, Factor } from '../api/client'
import { CorrelationWindowChart } from '../components/charts/SeriesChart'
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
  const tone =
    metric === 'nw_ic_p_value' ? (value < 0.05 ? 'good' : '') : value > 0 ? 'good' : value < 0 ? 'bad' : ''
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
  const isTestLibrary = mode === 'test'
  const [correlationPair, setCorrelationPair] = useState<CorrelationPairSelection | null>(null)
  const factors = useQuery({
    queryKey: [isTestLibrary ? 'test-factors' : 'factors'],
    queryFn: isTestLibrary ? api.testFactors : api.factors,
  })
  const correlation = useQuery({
    queryKey: ['factor-correlation'],
    queryFn: api.factorCorrelation,
    enabled: !isTestLibrary,
  })
  const pairCorrelation = useQuery({
    queryKey: ['factor-correlation-pair', correlationPair?.factorA, correlationPair?.factorB],
    queryFn: () => api.factorCorrelationPair(correlationPair!.factorA, correlationPair!.factorB),
    enabled: !isTestLibrary && correlationPair !== null,
  })
  // 筛选条件放在 URL 上：进详情页再返回（或刷新、分享链接）时不丢失
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
  const [selected, setSelected] = useState<React.Key[]>([])
  const [drawerFactors, setDrawerFactors] = useState<Factor[]>([])
  const [tagRun, setTagRun] = useState(false)
  const [tagEditorFactor, setTagEditorFactor] = useState<Factor | null>(null)
  // 展开的项目分组与各组页码在会话内记住，返回列表时保持原样
  const expandedKey = `factor-library:${mode}:expanded`
  const pagesKey = `factor-library:${mode}:pages`
  const [expandedProjects, setExpandedProjects] = useState<string[]>(() => {
    try {
      const raw = sessionStorage.getItem(expandedKey)
      return raw ? (JSON.parse(raw) as string[]) : []
    } catch {
      return []
    }
  })
  const [pages, setPages] = useState<Record<string, { current: number; pageSize: number }>>(() => {
    try {
      return JSON.parse(sessionStorage.getItem(pagesKey) || '{}')
    } catch {
      return {}
    }
  })
  const setProjectPage = (projectName: string, current: number, pageSize: number) => {
    setPages((prev) => {
      const next = { ...prev, [projectName]: { current, pageSize } }
      try {
        sessionStorage.setItem(pagesKey, JSON.stringify(next))
      } catch {
        // 存储不可用时静默降级为内存状态
      }
      return next
    })
  }
  const initializedProjects = useRef(sessionStorage.getItem(expandedKey) !== null)
  const tagCatalog = useQuery({
    queryKey: ['factor-tags', mode],
    queryFn: () => api.factorTags(mode),
  })
  const submit = useMutation({
    mutationFn: (factor: Factor) => api.submitFactor(factor.batch_id, factor.factor_name),
    onSuccess: (factor) => {
      message.success('已提交到因子库')
      queryClient.invalidateQueries({ queryKey: ['test-factors'] })
      queryClient.invalidateQueries({ queryKey: ['factors'] })
      navigate(`/factors/${factor.batch_id}/${factor.factor_name}`)
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

  const factorGroups = useMemo(() => {
    const groups = new Map<string, Factor[]>()
    for (const factor of filtered) {
      const rows = groups.get(factor.project) || []
      rows.push(factor)
      groups.set(factor.project, rows)
    }
    return [...groups.entries()].sort(([left], [right]) => left.localeCompare(right, 'zh-CN'))
  }, [filtered])

  useEffect(() => {
    if (!initializedProjects.current && factorGroups.length) {
      setExpandedProjects(factorGroups.map(([name]) => name))
      initializedProjects.current = true
    }
  }, [factorGroups])
  const handleExpandedChange = (keys: string | string[]) => {
    const next = Array.isArray(keys) ? keys.map(String) : [String(keys)]
    setExpandedProjects(next)
    try {
      sessionStorage.setItem(expandedKey, JSON.stringify(next))
    } catch {
      // 存储不可用时静默降级为内存状态
    }
  }

  const taggedFactors = useMemo(
    () => (factors.data || []).filter((factor) => matchesTags(factor, tagFilters, tagMatch)),
    [factors.data, tagFilters, tagMatch],
  )

  const selectedFactors = (factors.data || []).filter((factor) =>
    selected.includes(`${factor.batch_id}/${factor.factor_name}`),
  )
  const comparableRuns = selectedFactors
    .map((factor) => (factor.latest_run?.status === 'succeeded' ? factor.latest_run.id : null))
    .filter((id): id is string => id !== null)
  const librarySummary = useMemo(() => {
    const rows = factors.data || []
    const successful = rows.filter((factor) => factor.latest_run?.status === 'succeeded').length
    const inProgress = rows.filter((factor) => ['queued', 'running'].includes(factor.latest_run?.status || '')).length
    return {
      total: rows.length,
      successful,
      inProgress,
      withoutRun: rows.length - successful - inProgress,
    }
  }, [factors.data])
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

  return (
    <>
      <Flex justify="space-between" align="center" className="page-heading" wrap gap={16}>
        <div className="page-heading-copy">
          <Typography.Title level={2}>{isTestLibrary ? '测试库' : '因子库'}</Typography.Title>
          <Typography.Text type="secondary">
            {isTestLibrary
              ? `共 ${factors.data?.length ?? '…'} 个测试因子 · 当前项目因子先在这里反复回测，确认后再提交到因子库`
              : `共 ${factors.data?.length ?? '…'} 个已提交因子 · 这里只展示从测试库提交后的正式定义`}
          </Typography.Text>
        </div>
        <Space className="page-heading-actions">
          {isTestLibrary && (
            <Link to="/test-factors/new">
              <Button type="primary" icon={<PlusOutlined />}>
                新增测试因子
              </Button>
            </Link>
          )}
        </Space>
      </Flex>
      <div className="library-summary" aria-label="因子库概览">
        <div className="library-summary-item">
          <span>全部因子</span>
          <strong>{librarySummary.total}</strong>
        </div>
        <div className="library-summary-item">
          <span>已有成功结果</span>
          <strong>{librarySummary.successful}</strong>
        </div>
        <div className="library-summary-item">
          <span>正在评价</span>
          <strong>{librarySummary.inProgress}</strong>
        </div>
        <div className="library-summary-item">
          <span>尚待评价</span>
          <strong>{librarySummary.withoutRun}</strong>
        </div>
      </div>
      {!isTestLibrary && (
        <Collapse
          className="collapse-card"
          style={{ marginBottom: 20 }}
          items={[
            {
              key: 'correlation',
              label: (
                <Space wrap>
                  <span>因子相关性矩阵</span>
                  {correlation.data && (
                    <Tag color={correlation.data.passed ? 'green' : 'red'}>
                      {correlation.data.passed ? '符合阈值' : '存在超阈值因子对'}
                    </Tag>
                  )}
                  <Typography.Text type="secondary" style={{ fontWeight: 'normal' }}>
                    展开查看完整矩阵与分段相关性
                  </Typography.Text>
                </Space>
              ),
              children: (
                <>
                  <Typography.Paragraph type="secondary">
                    {correlation.data
                      ? `${correlation.data.sample_definition}。非对角元素按绝对值不超过 ${correlation.data.threshold.toFixed(2)} 校验；对角线为因子自身相关性 1。点击任一非对角数字可查看 60 日非重叠分段相关性。`
                      : '正在读取已保存的因子相关性矩阵…'}
                  </Typography.Paragraph>
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
                  ) : !correlation.isLoading ? (
                    <Typography.Text type="secondary">因子库为空，提交首个因子后会在此生成 1 × 1 矩阵。</Typography.Text>
                  ) : null}
                </>
              ),
            },
          ]}
        />
      )}
      <Card className="surface-card">
        <Space style={{ marginBottom: 18 }} wrap>
          <Input.Search
            allowClear
            placeholder="搜索因子名或表达式"
            value={search}
            onChange={(event) => updateParams({ q: event.target.value || undefined })}
            style={{ width: 320 }}
          />
          <Select
            allowClear
            placeholder="全部批次"
            style={{ width: 240 }}
            value={batch}
            onChange={(value) => updateParams({ batch: value })}
            options={batches.map(([id, name]) => ({ value: id, label: name }))}
          />
          <Select
            allowClear
            placeholder="全部项目"
            style={{ width: 180 }}
            value={project}
            onChange={(value) => updateParams({ project: value })}
            options={projects.map((name) => ({ value: name, label: name }))}
          />
          <Select
            allowClear
            mode="multiple"
            maxTagCount="responsive"
            placeholder="按标签筛选"
            style={{ width: 300 }}
            value={tagFilters}
            onChange={(values) => updateParams({ tags: values.length ? values.join(',') : undefined })}
            options={(tagCatalog.data || []).map((item) => ({
              value: item.tag,
              label: `${item.tag}（${item.count}）`,
            }))}
          />
          {tagFilters.length > 1 && (
            <Segmented
              value={tagMatch}
              onChange={(value) => updateParams({ match: value === 'all' ? 'all' : undefined })}
              options={[
                { label: '匹配任一', value: 'any' },
                { label: '同时包含', value: 'all' },
              ]}
            />
          )}
        </Space>
        {tagFilters.length > 0 && (
          <div className="tag-run-summary" role="status">
            <div>
              <Typography.Text strong>标签组合匹配 {taggedFactors.length} 个因子</Typography.Text>
              <Typography.Text type="secondary">
                提交时会按当前标签重新匹配，搜索和分页不会缩小组合。
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
        {selectedFactors.length > 0 && (
          <div className="selection-summary" role="status">
            <div>
              <Typography.Text strong>已选择 {selectedFactors.length} 个因子</Typography.Text>
              <Typography.Text type="secondary">可使用同一套配置批量评价</Typography.Text>
            </div>
            <Space wrap>
              <Button icon={<RocketOutlined />} onClick={() => setDrawerFactors(selectedFactors)}>
                评价所选
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
            activeKey={expandedProjects}
            onChange={handleExpandedChange}
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
          rowKey={(factor) => `${factor.batch_id}/${factor.factor_name}`}
          loading={factors.isLoading}
          dataSource={projectFactors}
          scroll={{ x: 1360 }}
          sticky={{ offsetHeader: 68 }}
          rowSelection={{
            selectedRowKeys: selected.filter((key) =>
              projectFactors.some((factor) => `${factor.batch_id}/${factor.factor_name}` === key),
            ),
            onChange: (projectSelectedKeys) => {
              const projectFactorKeys = new Set(
                projectFactors.map((factor) => `${factor.batch_id}/${factor.factor_name}`),
              )
              setSelected((current) => [
                ...current.filter((key) => !projectFactorKeys.has(String(key))),
                ...projectSelectedKeys,
              ])
            },
          }}
          pagination={{
            current: Math.min(
              pages[projectName]?.current ?? 1,
              Math.max(1, Math.ceil(projectFactors.length / (pages[projectName]?.pageSize ?? 20))),
            ),
            pageSize: pages[projectName]?.pageSize ?? 20,
            showSizeChanger: true,
            showTotal: (total) => `共 ${total} 个`,
            onChange: (current, pageSize) => setProjectPage(projectName, current, pageSize),
          }}
          columns={[
            {
              title: '因子',
              width: 220,
              fixed: 'left',
              render: (_, factor) => (
                <div>
                  <Link to={`/${isTestLibrary ? 'test-factors' : 'factors'}/${factor.batch_id}/${factor.factor_name}`}>
                    <Typography.Text strong>{factor.factor_name}</Typography.Text>
                  </Link>
                  <div>
                    <Typography.Text type="secondary" style={{ fontSize: 12 }}>
                      {factor.factor_category || factor.implementation_set}
                      {factor.editable && (
                        <Tag color="cyan" style={{ marginLeft: 6 }}>
                          可编辑
                        </Tag>
                      )}
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
              title: 'IC 均值',
              width: 96,
              sorter: sorter('ic_mean'),
              render: (_, factor) => <MetricValue factor={factor} metric="ic_mean" />,
            },
            {
              title: 'ICIR',
              width: 90,
              sorter: sorter('ir'),
              render: (_, factor) => <MetricValue factor={factor} metric="ir" />,
            },
            {
              title: (
                <Tooltip title="Newey-West 检验的 p 值，绿色表示 5% 显著">
                  <span>p 值</span>
                </Tooltip>
              ),
              width: 84,
              sorter: sorter('nw_ic_p_value'),
              render: (_, factor) => <MetricValue factor={factor} metric="nw_ic_p_value" />,
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
              title: '数据',
              dataIndex: 'required_symbols',
              width: 130,
              render: (symbols: string[]) =>
                symbols.map((symbol) => <Tag key={symbol}>{symbol}</Tag>),
            },
            {
              title: '',
              width: isTestLibrary ? 168 : 126,
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
                    <Tooltip title={factor.submitted ? '已在因子库中' : '提交后会出现在因子库'}>
                      <Button
                        size="small"
                        icon={<CheckCircleOutlined />}
                        disabled={factor.submitted || submit.isPending}
                        onClick={() => submit.mutate(factor)}
                      />
                    </Tooltip>
                  )}
                  <Button type="primary" ghost size="small" onClick={() => setDrawerFactors([factor])}>
                    回测
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
      </Card>
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
        title={
          correlationPair
            ? `${correlationPair.factorA} / ${correlationPair.factorB}`
            : '分段相关性'
        }
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
                  valueStyle={
                    pairCorrelation.data?.violation_window_count
                      ? { color: '#cf1322' }
                      : undefined
                  }
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
    </>
  )
}
