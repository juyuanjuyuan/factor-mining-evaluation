import { useQuery } from '@tanstack/react-query'
import { Alert, Card, Empty, Input, Select, Skeleton, Space, Tag, Typography } from 'antd'
import { useEffect, useMemo, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import { api, Method } from '../api/client'
import { LatexFormula } from '../components/LatexFormula'
import { SectionTabs } from '../components/SectionTabs'
import { METHOD_DEFINITIONS } from '../lib/methodDefinitions'
import { methodContractLabel, methodLabel } from '../lib/metrics'

type MethodWithDefinition = Method & {
  label: string
  category: string
  description: string
}

const ALL_CATEGORIES = '全部分类'

function methodSummary(method: Method): MethodWithDefinition {
  const definition = METHOD_DEFINITIONS[method.name]
  return {
    ...method,
    label: definition?.label || method.name,
    category: definition?.category || '未归类',
    description: definition?.description || '该方法已注册，但网页定义尚未补齐。',
  }
}

function stateEffect(methodName: string): string {
  if (
    methodName === 'market_cap_neutralize' ||
    methodName === 'industry_neutralize' ||
    methodName === 'industry_market_cap_neutralize'
  ) {
    const label =
      methodName === 'industry_neutralize'
        ? '行业中性化'
        : methodName === 'industry_market_cap_neutralize'
          ? '行业＋市值联合中性化'
          : '市值中性化'
    return `会替换当前工作因子：后续直接读取 state.factor 的 IC、分组收益和回测方法会自动使用${label}后的因子。`
  }
  if (methodName === 'tradability_filter') {
    return '会屏蔽当前工作因子的不可交易样本：后续评价方法只在剩余可交易样本上运行。'
  }
  if (methodName === 'cycle_context') {
    return '不修改当前工作因子，也不产生指标或明细；只重绘已有收益曲线图，并向运行详情页提供周期背景。'
  }
  return '不修改当前工作因子；只读取因子或上游明细并写出指标、明细或图表。'
}

export default function MethodLibrary() {
  const methods = useQuery({ queryKey: ['methods'], queryFn: api.methods })
  const [searchParams, setSearchParams] = useSearchParams()
  const [query, setQuery] = useState('')
  const [category, setCategory] = useState(ALL_CATEGORIES)
  const selectedFromUrl = searchParams.get('method') || undefined
  const [selectedName, setSelectedName] = useState<string | undefined>(selectedFromUrl)

  const catalog = useMemo(
    () => (methods.data || []).map(methodSummary).sort((a, b) => a.category.localeCompare(b.category) || a.name.localeCompare(b.name)),
    [methods.data],
  )

  const categories = useMemo(
    () => [ALL_CATEGORIES, ...Array.from(new Set(catalog.map((method) => method.category)))],
    [catalog],
  )

  const filtered = useMemo(() => {
    const needle = query.trim().toLowerCase()
    return catalog.filter((method) => {
      const matchesCategory = category === ALL_CATEGORIES || method.category === category
      const matchesQuery =
        !needle ||
        method.name.toLowerCase().includes(needle) ||
        method.label.toLowerCase().includes(needle) ||
        method.description.toLowerCase().includes(needle)
      return matchesCategory && matchesQuery
    })
  }, [catalog, category, query])

  useEffect(() => {
    if (!catalog.length) return
    if (selectedName && catalog.some((method) => method.name === selectedName)) return
    const fallback = filtered[0]?.name || catalog[0].name
    setSelectedName(fallback)
    setSearchParams({ method: fallback }, { replace: true })
  }, [catalog, filtered, selectedName, setSearchParams])

  const selected = catalog.find((method) => method.name === selectedName)
  const definition = selected ? METHOD_DEFINITIONS[selected.name] : undefined
  const missingDefinitions = catalog.filter((method) => !METHOD_DEFINITIONS[method.name])

  const selectMethod = (name: string) => {
    setSelectedName(name)
    setSearchParams({ method: name })
  }

  return (
    <>
      <div className="page-heading page-heading-row">
        <div className="page-heading-copy">
          <Typography.Title level={2}>评价模块库</Typography.Title>
          <Typography.Text type="secondary">
            这里集中维护每个评价函数的公式、定义、依赖、数据输入和使用解读；流水线模板页只负责组合执行顺序
          </Typography.Text>
        </div>
        <SectionTabs
          tabs={[
            { label: '流水线模板', path: '/pipelines' },
            { label: '评价模块库', path: '/methods' },
          ]}
        />
      </div>

      {missingDefinitions.length > 0 && (
        <Alert
          type="error"
          showIcon
          className="method-definition-warning"
          message="存在未完成网页定义的评价模块"
          description={`请补充：${missingDefinitions.map((method) => method.name).join('、')}`}
        />
      )}

      <div className="method-library-layout">
        <Card className="surface-card method-library-sidebar">
          <Space orientation="vertical" size="middle" style={{ width: '100%' }}>
            <Input.Search
              allowClear
              placeholder="搜索模块名、key 或说明"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
            />
            <Select
              value={category}
              onChange={setCategory}
              options={categories.map((item) => ({ label: item, value: item }))}
              style={{ width: '100%' }}
            />
            {methods.isLoading ? (
              <Skeleton active />
            ) : filtered.length === 0 ? (
              <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="没有匹配的模块" />
            ) : (
              <div className="method-library-list" role="listbox" aria-label="评价模块列表">
                {filtered.map((method) => (
                  <div
                    key={method.name}
                    role="option"
                    aria-selected={method.name === selected?.name}
                    tabIndex={0}
                    className={`method-library-list-item${method.name === selected?.name ? ' active' : ''}`}
                    onClick={() => selectMethod(method.name)}
                    onKeyDown={(event) => {
                      if (event.key === 'Enter' || event.key === ' ') {
                        event.preventDefault()
                        selectMethod(method.name)
                      }
                    }}
                  >
                    <Space size={6} wrap>
                      <Typography.Text strong>{method.label}</Typography.Text>
                      {method.is_default && <Tag color="blue">默认</Tag>}
                    </Space>
                    <div>
                      <Typography.Text code>{method.name}</Typography.Text>
                      <span className="method-library-list-desc"> · {method.category}</span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </Space>
        </Card>

        <Card
          className="surface-card method-definition-card method-library-detail"
          title={
            selected ? (
              <Space wrap>
                <Typography.Text strong>{selected.label}</Typography.Text>
                <Typography.Text code>{selected.name}</Typography.Text>
              </Space>
            ) : (
              '模块详情'
            )
          }
          extra={
            selected && (
              <Space>
                {selected.is_default && <Tag color="blue">默认</Tag>}
                <Tag>{selected.category}</Tag>
              </Space>
            )
          }
        >
          {!selected ? (
            <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="请选择一个评价模块" />
          ) : definition ? (
            <>
              <Typography.Paragraph type="secondary" className="method-definition-summary">
                {definition.description}
              </Typography.Paragraph>
              <LatexFormula formula={definition.formula} />
              <div className="method-definition-section">
                <Typography.Text className="method-definition-label">方法定义</Typography.Text>
                <Typography.Paragraph>{definition.definition}</Typography.Paragraph>
              </div>
              <div className="method-definition-section">
                <Typography.Text className="method-definition-label">结果解读</Typography.Text>
                <Typography.Paragraph type="secondary">{definition.interpretation}</Typography.Paragraph>
              </div>
              {definition.limitations && (
                <div className="method-definition-section">
                  <Typography.Text className="method-definition-label">局限与口径</Typography.Text>
                  <Typography.Paragraph type="secondary">{definition.limitations}</Typography.Paragraph>
                </div>
              )}
              <div className="method-definition-section">
                <Typography.Text className="method-definition-label">状态影响</Typography.Text>
                <Typography.Paragraph type="secondary">{stateEffect(selected.name)}</Typography.Paragraph>
              </div>
              <div className="method-definition-contract">
                <span>
                  依赖：
                  {selected.requires.length
                    ? selected.requires.map(methodContractLabel).join('、')
                    : '无'}
                </span>
                <span>
                  提供：
                  {selected.provides.length
                    ? selected.provides.map(methodContractLabel).join('、')
                    : methodLabel(selected.name)}
                </span>
                <span>
                  额外数据：
                  {selected.required_data_symbols.length
                    ? selected.required_data_symbols.join(' / ')
                    : '无'}
                </span>
              </div>
            </>
          ) : (
            <Alert
              type="error"
              showIcon
              message="定义缺失"
              description="该方法已经注册，但尚未补充数学公式、方法定义和中文解读。"
            />
          )}
        </Card>
      </div>
    </>
  )
}
