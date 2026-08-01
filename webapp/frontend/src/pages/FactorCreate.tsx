import { ArrowLeftOutlined } from '@ant-design/icons'
import { useMutation, useQuery } from '@tanstack/react-query'
import {
  Alert,
  AutoComplete,
  Button,
  Card,
  Checkbox,
  Collapse,
  Form,
  Input,
  Select,
  Space,
  Tag,
  Typography,
  message,
} from 'antd'
import type { TextAreaRef } from 'antd/es/input/TextArea'
import { useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { OperatorCausality, api } from '../api/client'

function useDebounced<T>(value: T, delay: number) {
  const [debounced, setDebounced] = useState(value)
  useEffect(() => {
    const timer = window.setTimeout(() => setDebounced(value), delay)
    return () => window.clearTimeout(timer)
  }, [value, delay])
  return debounced
}

const SYMBOLS: Array<[string, string]> = [
  ['c', '收盘价'],
  ['o', '开盘价'],
  ['h', '最高价'],
  ['l', '最低价'],
  ['vol', '成交量'],
  ['amt', '成交额'],
  ['vwap', 'VWAP 代理 (h+l)/2'],
  ['cap', '总市值'],
]

function groupOperators(operators: OperatorCausality[]) {
  const groups = new Map<string, OperatorCausality[]>()
  for (const operator of operators) {
    const group = groups.get(operator.group) || []
    group.push(operator)
    groups.set(operator.group, group)
  }
  return [...groups.entries()].map(([group, items]) => ({ group, items }))
}

export default function FactorCreate() {
  const navigate = useNavigate()
  const [form] = Form.useForm()
  const [expression, setExpression] = useState('')
  const expressionRef = useRef<TextAreaRef>(null)
  // 在光标处插入符号并保持焦点，而不是追加到表达式末尾
  const insertSymbol = (symbol: string) => {
    const textArea = expressionRef.current?.resizableTextArea?.textArea
    const current: string = form.getFieldValue('expression') || ''
    const start = textArea?.selectionStart ?? current.length
    const end = textArea?.selectionEnd ?? current.length
    const next = current.slice(0, start) + symbol + current.slice(end)
    form.setFieldValue('expression', next)
    setExpression(next)
    window.setTimeout(() => {
      textArea?.focus()
      textArea?.setSelectionRange(start + symbol.length, start + symbol.length)
    }, 0)
  }
  const usesProxy = Form.useWatch('uses_proxy', form)
  const debounced = useDebounced(expression, 450)
  const validation = useQuery({
    queryKey: ['expression-validation', debounced],
    queryFn: () => api.validateExpression(debounced),
    enabled: debounced.trim().length > 0,
    retry: false,
  })
  const operatorCatalog = useQuery({
    queryKey: ['operator-causality'],
    queryFn: api.operatorCausality,
  })
  const tagCatalog = useQuery({
    queryKey: ['factor-tags', 'test'],
    queryFn: () => api.factorTags('test'),
  })
  const projectCatalog = useQuery({
    queryKey: ['test-factors'],
    queryFn: api.testFactors,
  })
  const operatorsByName = useMemo(() => {
    const mapping = new Map<string, OperatorCausality>()
    for (const operator of operatorCatalog.data || []) mapping.set(operator.name, operator)
    return mapping
  }, [operatorCatalog.data])
  const operatorGroups = useMemo(
    () => groupOperators(operatorCatalog.data || []),
    [operatorCatalog.data],
  )
  const projectOptions = useMemo(
    () => [...new Set((projectCatalog.data || []).map((factor) => factor.project))]
      .sort((left, right) => left.localeCompare(right, 'zh-CN'))
      .map((project) => ({ value: project })),
    [projectCatalog.data],
  )
  const mutation = useMutation({
    mutationFn: api.createTestFactor,
    onSuccess: (factor) => {
      message.success('因子已保存到测试库')
      navigate(`/test-factors/${factor.batch_id}/${factor.factor_name}`)
    },
    onError: (error) => message.error(error.message),
  })

  return (
    <>
      <div className="page-heading">
        <div className="page-heading-copy">
          <Button
            type="text"
            icon={<ArrowLeftOutlined />}
            onClick={() => navigate('/test-factors')}
            className="back-button"
          >
            返回测试库
          </Button>
          <Typography.Title level={2}>新增测试因子</Typography.Title>
          <Typography.Text type="secondary">
            表达式实时校验（与 CLI 相同的 AST 白名单），保存后进入测试库反复回测，确认后再提交到因子库
          </Typography.Text>
        </div>
      </div>
      <div className="factor-create-grid">
        <div>
          <Card className="surface-card">
            <Form
              form={form}
              layout="vertical"
              onFinish={(values) => mutation.mutate(values)}
              initialValues={{ project: '自定义因子', uses_proxy: false, paper_expression: '', proxy_description: '' }}
            >
              <Form.Item
                label="因子名称"
                name="factor_name"
                extra="请用新的名称保留不同研究假设。同名只应用于修正原有表达式。"
                rules={[
                  { required: true, message: '请输入因子名称' },
                  { pattern: /^[a-z0-9][a-z0-9_-]*$/, message: '仅小写字母、数字、下划线和连字符，且以字母或数字开头' },
                ]}
              >
                <Input placeholder="my_factor_001" />
              </Form.Item>
              <Form.Item
                label="项目"
                name="project"
                extra="用于在测试库和因子库中分组。可选择已有项目，也可直接输入新项目，例如：华夏191。"
                rules={[{ required: true, whitespace: true, message: '请输入项目名称' }]}
              >
                <AutoComplete options={projectOptions} filterOption={(input, option) =>
                  String(option?.value || '').toLocaleLowerCase().includes(input.toLocaleLowerCase())
                }>
                  <Input placeholder="例如：Alpha101、华夏191" />
                </AutoComplete>
              </Form.Item>
              <Form.Item label="表达式" name="expression" rules={[{ required: true, message: '请输入表达式' }]}>
                <Input.TextArea
                  ref={expressionRef}
                  rows={6}
                  className="expression-input"
                  onChange={(event) => setExpression(event.target.value)}
                  placeholder="例如：ts_mean(np.abs((h-l)/(h+l+1e-6)), 20)"
                />
              </Form.Item>
              {validation.data && (
                <Alert
                  type="success"
                  showIcon
                  title="表达式有效"
                  description={
                    <Space orientation="vertical" size={6}>
                      <Space wrap>
                        所需数据：
                        {validation.data.symbols.map((symbol) => (
                          <Tag key={symbol}>{symbol}</Tag>
                        ))}
                      </Space>
                      {validation.data.operators.length > 0 && (
                        <Space wrap>
                          使用算子：
                          {validation.data.operators.map((name) => {
                            const operator = operatorsByName.get(name)
                            return (
                              <Tag key={name} color={operator?.passed ? 'green' : 'default'}>
                                {operator?.signature || name}
                              </Tag>
                            )
                          })}
                        </Space>
                      )}
                    </Space>
                  }
                  style={{ marginBottom: 20 }}
                />
              )}
              {validation.isError && (
                <Alert
                  type="error"
                  showIcon
                  title="表达式无效"
                  description={(validation.error as Error).message}
                  style={{ marginBottom: 20 }}
                />
              )}
              <Form.Item label="论文表达式（可选，仅备注用）" name="paper_expression">
                <Input.TextArea rows={2} />
              </Form.Item>
              <Form.Item
                label="研究标签（可选）"
                name="tags"
                extra="用于保存固定研究组合。输入后按 Enter，或用逗号分隔。"
              >
                <Select
                  mode="tags"
                  tokenSeparators={[',', '，']}
                  placeholder="例如：回测组合、候选池"
                  options={(tagCatalog.data || []).map((item) => ({
                    value: item.tag,
                    label: `${item.tag}（${item.count}）`,
                  }))}
                />
              </Form.Item>
              <Form.Item name="uses_proxy" valuePropName="checked" style={{ marginBottom: usesProxy ? undefined : 8 }}>
                <Checkbox>使用了代理数据口径（如 vwap 实为 (h+l)/2）</Checkbox>
              </Form.Item>
              {usesProxy && (
                <Form.Item label="代理口径说明" name="proxy_description">
                  <Input placeholder="例如：vwap 使用 (high+low)/2 代理" />
                </Form.Item>
              )}
              <Button
                type="primary"
                htmlType="submit"
                loading={mutation.isPending}
                disabled={!validation.data}
              >
                保存到测试库
              </Button>
              {!validation.data && expression.trim() && !validation.isError && (
                <Typography.Text type="secondary" style={{ marginLeft: 12 }}>
                  校验中…
                </Typography.Text>
              )}
            </Form>
          </Card>
        </div>
        <aside>
          <Card className="surface-card" title="表达式速查" size="small">
            <Typography.Text type="secondary">数据符号（点击插入）</Typography.Text>
            <div className="symbol-chips">
              {SYMBOLS.map(([symbol, label]) => (
                <Tag
                  key={symbol}
                  className="clickable-tag"
                  onClick={() => insertSymbol(symbol)}
                >
                  <code>{symbol}</code> {label}
                </Tag>
              ))}
            </div>
            <Collapse
              ghost
              size="small"
              defaultActiveKey={['时间序列']}
              items={operatorGroups.map((section) => ({
                key: section.group,
                label: section.group,
                children: (
                  <div className="operator-list">
                    {section.items.map((operator) => (
                      <div
                        key={operator.name}
                        className={`operator-row ${operator.passed ? 'operator-row-safe' : ''}`}
                      >
                        <Space align="center" wrap>
                          <Typography.Text code>{operator.signature}</Typography.Text>
                          {operator.passed && <Tag color="green">因果通过</Tag>}
                        </Space>
                        <Typography.Text type="secondary">{operator.description}</Typography.Text>
                      </div>
                    ))}
                  </div>
                ),
              }))}
            />
            {operatorCatalog.isLoading && (
              <Typography.Text type="secondary">算子因果白名单加载中…</Typography.Text>
            )}
          </Card>
        </aside>
      </div>
    </>
  )
}
