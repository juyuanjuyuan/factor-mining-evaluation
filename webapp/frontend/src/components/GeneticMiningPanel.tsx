import {
  BranchesOutlined,
  PauseCircleOutlined,
  PlayCircleOutlined,
  ReloadOutlined,
} from '@ant-design/icons'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import {
  Alert,
  Button,
  Card,
  Checkbox,
  Collapse,
  Descriptions,
  Drawer,
  Empty,
  Flex,
  Form,
  Input,
  InputNumber,
  Popconfirm,
  Progress,
  Select,
  Space,
  Statistic,
  Steps,
  Tag,
  Typography,
  message,
} from 'antd'
import { useEffect, useState } from 'react'
import { GeneticCampaign, GeneticCampaignInput, api } from '../api/client'
import { shortTime } from '../lib/metrics'

const PREFERRED_TRAIN_END = '2019-09-30'
const PREFERRED_TEST_START = '2019-10-10'

const statusLabel: Record<string, string> = {
  created: '待启动',
  running: '运行中',
  stopped: '已停止',
  succeeded: '已完成',
  failed: '失败',
}

const statusColor: Record<string, string> = {
  created: 'default',
  running: 'blue',
  stopped: 'gold',
  succeeded: 'green',
  failed: 'red',
}

const preprocessLabel: Record<GeneticCampaignInput['preprocess_mode'], string> = {
  paper_local: '论文本地适配（四风格）',
  market_cap_industry: '市值 + 行业联合中性化',
  market_cap: '仅市值中性（旧任务）',
  none: '不预处理',
}

function campaignName() {
  const now = new Date()
  const part = (value: number) => String(value).padStart(2, '0')
  return `gp_web_${now.getFullYear()}${part(now.getMonth() + 1)}${part(now.getDate())}_${part(now.getHours())}${part(now.getMinutes())}${part(now.getSeconds())}`
}

function defaultDates(days: string[]) {
  if (!days.length) return { train_start: '', train_end: '', test_start: '', test_end: '' }
  let trainEnd = [...days].reverse().find((day) => day <= PREFERRED_TRAIN_END)
  let testStart = days.find((day) => day >= PREFERRED_TEST_START)
  if (!trainEnd || !testStart || trainEnd >= testStart) {
    const split = Math.max(61, Math.min(days.length - 62, Math.floor(days.length * 0.6)))
    trainEnd = days[split]
    testStart = days[split + 1]
  }
  return {
    train_start: days[0],
    train_end: trainEnd,
    test_start: testStart,
    test_end: days[days.length - 1],
  }
}

function campaignStep(campaign: GeneticCampaign) {
  if (campaign.status === 'succeeded') return 4
  if (campaign.current_stage === 'running') return 0
  if (campaign.current_stage === 'testing') return 1
  if (campaign.factor_library_pending_count > 0) return 3
  if (campaign.latest_cycle) return 4
  return 0
}

function CandidateRows({ campaign }: { campaign: GeneticCampaign }) {
  const candidates = campaign.latest_cycle?.candidates || []
  if (!candidates.length) {
    return <Typography.Text type="secondary">本轮候选尚未生成。</Typography.Text>
  }
  return (
    <div className="gp-candidate-list">
      {candidates.map((candidate) => {
        const profitPassed = Boolean(candidate.standard_gates?.profitability_test?.passed)
        const icChecked = candidate.ic_checked === true || candidate.standard_gates?.ic_test !== undefined
        const icPassed = Boolean(candidate.standard_gates?.ic_test?.passed)
        const submission = candidate.factor_library_submission
        const submissionTag = !candidate.test_overall_passed
          ? { color: candidate.status === 'failed' ? 'red' : 'default', text: candidate.status === 'failed' ? '测试异常' : '未通过测试' }
          : submission?.status === 'admitted'
            ? { color: 'green', text: submission.already_present ? '已在因子库' : submission.replaced_factor_names?.length ? `已入库并替换 ${submission.replaced_factor_names.length} 个高相关因子` : '因子库已入库' }
            : submission?.status === 'rejected_correlation' || submission?.status === 'rejected_performance'
              ? { color: 'orange', text: submission.status === 'rejected_performance' ? '高相关且性能未胜出，保留测试库' : '因子库相关性未通过' }
              : submission?.status === 'failed' || submission?.status === 'name_conflict'
                ? { color: 'red', text: '因子库提交失败' }
                : candidate.factor_library_submission_requested !== true
                  ? { color: 'gold', text: '旧任务：未交接因子库' }
                  : { color: 'blue', text: '已提交因子库，等待相关性检验' }
        return (
          <div className="gp-candidate-row" key={candidate.factor_name}>
            <div>
              <Typography.Text strong>{candidate.factor_name}</Typography.Text>
              <Typography.Text code ellipsis={{ tooltip: candidate.expression }}>
                {candidate.expression}
              </Typography.Text>
            </div>
            <Space wrap size={[4, 4]}>
              <Tag color={profitPassed ? 'green' : 'red'}>盈利 {profitPassed ? '通过' : '未通过'}</Tag>
              <Tag color={!icChecked ? 'default' : icPassed ? 'green' : 'red'}>
                IC {!icChecked ? '未检' : icPassed ? '通过' : '未通过'}
              </Tag>
              <Tag color={submissionTag.color}>
                {submissionTag.text}
              </Tag>
            </Space>
          </div>
        )
      })}
    </div>
  )
}

function CampaignCard({ campaign }: { campaign: GeneticCampaign }) {
  const queryClient = useQueryClient()
  const refresh = () => queryClient.invalidateQueries({ queryKey: ['genetic-campaigns'] })
  const start = useMutation({
    mutationFn: () => api.startGeneticCampaign(campaign.campaign),
    onSuccess: () => {
      message.success('遗传挖掘已继续运行')
      refresh()
    },
    onError: (error) => message.error(error.message),
  })
  const stop = useMutation({
    mutationFn: () => api.stopGeneticCampaign(campaign.campaign),
    onSuccess: () => {
      message.success('已停止遗传挖掘；checkpoint 保留，可继续运行')
      refresh()
    },
    onError: (error) => message.error(error.message),
  })
  const currentGeneration = campaign.current_stage === 'completed' || campaign.status === 'succeeded'
    ? `${campaign.config.generations}/${campaign.config.generations} 代已完成`
    : campaign.current_generation
      ? `第 ${campaign.current_generation}/${campaign.config.generations} 代`
      : null
  const restartLabel = campaign.status === 'succeeded' ? '继续下一轮' : '从 checkpoint 继续'
  return (
    <Card className="gp-campaign-card" size="small">
      <Flex justify="space-between" align="flex-start" gap={12} wrap>
        <div>
          <Space wrap>
            <Typography.Text strong>{campaign.campaign}</Typography.Text>
            <Tag color={statusColor[campaign.status]}>{statusLabel[campaign.status] || campaign.status}</Tag>
            {campaign.config.continuous && <Tag>连续模式</Tag>}
            <Tag color={campaign.config.compute_backend === 'mps' ? 'purple' : 'default'}>
              {campaign.config.compute_backend === 'mps' ? 'Apple GPU / MPS' : 'CPU'}
            </Tag>
          </Space>
          <Typography.Text type="secondary" className="gp-campaign-dates">
            训练 {campaign.config.train_start} ~ {campaign.config.train_end} · 测试 {campaign.config.test_start} ~ {campaign.config.test_end}
          </Typography.Text>
        </div>
        <Space>
          {campaign.status === 'running' ? (
            <Popconfirm
              title="停止遗传挖掘？"
              description="当前 checkpoint 会保留，之后可以继续。"
              onConfirm={() => stop.mutate()}
            >
              <Button danger icon={<PauseCircleOutlined />} loading={stop.isPending}>停止</Button>
            </Popconfirm>
          ) : (
            <Button icon={<PlayCircleOutlined />} loading={start.isPending} onClick={() => start.mutate()}>
              {restartLabel}
            </Button>
          )}
        </Space>
      </Flex>
      {campaign.legacy_auto_admission && (
        <Alert
          className="gp-contract-alert"
          type="warning"
          showIcon
          title="这是旧版自动入库 campaign"
          description="旧进程会在 GP 内直接执行相关性准入。请停止后以新名称创建任务；新版只向因子库服务交接通过测试的因子，由因子库完成相关性检验与入库裁决。"
        />
      )}
      <Steps
        size="small"
        current={campaignStep(campaign)}
        status={campaign.status === 'failed' ? 'error' : campaign.status === 'stopped' ? 'wait' : 'process'}
        items={[
          { title: '训练集进化', content: currentGeneration || '等待种群计算' },
          { title: '测试集盈利标准', content: campaign.current_stage === 'testing' ? '先筛冻结表达式' : '市值+行业中性化后的净分组收益' },
          { title: '测试集 IC', content: '仅对盈利存活者检测' },
          { title: '提交因子库', content: '相关性冲突时以 60 日 Sharpe 中位数、再以 Fitness 裁决替换' },
          { title: '完成本轮', content: `${campaign.completed_cycles} 个 cycle` },
        ]}
      />
      {campaign.current_generation_total && campaign.current_stage === 'running' ? (
        <div className="gp-generation-progress">
          <Progress
            percent={Math.round((campaign.current_generation_progress || 0) * 1000) / 10}
            status={campaign.status === 'failed' ? 'exception' : 'active'}
          />
          <Typography.Text type="secondary">
            当前代已计算 {campaign.current_generation_completed || 0} / {campaign.current_generation_total}
            {campaign.current_generation_failed ? ` · 无效公式 ${campaign.current_generation_failed}` : ''}
          </Typography.Text>
        </div>
      ) : null}
      <div className="gp-campaign-stats">
        <Statistic title="当前 cycle" value={campaign.current_cycle ?? '—'} />
        <Statistic title="测试通过" value={campaign.test_passed_count} />
        <Statistic title="因子库已入库" value={campaign.factor_library_admitted_count} styles={{ content: campaign.factor_library_admitted_count ? { color: '#008A3E' } : undefined }} />
        <Statistic title="因子库冲突未入库" value={campaign.factor_library_rejected_count} />
        <Statistic title="候选失败" value={campaign.failed_candidate_count} />
      </div>
      <Collapse
        ghost
        size="small"
        items={[
          {
            key: 'candidates',
            label: `最近 cycle 候选${campaign.latest_cycle ? `（${campaign.latest_cycle.candidates.length}）` : ''}`,
            children: <CandidateRows campaign={campaign} />,
          },
          {
            key: 'contract',
            label: '固定口径与运行记录',
            children: (
              <>
                <Descriptions size="small" column={{ xs: 1, sm: 2, md: 3 }}>
                  <Descriptions.Item label="种群/代数">{campaign.config.population_size} / {campaign.config.generations}</Descriptions.Item>
                  <Descriptions.Item label="冻结测试候选（Pareto 多层排序前 N）">{campaign.config.components}</Descriptions.Item>
                  <Descriptions.Item label="预处理">{preprocessLabel[campaign.config.preprocess_mode] || campaign.config.preprocess_mode}</Descriptions.Item>
                  <Descriptions.Item label="训练后端">{campaign.config.compute_backend === 'mps' ? 'MPS / Apple GPU' : 'CPU'}</Descriptions.Item>
                  <Descriptions.Item label="CPU 线程">{campaign.config.n_jobs}</Descriptions.Item>
                  <Descriptions.Item label="创建时间">{shortTime(campaign.created_at)}</Descriptions.Item>
                </Descriptions>
                {(campaign.error || campaign.stderr_tail || campaign.stdout_tail) && (
                  <pre className="gp-log-tail">{campaign.error || campaign.stderr_tail || campaign.stdout_tail}</pre>
                )}
                <Typography.Text type="secondary" copyable={{ text: campaign.output_dir }}>
                  输出目录：{campaign.output_dir}
                </Typography.Text>
              </>
            ),
          },
        ]}
      />
    </Card>
  )
}

export function GeneticMiningPanel({ open, onOpen, onClose }: { open: boolean; onOpen: () => void; onClose: () => void }) {
  const [form] = Form.useForm<GeneticCampaignInput>()
  const [panelKeys, setPanelKeys] = useState<string[]>([])
  const queryClient = useQueryClient()
  const timeline = useQuery({ queryKey: ['market-timeline'], queryFn: api.marketTimeline, staleTime: Infinity })
  const backends = useQuery({ queryKey: ['genetic-fitness-backends'], queryFn: api.geneticFitnessBackends, staleTime: Infinity })
  const campaigns = useQuery({
    queryKey: ['genetic-campaigns'],
    queryFn: api.geneticCampaigns,
    refetchInterval: (query) => query.state.data?.some((item) => item.status === 'running') ? 2500 : 10000,
  })
  const continuous = Form.useWatch('continuous', form)
  const computeBackend = Form.useWatch('compute_backend', form)
  useEffect(() => {
    if (!open || !timeline.data?.trading_days.length) return
    const dates = defaultDates(timeline.data.trading_days)
    const mpsAvailable = backends.data?.some((backend) => backend.name === 'mps' && backend.available) || false
    form.setFieldsValue({
      campaign: campaignName(),
      ...dates,
      horizon: 1,
      n_quantiles: 10,
      preprocess_mode: 'paper_local',
      population_size: 1000,
      generations: 3,
      hall_of_fame: 100,
      components: 100,
      tournament_size: 20,
      compute_backend: mpsAvailable ? 'mps' : 'cpu',
      n_jobs: mpsAvailable ? 1 : 2,
      seed: 20190610,
      continuous: false,
      pause_seconds: 60,
      max_cycles: null,
    })
  }, [backends.data, form, open, timeline.data])
  useEffect(() => {
    if (computeBackend === 'mps' && form.getFieldValue('n_jobs') !== 1) {
      form.setFieldValue('n_jobs', 1)
    }
  }, [computeBackend, form])
  const create = useMutation({
    mutationFn: (values: GeneticCampaignInput) => api.createGeneticCampaign({
      ...values,
      // The current GP protocol freezes up to N candidates from the Pareto-ranked HOF.
      // Keep the archive/test caps aligned instead of exposing two copies of N.
      components: values.hall_of_fame,
      max_cycles: values.continuous ? values.max_cycles || null : null,
    }),
    onSuccess: (campaign) => {
      message.success(`已启动 ${campaign.campaign}`)
      queryClient.invalidateQueries({ queryKey: ['genetic-campaigns'] })
      onClose()
    },
    onError: (error) => message.error(error.message),
  })
  const days = timeline.data?.trading_days || []
  const dateOptions = days.map((day) => ({ value: day, label: day }))
  const active = (campaigns.data || []).find((campaign) => campaign.status === 'running')
  useEffect(() => {
    if (active) setPanelKeys(['campaigns'])
  }, [active])

  return (
    <>
      <Collapse
        className="collapse-card gp-campaign-panel"
        activeKey={panelKeys}
        onChange={(keys) => setPanelKeys(Array.isArray(keys) ? keys.map(String) : [String(keys)])}
        items={[
          {
            key: 'campaigns',
            label: (
              <Flex justify="space-between" align="center" gap={12} wrap>
                <Space wrap>
                  <BranchesOutlined />
                  <span>遗传规划挖掘任务</span>
                  {active ? <Tag color="blue">{active.campaign} 运行中</Tag> : <Tag>当前无运行任务</Tag>}
                  <Typography.Text type="secondary" style={{ fontWeight: 'normal' }}>
                    独立进程，不阻塞普通因子评价
                  </Typography.Text>
                </Space>
                <Button size="small" icon={<ReloadOutlined />} onClick={(event) => {
                  event.stopPropagation()
                  campaigns.refetch()
                }}>刷新</Button>
              </Flex>
            ),
            children: campaigns.isError ? (
              <Alert type="error" showIcon title="无法读取遗传挖掘任务" description={campaigns.error.message} />
            ) : campaigns.data?.length ? (
              <div className="gp-campaign-list">
                {campaigns.data.map((campaign) => <CampaignCard campaign={campaign} key={campaign.campaign} />)}
              </div>
            ) : (
              <Empty description="尚未创建遗传挖掘任务">
                <Button type="primary" icon={<BranchesOutlined />} onClick={onOpen}>创建第一个任务</Button>
              </Empty>
            ),
          },
        ]}
      />
      <Drawer
        title="遗传规划添加因子"
        width={720}
        open={open}
        onClose={onClose}
        destroyOnHidden
        extra={<Tag color="blue">训练挖掘 / 测试筛选</Tag>}
      >
        <Alert
          className="gp-contract-alert"
          type="info"
          showIcon
          title="冻结训练/测试边界"
          description="训练集只用于表达式进化、规范化去重和多层 Pareto HOF 排名；第一前沿优先，随后依次使用后续前沿补足冻结候选。父代锦标赛仍使用 IC 减复杂度惩罚。冻结后的表达式先用市值+行业中性化后的净分组收益筛选；仅盈利存活者再做测试集 IC 检测。两关均通过后会自动交接给因子库；若相关性冲突，因子库以同口径的 60 日窗口 Sharpe 中位数优先、同分再以 Fitness 比较，只有严格更优的候选才能替换旧正式因子，旧定义保留在测试库。"
        />
        {active && (
          <Alert
            className="gp-contract-alert"
            type="warning"
            showIcon
            title={`当前已有任务 ${active.campaign} 运行中`}
            description="为控制宽矩阵内存占用，Webapp 同时只运行一个遗传挖掘任务。"
          />
        )}
        <Form form={form} layout="vertical" onFinish={(values) => create.mutate(values)}>
          <Form.Item
            label="Campaign 名称"
            name="campaign"
            extra="配置和 checkpoint 都绑定此名称；改变日期或参数时请使用新名称。"
            rules={[
              { required: true, message: '请输入 Campaign 名称' },
              { pattern: /^[a-z0-9][a-z0-9_-]*$/, message: '仅小写字母、数字、下划线和连字符' },
            ]}
          >
            <Input />
          </Form.Item>
          <Typography.Text strong>训练 / 测试区间</Typography.Text>
          <div className="gp-date-grid">
            {[
              ['train_start', '训练开始'],
              ['train_end', '训练结束'],
              ['test_start', '测试开始'],
              ['test_end', '测试结束'],
            ].map(([name, label]) => (
              <Form.Item key={name} label={label} name={name} rules={[{ required: true, message: `请选择${label}` }]}>
                <Select showSearch optionFilterProp="label" options={dateOptions} loading={timeline.isLoading} />
              </Form.Item>
            ))}
          </div>
          <Alert
            className="gp-contract-alert"
            type="warning"
            showIcon
            title="固定测试集反复筛选会逐渐变成验证集"
            description="连续挖掘适合构建候选库；若需要最终无偏结论，应另留一段不参与 GP 测试筛选的最终留出期。"
          />
          <div className="gp-parameter-grid">
            <Form.Item label="收益周期 H" name="horizon" rules={[{ required: true }]}>
              <InputNumber min={1} max={60} />
            </Form.Item>
            <Form.Item label="分组数量" name="n_quantiles" rules={[{ required: true }]}>
              <InputNumber min={3} max={20} />
            </Form.Item>
            <Form.Item label="训练预处理" name="preprocess_mode" rules={[{ required: true }]}>
              <Select options={[
                { value: 'paper_local', label: '论文本地适配（推荐）' },
                { value: 'market_cap_industry', label: '市值 + 行业联合中性化' },
                { value: 'none', label: '不预处理' },
              ]} />
            </Form.Item>
            <Form.Item
              label="训练计算后端"
              name="compute_backend"
              rules={[{ required: true }]}
              extra={backends.data?.find((backend) => backend.name === 'mps')?.reason || '测试集盈利与 IC 标准均始终使用 CPU 评价。'}
            >
              <Select options={(backends.data || [
                { name: 'cpu', available: true, device_name: 'NumPy/Pandas CPU' },
              ]).map((backend) => ({
                value: backend.name,
                disabled: !backend.available,
                label: backend.name === 'mps'
                  ? `MPS · ${backend.device_name || 'Apple GPU'}${backend.available ? '（推荐）' : '（不可用）'}`
                  : 'CPU · NumPy/Pandas',
              }))} />
            </Form.Item>
            <Form.Item label="CPU 并行线程" name="n_jobs" rules={[{ required: true }]} extra={computeBackend === 'mps' ? 'MPS 模式下固定为 1，并行由 GPU 提供。' : undefined}>
              <InputNumber min={1} max={32} disabled={computeBackend === 'mps'} />
            </Form.Item>
          </div>
          <Collapse
            className="gp-advanced"
            items={[
              {
                key: 'evolution',
                label: '进化参数（阶段 1 优化）',
                children: (
                  <div className="gp-parameter-grid">
                    <Form.Item label="种群规模" name="population_size" rules={[{ required: true }]}>
                      <InputNumber min={2} max={10000} />
                    </Form.Item>
                    <Form.Item label="进化代数" name="generations" rules={[{ required: true }]}>
                      <InputNumber min={1} max={100} />
                    </Form.Item>
                    <Form.Item
                      label="Pareto HOF / 冻结测试候选数"
                      name="hall_of_fame"
                      rules={[{ required: true }]}
                      extra="跨代按同向训练 IC 与节点数执行多层 Pareto 排名；第一前沿不足 N 时依次从后续前沿补足，只有有效且唯一表达式不足时才少于 N。"
                    >
                      <InputNumber min={1} max={500} />
                    </Form.Item>
                    <Form.Item label="锦标赛规模" name="tournament_size" rules={[{ required: true }]}>
                      <InputNumber min={1} max={1000} />
                    </Form.Item>
                    <Form.Item label="随机种子" name="seed" rules={[{ required: true }]}>
                      <InputNumber min={0} max={4294967295} />
                    </Form.Item>
                  </div>
                ),
              },
            ]}
          />
          <Card size="small" className="gp-run-mode">
            <Form.Item name="continuous" valuePropName="checked" style={{ marginBottom: continuous ? 14 : 0 }}>
              <Checkbox>连续运行（每个 cycle 使用新随机种子）</Checkbox>
            </Form.Item>
            {continuous && (
              <div className="gp-parameter-grid compact">
                <Form.Item label="Cycle 间隔（秒）" name="pause_seconds" rules={[{ required: true }]}>
                  <InputNumber min={0} max={86400} />
                </Form.Item>
                <Form.Item label="本次启动最多 Cycle（留空为无限）" name="max_cycles">
                  <InputNumber min={1} max={100000} placeholder="无限" />
                </Form.Item>
              </div>
            )}
          </Card>
          <Flex justify="flex-end" gap={10} style={{ marginTop: 22 }}>
            <Button onClick={onClose}>取消</Button>
            <Button type="primary" htmlType="submit" icon={<BranchesOutlined />} loading={create.isPending} disabled={Boolean(active)}>
              启动挖掘
            </Button>
          </Flex>
        </Form>
      </Drawer>
    </>
  )
}
