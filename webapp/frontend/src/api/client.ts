export type GateOp = 'gte' | 'lte' | 'gt' | 'lt' | 'between'

export type GateCondition = {
  metric: string
  op: GateOp
  value: number
  value2?: number | null
}

export type GateSpec = {
  conditions: GateCondition[]
  match: 'all' | 'any'
}

export type GateConditionResult = GateCondition & {
  actual: number | null
  passed: boolean
  missing: boolean
}

export type MarketCycleBackground = {
  start_day: string
  end_day: string | null
  direction: 'up' | 'down'
  label: string
  provisional: boolean
}

export type Run = {
  id: string
  job_id: string
  factor_name: string
  batch_id?: string
  expression: string
  stage?: string
  gate_outcome?: string
  gate_value?: GateConditionResult[] | number | null
  gate_explanation?: string
  status: string
  horizon: number
  n_quantiles: number
  methods: string[]
  run_params?: Record<string, unknown>
  result?: Record<string, unknown>
  error?: string
  created_at: string
  started_at?: string
  finished_at?: string
  detail_names?: string[]
  market_cycle_backgrounds?: MarketCycleBackground[]
}

export type Factor = {
  number: number
  factor_name: string
  batch_id: string
  batch_name: string
  project: string
  entered_at?: string
  updated_at?: string
  expression: string
  required_symbols: string[]
  implementation_set: string
  library_scope?: 'test' | 'factor'
  factor_category?: string
  uses_proxy: boolean
  editable: boolean
  submitted?: boolean
  tags: string[]
  latest_run?: Run
  runs?: Run[]
}

export type FactorTag = {
  tag: string
  count: number
}

export type MarketTimeline = {
  trading_days: string[]
  start_day: string
  end_day: string
  count: number
}

export type FactorCorrelationMatrix = {
  factor_names: string[]
  matrix: number[][]
  threshold: number
  method: 'pooled_pearson'
  sample_definition: string
  max_abs_off_diagonal: number
  passed: boolean
  violations: Array<{
    factor_a: string
    factor_b: string
    correlation: number
    abs_correlation: number
  }>
  updated_at?: string | null
}

export type FactorCorrelationWindow = {
  window_number: number
  start_day: string
  end_day: string
  correlation: number | null
  paired_observations: number
  exceeds_threshold: boolean
}

export type FactorCorrelationPair = {
  factor_a: string
  factor_b: string
  method: 'pooled_pearson'
  window_size: number
  step_size: number
  threshold: number
  sample_definition: string
  max_abs_correlation: number | null
  violation_window_count: number
  windows: FactorCorrelationWindow[]
}

export type Job = {
  id: string
  kind: 'evaluate' | 'funnel'
  title: string
  status: string
  total_runs: number
  finished_runs: number
  created_at: string
  runs?: Run[]
}

export type GeneticCampaignInput = {
  campaign: string
  train_start: string
  train_end: string
  test_start: string
  test_end: string
  horizon: number
  n_quantiles: number
  preprocess_mode: 'paper_local' | 'market_cap_industry' | 'market_cap' | 'none'
  population_size: number
  generations: number
  hall_of_fame: number
  components: number
  tournament_size: number
  n_jobs: number
  compute_backend: 'cpu' | 'mps'
  seed: number
  continuous: boolean
  pause_seconds: number
  max_cycles?: number | null
}

export type GeneticFitnessBackend = {
  name: 'cpu' | 'mps'
  available: boolean
  reason?: string | null
  device_name?: string | null
  torch_version?: string | null
}

export type GeneticCandidate = {
  factor_name: string
  expression: string
  status: string
  profitability_passed?: boolean
  ic_checked?: boolean
  ic_passed?: boolean | null
  test_overall_passed: boolean
  factor_library_submission_requested?: boolean
  factor_library_submission?: {
    status: 'pending' | 'admitted' | 'rejected_correlation' | 'name_conflict' | 'failed'
    factor_name?: string
    already_present?: boolean
    correlation_checked?: boolean
    correlation_passed?: boolean
    correlation_threshold?: number
    explanation?: string
    requested_at?: string
    completed_at?: string
  }
  error?: string
  standard_gates?: Record<string, { passed?: boolean }>
}

export type GeneticCycleSummary = {
  cycle: number
  status: string
  test_passed_count: number
  failed_count: number
  candidates: GeneticCandidate[]
}

export type GeneticCampaign = {
  campaign: string
  status: 'created' | 'running' | 'stopped' | 'succeeded' | 'failed'
  config: GeneticCampaignInput
  pid?: number | null
  process_alive: boolean
  output_dir: string
  current_cycle?: number | null
  current_stage?: 'running' | 'testing' | 'completed' | null
  current_generation?: number | null
  current_generation_completed?: number | null
  current_generation_total?: number | null
  current_generation_failed?: number | null
  current_generation_progress?: number | null
  completed_cycles: number
  test_passed_count: number
  failed_candidate_count: number
  factor_library_pending_count: number
  factor_library_admitted_count: number
  factor_library_rejected_count: number
  legacy_auto_admission?: boolean
  latest_cycle?: GeneticCycleSummary | null
  last_error?: Record<string, unknown> | null
  stdout_tail: string
  stderr_tail: string
  error?: string | null
  created_at: string
  started_at?: string | null
  updated_at: string
  finished_at?: string | null
}

export type ModelTerm = {
  batch_id: string
  factor_name: string
  weight: number
  expression?: string
}

export type ModelTestInput = {
  model_name: string
  terms: ModelTerm[]
  train_start: string
  train_end: string
  test_start: string
  test_end: string
  horizon: number
  n_quantiles: number
  methods: string[]
  training_method: string
  training_params: Record<string, number>
}

export type ModelTrainingParameter = {
  name: string
  label: string
  default: number
  minimum: number
  maximum: number
  step: number
  description: string
}

export type ModelTrainingMethod = {
  name: string
  label: string
  description: string
  requires_fitting: boolean
  term_weight_editable: boolean
  parameters: ModelTrainingParameter[]
}

export type ModelFitResult = {
  method: string
  parameters: Record<string, number>
  terms: ModelTerm[]
  expression: string
  diagnostics: Record<string, string | number | null>
}

export type ModelTest = ModelTestInput & {
  id: string
  expression: string
  training_run_id?: string | null
  testing_run_id?: string | null
  training_run?: Run | null
  testing_run?: Run | null
  created_at: string
  updated_at: string
  locked_at?: string | null
  locked: boolean
  can_run_testing: boolean
  fit_result?: ModelFitResult | null
}

export type Method = {
  name: string
  requires: string[]
  provides: string[]
  required_data_symbols: string[]
  is_default: boolean
}

export type OperatorCausality = {
  name: string
  signature: string
  description: string
  group: string
  expression: string
  passed: boolean
  prefix_passed: boolean
  perturbation_passed: boolean
  reason?: string
}

export type FunnelStage = {
  name: string
  methods: string[]
  gate_metric?: string
  required_methods: string[]
}

export type Template = {
  id: number
  name: string
  kind: 'methods' | 'funnel'
  methods: string[]
  params: Record<string, unknown>
  is_builtin: boolean
}

export type Detail = {
  name: string
  index: string[]
  columns: string[]
  data: Array<Array<number | string | boolean | null>>
  summary?: Record<string, number | null>
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`/api${path}`, {
    headers: { 'Content-Type': 'application/json', ...(init?.headers || {}) },
    ...init,
  })
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}))
    throw new Error(payload.detail || `请求失败（${response.status}）`)
  }
  if (response.status === 204) return undefined as T
  return response.json() as Promise<T>
}

export const api = {
  factors: () => request<Factor[]>('/factors'),
  factorCorrelation: () => request<FactorCorrelationMatrix>('/factor-correlation'),
  factorCorrelationPair: (factorA: string, factorB: string) => {
    const query = new URLSearchParams({ factor_a: factorA, factor_b: factorB })
    return request<FactorCorrelationPair>(`/factor-correlation/pair?${query.toString()}`)
  },
  testFactors: () => request<Factor[]>('/test-factors'),
  factor: (batchId: string, name: string) =>
    request<Factor>(`/factors/${encodeURIComponent(batchId)}/${encodeURIComponent(name)}`),
  createFactor: (body: Record<string, unknown>) =>
    request<Factor>('/factors', { method: 'POST', body: JSON.stringify(body) }),
  createTestFactor: (body: Record<string, unknown>) =>
    request<Factor>('/test-factors', { method: 'POST', body: JSON.stringify(body) }),
  factorTags: (library: 'test' | 'factor' = 'test') =>
    request<FactorTag[]>(`/factor-tags?library=${library}`),
  updateFactorTags: (batchId: string, name: string, tags: string[]) =>
    request<Factor>(`/factors/${encodeURIComponent(batchId)}/${encodeURIComponent(name)}/tags`, {
      method: 'PUT',
      body: JSON.stringify({ tags }),
    }),
  updateFactorProject: (batchId: string, name: string, project: string) =>
    request<Factor>(`/factors/${encodeURIComponent(batchId)}/${encodeURIComponent(name)}/project`, {
      method: 'PUT',
      body: JSON.stringify({ project }),
    }),
  submitFactor: (batchId: string, name: string) =>
    request<Factor>(`/test-factors/${encodeURIComponent(batchId)}/${encodeURIComponent(name)}/submit`, {
      method: 'POST',
    }),
  deleteFactor: (batchId: string, name: string) =>
    request<void>(`/factors/${encodeURIComponent(batchId)}/${encodeURIComponent(name)}`, { method: 'DELETE' }),
  removeFromFactorLibrary: (batchId: string, name: string) =>
    request<void>(`/factors/${encodeURIComponent(batchId)}/${encodeURIComponent(name)}`, { method: 'DELETE' }),
  validateExpression: (expression: string) =>
    request<{ valid: boolean; symbols: string[]; operators: string[] }>('/expressions/validate', {
      method: 'POST',
      body: JSON.stringify({ expression }),
    }),
  marketTimeline: () => request<MarketTimeline>('/market-data/timeline'),
  methods: () => request<Method[]>('/methods'),
  operatorCausality: () => request<OperatorCausality[]>('/operators/causality'),
  funnelStages: () => request<FunnelStage[]>('/funnel-stages'),
  resolveMethods: (names: string[]) =>
    request<{ methods: string[] }>('/methods/resolve', {
      method: 'POST',
      body: JSON.stringify({ names }),
    }),
  templates: () => request<Template[]>('/templates'),
  createTemplate: (body: Partial<Template>) =>
    request<Template>('/templates', { method: 'POST', body: JSON.stringify(body) }),
  updateTemplate: (id: number, body: Partial<Template>) =>
    request<Template>(`/templates/${id}`, { method: 'PUT', body: JSON.stringify(body) }),
  deleteTemplate: (id: number) => request<void>(`/templates/${id}`, { method: 'DELETE' }),
  jobs: () => request<Job[]>('/jobs'),
  job: (id: string) => request<Job>(`/jobs/${id}`),
  deleteJob: (id: string) => request<void>(`/jobs/${encodeURIComponent(id)}`, { method: 'DELETE' }),
  createJob: (body: Record<string, unknown>) =>
    request<Job>('/jobs', { method: 'POST', body: JSON.stringify(body) }),
  geneticCampaigns: () => request<GeneticCampaign[]>('/genetic-campaigns'),
  geneticFitnessBackends: () => request<GeneticFitnessBackend[]>('/genetic-campaigns/backends'),
  geneticCampaign: (campaign: string) =>
    request<GeneticCampaign>(`/genetic-campaigns/${encodeURIComponent(campaign)}`),
  createGeneticCampaign: (body: GeneticCampaignInput) =>
    request<GeneticCampaign>('/genetic-campaigns', { method: 'POST', body: JSON.stringify(body) }),
  startGeneticCampaign: (campaign: string) =>
    request<GeneticCampaign>(`/genetic-campaigns/${encodeURIComponent(campaign)}/start`, { method: 'POST' }),
  stopGeneticCampaign: (campaign: string) =>
    request<GeneticCampaign>(`/genetic-campaigns/${encodeURIComponent(campaign)}/stop`, { method: 'POST' }),
  models: () => request<ModelTest[]>('/models'),
  modelTrainingMethods: () => request<ModelTrainingMethod[]>('/models/training-methods'),
  model: (id: string) => request<ModelTest>(`/models/${encodeURIComponent(id)}`),
  createModel: (body: ModelTestInput) =>
    request<ModelTest>('/models', { method: 'POST', body: JSON.stringify(body) }),
  updateModel: (id: string, body: ModelTestInput) =>
    request<ModelTest>(`/models/${encodeURIComponent(id)}`, { method: 'PUT', body: JSON.stringify(body) }),
  deleteModel: (id: string) => request<void>(`/models/${encodeURIComponent(id)}`, { method: 'DELETE' }),
  trainModel: (id: string) => request<ModelTest>(`/models/${encodeURIComponent(id)}/train`, { method: 'POST' }),
  testModel: (id: string) => request<ModelTest>(`/models/${encodeURIComponent(id)}/test`, { method: 'POST' }),
  cancelJob: (id: string, force = false) =>
    request<Job>(`/jobs/${id}/cancel?force=${force}`, { method: 'POST' }),
  runs: (status?: string, factorName?: string) => {
    const query = new URLSearchParams()
    if (status) query.set('status', status)
    if (factorName) query.set('factor_name', factorName)
    const qs = query.toString()
    return request<Run[]>(`/runs${qs ? `?${qs}` : ''}`)
  },
  run: (id: string) => request<Run>(`/runs/${id}`),
  detail: (id: string, name: string) => request<Detail>(`/runs/${id}/details/${name}`),
  compare: (runIds: string[]) =>
    request<CompareResult>('/compare', {
      method: 'POST',
      body: JSON.stringify({ run_ids: runIds }),
    }),
}

export type CompareResult = {
  runs: Array<{ id: string; factor_name: string }>
  metric_rows: Array<{ metric: string; values: Record<string, number | boolean | null> }>
  curves: {
    top_quantile: Curve[]
    cumulative_ic: Curve[]
  }
}

export type Curve = {
  run_id: string
  name: string
  data: Array<[string, number | null]>
}
