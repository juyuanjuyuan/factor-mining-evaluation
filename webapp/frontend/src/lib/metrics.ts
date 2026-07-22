/**
 * 指标/方法/阶段的中文字典与格式化规则。
 * direction 用于对比页高亮：higher 越大越好、lower 越小越好、
 * abs 绝对值越大越好（IC/IR 这类带符号指标）、none 不高亮。
 */
import { METHOD_DEFINITIONS } from './methodDefinitions'

// 'drawdown_pct' 是回撤专用格式：底层数值是 ≤0 的分数（如 -0.2310），
// 但研究员习惯说“回撤 23%”，所以展示为正的百分比幅度。
export type MetricFormat = 'number' | 'percent' | 'pvalue' | 'int' | 'bool' | 'text' | 'drawdown_pct'
export type MetricDirection = 'higher' | 'lower' | 'abs' | 'none'

export type MetricSpec = {
  label: string
  tooltip?: string
  format: MetricFormat
  direction: MetricDirection
  group: string
}

export const METRIC_GROUPS = [
  'IC 分析',
  '显著性检验',
  '分位组收益',
  '头部组合',
  '滚动风险',
  '数据质量',
  '运行信息',
] as const

const spec = (
  label: string,
  group: string,
  format: MetricFormat = 'number',
  direction: MetricDirection = 'none',
  tooltip?: string,
): MetricSpec => ({ label, group, format, direction, tooltip })

export const METRIC_SPECS: Record<string, MetricSpec> = {
  ic_mean: spec('IC 均值', 'IC 分析', 'number', 'abs', '日度 Spearman 秩相关的均值，绝对值越大预测力越强'),
  ic_std: spec('IC 标准差', 'IC 分析', 'number', 'lower'),
  ir: spec('ICIR', 'IC 分析', 'number', 'abs', 'IC 均值 / IC 标准差，衡量稳定性'),
  ic_positive_ratio: spec('IC 为正占比', 'IC 分析', 'percent', 'none'),
  ic_horizon_decay_min_horizon: spec('IC 衰减最短持有期', 'IC 分析', 'int', 'none', '曲线起点 H=20'),
  ic_horizon_decay_max_horizon: spec('IC 衰减最远持有期', 'IC 分析', 'int', 'none', '曲线终点 H=252'),
  ic_horizon_decay_horizon_count: spec('IC 衰减期限数', 'IC 分析', 'int'),
  ic_horizon_decay_structural_signal_day_count: spec('H=252 可支持日期数', 'IC 分析', 'int', 'none', '仅按最大持有期和样本结束边界筛选后的信号日期数'),
  ic_horizon_decay_common_signal_day_count: spec('IC 衰减共同日期数', 'IC 分析', 'int', 'none', '在 H=20...252 每一期限均能形成有限 Rank IC 的固定日期集合大小'),
  ic_horizon_decay_dropped_signal_day_count: spec('共同样本剔除日期数', '数据质量', 'int', 'lower', '结构可用但至少一个期限无法形成有限 Rank IC 的日期数'),
  ic_horizon_decay_common_start_day: spec('IC 衰减共同样本起始日', '运行信息', 'text'),
  ic_horizon_decay_common_end_day: spec('IC 衰减共同样本结束日', '运行信息', 'text'),
  ic_horizon_decay_h20_mean_ic: spec('H=20 平均 Rank IC', 'IC 分析', 'number', 'abs'),
  ic_horizon_decay_h252_mean_ic: spec('H=252 平均 Rank IC', 'IC 分析', 'number', 'abs'),
  ic_peak_decay_window_days: spec('IC 滚动窗口', 'IC 分析', 'int', 'none', '每个窗口包含的有效日度 IC 观测数；当前为 60 日'),
  ic_peak_decay_step_days: spec('IC 滚动步长', 'IC 分析', 'int', 'none', '相邻 60 日窗口之间前移的有效日度 IC 观测数；当前为 10 日'),
  ic_peak_decay_overlap_days: spec('IC 窗口重叠', 'IC 分析', 'int', 'none', '相邻窗口重叠的有效日度 IC 观测数；当前为 50 日'),
  ic_peak_decay_valid_ic_days: spec('用于 Mean IC 的有效天数', 'IC 分析', 'int'),
  ic_peak_decay_full_window_count: spec('IC 完整滚动窗口数', 'IC 分析', 'int'),
  ic_trend_filter_process_to_observation_ratio: spec('卡尔曼 Q/R', 'IC 分析', 'number', 'none', '局部水平模型的过程噪声与观测噪声之比；当前固定为 0.01'),
  ic_trend_filter_kalman_gain: spec('卡尔曼更新权重', 'IC 分析', 'number', 'none', '新一天 IC 在趋势更新中的权重；Q/R=0.01 时约为 0.095'),
  ic_trend_filter_valid_ic_days: spec('趋势滤波有效 IC 天数', 'IC 分析', 'int'),
  ic_trend_filter_start: spec('卡尔曼趋势起点', 'IC 分析', 'number', 'none'),
  ic_trend_filter_final: spec('最新卡尔曼 IC 趋势', 'IC 分析', 'number', 'none', '最新日期仅使用截至该日期的日度 IC 得出的卡尔曼趋势值'),
  ic_trend_filter_change: spec('卡尔曼趋势变化', 'IC 分析', 'number', 'none', '最新卡尔曼 IC 趋势减去起点'),
  ic_trend_filter_second_order_period: spec('二阶低通周期参数', 'IC 分析', 'int', 'none', '递推公式中的周期参数；当前为 31 个有效 IC 观测'),
  ic_trend_filter_second_order_start: spec('二阶低通趋势起点', 'IC 分析', 'number', 'none'),
  ic_trend_filter_second_order_final: spec('最新二阶低通 IC 趋势', 'IC 分析', 'number', 'none', '因果二阶低通递推在最新日期的趋势值'),
  ic_trend_filter_second_order_change: spec('二阶低通趋势变化', 'IC 分析', 'number', 'none'),
  ic_trend_filter_fourier_min_period: spec('傅里叶最短保留周期', 'IC 分析', 'int', 'none', '保留周期不短于 31 个有效 IC 观测的频率成分'),
  ic_trend_filter_fourier_retained_frequency_count: spec('傅里叶保留频率数', 'IC 分析', 'int', 'none'),
  ic_trend_filter_fourier_start: spec('傅里叶重构起点', 'IC 分析', 'number', 'none'),
  ic_trend_filter_fourier_final: spec('最新傅里叶重构 IC', 'IC 分析', 'number', 'none', '全样本低频重构的末端值；仅用于事后诊断'),
  ic_trend_filter_fourier_change: spec('傅里叶重构变化', 'IC 分析', 'number', 'none'),
  ic_trend_filter_mean_window_days: spec('滤波后 IC Mean 窗口', 'IC 分析', 'int', 'none', '每个均值窗口包含 10 个有效 IC 观测'),
  ic_trend_filter_mean_step_days: spec('滤波后 IC Mean 步长', 'IC 分析', 'int', 'none', '相邻窗口前移 5 个有效 IC 观测'),
  ic_trend_filter_mean_overlap_days: spec('滤波后 IC Mean 重叠', 'IC 分析', 'int', 'none', '相邻 10 日窗口重叠 5 个有效 IC 观测'),
  ic_trend_filter_mean_full_window_count: spec('滤波后 IC Mean 窗口数', 'IC 分析', 'int'),
  ic_count: spec('IC 天数', '运行信息', 'int'),
  pair_count: spec('样本对数', '运行信息', 'int'),
  start_day: spec('起始日', '运行信息', 'text'),
  end_day: spec('结束日', '运行信息', 'text'),
  horizon: spec('持有期', '运行信息', 'int'),
  n_quantiles: spec('分位组数', '运行信息', 'int'),

  nw_ic_mean: spec('IC 均值 (NW)', '显著性检验', 'number', 'abs'),
  nw_ic_lag: spec('NW 滞后阶数', '显著性检验', 'int'),
  nw_ic_long_run_variance: spec('长期方差', '显著性检验', 'number'),
  nw_ic_standard_error: spec('标准误', '显著性检验', 'number', 'lower'),
  nw_ic_t_stat: spec('t 统计量', '显著性检验', 'number', 'abs'),
  nw_ic_p_value: spec('p 值 (Newey-West)', '显著性检验', 'pvalue', 'lower', '对 IC 均值 ≠ 0 的 HAC 显著性检验'),
  nw_ic_significant_5pct: spec('5% 显著', '显著性检验', 'bool', 'higher'),
  nw_ic_autocorr_bound: spec('自相关显著性带', '显著性检验', 'number'),
  nw_ic_lag_min: spec('NW 最小滞后阶数', '显著性检验', 'int'),

  g1_final_cumulative: spec('最低组累计收益', '分位组收益', 'percent', 'none'),
  gn_final_cumulative: spec('最高组累计收益', '分位组收益', 'percent', 'higher'),
  gn_mean_daily_buy_turnover: spec('最高组日均买入换手', '分位组收益', 'percent', 'lower'),
  gn_mean_daily_sell_turnover: spec('最高组日均卖出换手', '分位组收益', 'percent', 'lower'),
  gn_mean_daily_one_way_turnover: spec('最高组日均单边换手', '分位组收益', 'percent', 'lower'),
  gn_mean_daily_transaction_cost: spec('最高组日均交易成本', '分位组收益', 'percent', 'lower'),
  gn_mean_daily_net_return: spec('最高组日均净收益', '分位组收益', 'percent', 'higher'),

  top_group_mean_return: spec('最高组日均收益', '头部组合', 'percent', 'higher', '最高分位组 GN 的日收益均值'),
  top_group_final_cumulative: spec('最高组累计收益', '头部组合', 'percent', 'higher', '最高分位组 GN 的全样本复利累计收益'),
  top_group_annualized_return: spec('最高组年化收益', '头部组合', 'percent', 'higher', '以最高组全样本复利收益按每年 252 个交易日进行几何年化'),
  other_groups_mean_return: spec('其余组日均收益', '头部组合', 'percent'),
  top_group_outperformance_ratio: spec('最高组跑赢占比', '头部组合', 'percent', 'higher', '单日最高组收益不低于每个非头部分组收益的占比'),
  best_lower_group_mean_return: spec('最佳其余组日均收益', '头部组合', 'percent', 'none', 'G1 到 G(N-1) 中日均收益最高的一组'),
  top_group_above_every_lower_group: spec('最高组全面占优', '头部组合', 'bool', 'higher', '最高组是否优于每一个更低分组'),
  top_group_60d_window_count: spec('60日窗口数', '头部组合', 'int', 'none', '完整非重叠 60 个交易日窗口数量，尾部不足 60 日不纳入'),
  top_group_60d_best_window_count: spec('60日最高组第一窗口数', '头部组合', 'int', 'higher', '最高分位组 GN 在非重叠 60 日窗口中累计收益排名第一的窗口数'),
  top_group_60d_best_window_ratio: spec('60日最高组第一占比', '头部组合', 'percent', 'higher', '非重叠 60 日窗口中，最高组累计收益不低于所有更低分组的比例'),
  top_group_60d_mean_cumulative: spec('60日最高组平均累计收益', '头部组合', 'percent', 'higher', '最高分位组 GN 在所有完整 60 日窗口中的平均累计收益'),
  top_group_60d_mean_excess_vs_best_group: spec('60日最高组相对最佳组平均超额', '头部组合', 'percent', 'higher', '最高分位组 GN 相对每个 60 日窗口内最佳其余组的平均累计收益差'),

  future_perturbation_checkpoint_count: spec('检查点数', '数据质量', 'int'),
  future_perturbation_total_comparisons: spec('比较次数', '数据质量', 'int'),
  future_perturbation_total_changed: spec('变化单元数', '数据质量', 'int', 'lower'),
  future_perturbation_checkpoints: spec('扰动检查点数', '数据质量', 'int'),
  future_perturbation_compared_values: spec('扰动比较单元数', '数据质量', 'int'),
  future_perturbation_changed_values: spec('扰动变化单元数', '数据质量', 'int', 'lower'),
  future_perturbation_changed_checkpoints: spec('扰动失败检查点数', '数据质量', 'int', 'lower'),
  future_perturbation_max_abs_difference: spec('扰动最大绝对差', '数据质量', 'number', 'lower'),
  future_perturbation_seed: spec('扰动随机种子', '运行信息', 'int'),
  prefix_truncation_passed: spec('前缀截断一致性', '数据质量', 'bool', 'higher', '只用截至检查点的数据重算后，当日因子不变则通过'),
  prefix_truncation_checkpoints: spec('前缀检查点数', '数据质量', 'int'),
  prefix_truncation_compared_values: spec('前缀比较单元数', '数据质量', 'int'),
  prefix_truncation_changed_values: spec('前缀变化单元数', '数据质量', 'int', 'lower'),
  prefix_truncation_changed_checkpoints: spec('前缀失败检查点数', '数据质量', 'int', 'lower'),
  prefix_truncation_max_abs_difference: spec('前缀最大绝对差', '数据质量', 'number', 'lower'),
  market_cap_neutralized_days: spec('中性化天数', '数据质量', 'int'),
  market_cap_neutralization_mean_r2: spec('市值回归平均 R²', '数据质量', 'number', 'none', 'R² 越高说明因子与市值相关性越强'),
  industry_neutralized_days: spec('行业中性化天数', '数据质量', 'int'),
  industry_neutralization_mean_r2: spec('行业平均 R²', '数据质量', 'number', 'none', 'R² 越高说明行业固定效应解释的因子横截面方差越多'),
  industry_neutralization_covered_share: spec('行业中性化覆盖率', '数据质量', 'percent', 'higher', '进入行业内去均值的有效因子观测占全部有效因子观测的比例'),
  industry_neutralization_unclassified_obs: spec('行业未分类观测', '数据质量', 'int', 'lower'),
  industry_neutralization_small_industry_obs: spec('行业小组剔除观测', '数据质量', 'int', 'lower'),
  industry_market_cap_neutralized_days: spec('联合中性化天数', '数据质量', 'int'),
  industry_market_cap_neutralization_mean_r2: spec('行业市值联合平均 R²', '数据质量', 'number', 'none', 'R² 越高说明因子横截面方差越多可由行业和市值联合解释'),
  industry_market_cap_neutralization_covered_share: spec('联合中性化覆盖率', '数据质量', 'percent', 'higher', '进入行业市值联合回归的有效因子观测占全部有效因子观测的比例'),
  industry_market_cap_neutralization_unclassified_obs: spec('联合回归未分类观测', '数据质量', 'int', 'lower'),
  industry_market_cap_neutralization_invalid_cap_obs: spec('联合回归无效市值观测', '数据质量', 'int', 'lower'),
  industry_market_cap_neutralization_small_industry_obs: spec('联合回归小行业剔除观测', '数据质量', 'int', 'lower'),
  tradability_masked_obs: spec('剔除样本数', '数据质量', 'int'),
  tradability_masked_share: spec('剔除样本占比', '数据质量', 'percent', 'lower'),
  tradability_masked_days: spec('发生剔除天数', '数据质量', 'int'),
  tradability_masked_limit_up_obs: spec('开盘涨停剔除', '数据质量', 'int'),
  tradability_masked_limit_down_obs: spec('开盘跌停剔除', '数据质量', 'int'),
  tradability_masked_st_obs: spec('ST 剔除', '数据质量', 'int'),
}

/** 运行参数键：已经展示在页头，不放进指标卡。 */
export const PARAM_KEYS = new Set([
  'run_id',
  'evaluated_at',
  'factor_name',
  'artifact_name',
  'expression',
  'horizon',
  'return_definition',
  'n_quantiles',
  'evaluation_methods',
  'evaluation_details',
  'evaluation_artifacts',
  'signal_start',
  'signal_end',
  'sample_start_day',
  'sample_end_day',
  'model_training_method',
  'model_training_result',
  'model_training_artifact',
  // Transaction-cost rates are evaluation parameters, not realized metrics.
  'quantile_net_commission_rate',
  'quantile_net_handling_fee_rate',
  'quantile_net_regulatory_fee_rate',
  'quantile_net_transfer_fee_rate',
  'quantile_net_stamp_tax_rate',
  'quantile_net_buy_cost_rate',
  'quantile_net_sell_cost_rate',
  'quantile_net_round_trip_cost_rate',
])

/** 详情页顶部重点指标（存在才展示，按此顺序）。 */
export const KEY_METRICS = [
  'ic_mean',
  'ir',
  'ic_positive_ratio',
  'ic_trend_filter_final',
  'nw_ic_p_value',
  'gn_final_cumulative',
]

const ROLLING_PATTERN = /^(gn)_rolling_(sharpe|drawdown)_(\d+)_(pos_share|min|median|worst)$/

const ROLLING_SERIES: Record<string, string> = { gn: '最高组' }
const ROLLING_KIND: Record<string, string> = { sharpe: '分段夏普', drawdown: '分段回撤' }
const ROLLING_STAT: Record<string, string> = {
  pos_share: '为正占比',
  min: '最小值',
  median: '中位数',
  worst: '最差值',
}

/**
 * 同一方法在流水线中执行多次时，后续产出的键带 `__2`/`__3` 版本后缀
 * （如市值中性化后再算 IC → ic_mean__2）。拆出基础键与执行轮次。
 */
export function splitVersion(key: string): { base: string; version: number } {
  const match = key.match(/^(.*?)__(\d+)$/)
  return match ? { base: match[1], version: Number(match[2]) } : { base: key, version: 1 }
}

export function metricSpec(key: string): MetricSpec {
  const { base, version } = splitVersion(key)
  if (version > 1) {
    const inner = metricSpec(base)
    return { ...inner, label: `${inner.label}（第${version}次）` }
  }
  const known = METRIC_SPECS[key]
  if (known) return known
  const rolling = key.match(ROLLING_PATTERN)
  if (rolling) {
    const [, series, kind, window, stat] = rolling
    // 回撤幅度值展示为正的百分比；夏普占比是百分比；其余为普通数字。
    const format: MetricFormat =
      stat === 'pos_share' ? 'percent' : kind === 'drawdown' ? 'drawdown_pct' : 'number'
    // direction 作用于底层原始值：回撤原始值越大（越接近 0）幅度越小、越好。
    const direction: MetricDirection =
      kind === 'drawdown' ? (stat === 'worst' ? 'higher' : 'none') : stat === 'min' || stat === 'median' || stat === 'pos_share' ? 'higher' : 'none'
    return spec(
      `${ROLLING_SERIES[series]}${ROLLING_KIND[kind]}(${window}日)${ROLLING_STAT[stat]}`,
      '滚动风险',
      format,
      direction,
    )
  }
  return spec(key, '运行信息', 'number')
}

export function formatMetric(key: string, value: unknown): string {
  if (value === null || value === undefined) return '暂无'
  const { format } = metricSpec(key)
  if (typeof value === 'boolean') return value ? '是' : '否'
  if (typeof value !== 'number') return String(value)
  switch (format) {
    case 'percent':
      return `${(value * 100).toFixed(2)}%`
    case 'drawdown_pct':
      // 底层是 ≤0 的分数，展示为正的回撤幅度（-0.2310 → 23.10%）。
      return `${(Math.abs(value) * 100).toFixed(2)}%`
    case 'pvalue':
      return value < 0.0001 ? value.toExponential(2) : value.toFixed(4)
    case 'int':
      return Math.round(value).toLocaleString()
    case 'text':
      return String(value)
    default:
      return Math.abs(value) >= 1000 ? value.toLocaleString(undefined, { maximumFractionDigits: 2 }) : value.toFixed(4)
  }
}

/** 对比页：返回该指标在一组数值中应高亮的值（无可比方向时返回 null）。 */
export function bestValue(key: string, values: number[]): number | null {
  if (values.length < 2) return null
  const { direction } = metricSpec(key)
  switch (direction) {
    case 'higher':
      return Math.max(...values)
    case 'lower':
      return Math.min(...values)
    case 'abs': {
      let best = values[0]
      for (const value of values) if (Math.abs(value) > Math.abs(best)) best = value
      return best
    }
    default:
      return null
  }
}

export const DETAIL_TITLES: Record<string, string> = {
  cumulative_returns: '分位组累计收益',
  group_returns: '分位组日收益',
  ic: '全样本日度 IC 变化与累计 IC',
  ic_horizon_decay: 'IC 持有期衰减（H=20–252，共同日期样本）',
  ic_peak_decay: '60 日滚动 Mean IC（50 日重叠）',
  ic_trend_filter: 'IC 趋势滤波对比（卡尔曼 / 二阶低通 / 傅里叶）',
  ic_trend_filter_mean_10: '滤波后 IC Mean（10 日窗口 / 5 日重叠）',
  quantile_transaction_cost: '分位组交易成本',
  quantile_turnover: '分位组换手率',
  newey_west_ic_autocovariances: 'IC 自协方差 (Newey-West)',
  top_quantile_performance: '最高组表现',
  top_quantile_leadership_60: '最高组 60 日窗口排名',
  top_quantile_missed_windows_60: '最高组 60 日未跑赢窗口',
  tradability_filter: '可交易性过滤明细',
  future_data_perturbation: '未来数据扰动检验',
  prefix_truncation_consistency: '前缀截断一致性检验',
  market_cap_neutralization: '市值中性化统计',
  industry_neutralization: '行业中性化统计',
  industry_market_cap_neutralization: '行业＋市值联合中性化统计',
}

const DETAIL_COLUMN_TITLES: Record<string, string> = {
  mean_rank_ic: '共同日期样本平均 Rank IC',
  mean_ic: '60日滚动 Mean IC（50日重叠）',
  daily_ic: '日度 IC',
  filtered_ic: '卡尔曼滤波 IC 趋势',
  second_order_low_pass_ic: '二阶低通 IC 趋势',
  fourier_low_pass_ic: '傅里叶低频重构 IC',
  kalman_filtered_ic_mean: '卡尔曼 10日 IC Mean',
  second_order_low_pass_ic_mean: '二阶低通 10日 IC Mean',
  fourier_low_pass_ic_mean: '傅里叶 10日 IC Mean',
  top_group_return: '最高组日收益',
  other_groups_return: '其余组平均日收益',
  best_lower_group_return: '最佳其余组日收益',
  top_group_minus_other: '最高组相对其余组均值',
  top_group_minus_best_lower: '最高组相对最佳其余组',
  top_group_is_daily_best: '最高组当日第一',
  top_group_cumulative_return: '最高组累计收益',
  window_number: '窗口编号',
  window_start_ordinal: '窗口开始日期序号',
  window_end_ordinal: '窗口结束日期序号',
  top_group_cumulative: '最高组窗口累计收益',
  best_group_number: '最佳其余组编号',
  best_group_cumulative: '最佳其余组窗口累计收益',
  top_group_excess_vs_best_group: '最高组相对最佳其余组超额',
  top_group_is_best: '最高组窗口第一',
}

/** 详情图表展示顺序：旗舰图在前。 */
const DETAIL_ORDER = [
  'cumulative_returns',
  'ic',
  'ic_horizon_decay',
  'ic_peak_decay',
  'ic_trend_filter',
  'ic_trend_filter_mean_10',
  'group_returns',
  'quantile_turnover',
  'quantile_transaction_cost',
  'top_quantile_performance',
  'top_quantile_leadership_60',
  'top_quantile_missed_windows_60',
  'newey_west_ic_autocovariances',
  'future_data_perturbation',
  'prefix_truncation_consistency',
  'market_cap_neutralization',
  'industry_neutralization',
  'industry_market_cap_neutralization',
  'tradability_filter',
]

export function detailTitle(name: string): string {
  const { base, version } = splitVersion(name)
  const suffix = version > 1 ? `（第${version}次）` : ''
  if (DETAIL_TITLES[base]) return DETAIL_TITLES[base] + suffix
  const rolling = base.match(/^rolling_(sharpe|drawdown)_(\d+)$/)
  if (rolling) return `${rolling[1] === 'sharpe' ? '分段年化夏普' : '分段最大回撤'}（${rolling[2]} 日窗口）${suffix}`
  return name
}

export function detailColumnTitle(column: string): string {
  if (DETAIL_COLUMN_TITLES[column]) return DETAIL_COLUMN_TITLES[column]
  const quantileCumulative = column.match(/^g(\d+)_cumulative$/)
  if (quantileCumulative) return `G${quantileCumulative[1]}窗口累计收益`
  return column
}

export function sortDetails(names: string[]): string[] {
  return [...names].sort((a, b) => {
    const va = splitVersion(a)
    const vb = splitVersion(b)
    const ia = DETAIL_ORDER.indexOf(va.base)
    const ib = DETAIL_ORDER.indexOf(vb.base)
    const orderDiff = (ia === -1 ? 99 : ia) - (ib === -1 ? 99 : ib)
    return orderDiff !== 0 ? orderDiff : va.version - vb.version
  })
}

export const METHOD_INFO = METHOD_DEFINITIONS

export function methodLabel(name: string): string {
  return METHOD_INFO[name]?.label || name
}

export function methodContractLabel(name: string): string {
  if (name === 'quantile_returns') return '分组收益（毛/净）'
  return methodLabel(name)
}

export const STAGE_INFO: Record<string, { label: string; hint: string }> = {
  stage1_validity: { label: '① 有效性', hint: '未来数据扰动检验，淘汰有前视偏差的因子' },
  stage2_ic: { label: '② 原始 IC', hint: 'Newey-West 检验 IC 显著性，p ≥ 阈值淘汰' },
  stage2b_neutral: { label: '③ 市值中性', hint: '市值中性化后再次检验 IC 显著性' },
  stage3_portfolio: { label: '④ 组合表现', hint: '分组收益、头部表现、滚动风险，人工评审' },
}

export function stageLabel(stage?: string | null): string {
  if (!stage) return '自由组合'
  return STAGE_INFO[stage]?.label || stage
}

/** "2026-07-05T21:13:45.123456+08:00" → "07-05 21:13" */
export function shortTime(iso?: string | null): string {
  if (!iso) return '暂无'
  const match = iso.match(/^\d{4}-(\d{2}-\d{2})T(\d{2}:\d{2})/)
  return match ? `${match[1]} ${match[2]}` : iso
}

export function durationText(start?: string | null, end?: string | null): string {
  if (!start || !end) return '暂无'
  const seconds = Math.max(0, (new Date(end).getTime() - new Date(start).getTime()) / 1000)
  if (seconds < 60) return `${seconds.toFixed(0)} 秒`
  const minutes = Math.floor(seconds / 60)
  return `${minutes} 分 ${(seconds - minutes * 60).toFixed(0)} 秒`
}
