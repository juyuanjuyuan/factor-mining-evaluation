export type MethodDefinition = {
  label: string
  category: string
  description: string
  definition: string
  formula: string
  interpretation: string
}

/**
 * 网页评价方法目录的唯一内容源。
 *
 * 新增任何 @evaluation_method 后，必须在这里补齐中文名称、数学公式、方法定义和
 * 解读，并通过页面检查；否则方法目录会显示“定义缺失”警告。
 */
export const METHOD_DEFINITIONS: Record<string, MethodDefinition> = {
  rank_ic: {
    label: 'IC 序列',
    category: '预测能力',
    description: '逐日计算因子与未来收益的横截面 Spearman 秩相关。',
    formula: String.raw`\begin{aligned}
R_{i,t}^{(H)} &= \frac{\operatorname{open}_{i,t+1+H}}{\operatorname{open}_{i,t+1}} - 1 \\
IC_t &= \operatorname{Corr}\!\left(\operatorname{rank}_i(f_{i,t}),\operatorname{rank}_i(R_{i,t}^{(H)})\right)
\end{aligned}`,
    definition:
      '每个交易日在同时具有有限因子值和未来收益的股票上计算 Spearman 相关；至少需要 3 个样本，且因子秩与收益秩均不能为常数。',
    interpretation: 'IC 的符号表示预测方向，绝对值表示当日横截面排序能力。',
  },
  rank_icir: {
    label: 'ICIR',
    category: '预测能力',
    description: '汇总最新 IC 序列的均值、波动和信息比率。',
    formula: String.raw`\begin{aligned}
\overline{IC} &= \frac{1}{T}\sum_{t=1}^{T} IC_t \\
s_{IC} &= \sqrt{\frac{\sum_{t=1}^{T}(IC_t-\overline{IC})^2}{T-1}} \\
ICIR &= \frac{\overline{IC}}{s_{IC}}
\end{aligned}`,
    definition:
      '对最近一次 rank_ic 生成的有效日度 IC 使用样本标准差（ddof=1），同时输出 IC 为正占比和 |ICIR|。',
    interpretation: 'ICIR 衡量预测方向的一致性；必须结合带符号 IC 均值判断，不能只看绝对值。',
  },
  ic_horizon_decay: {
    label: 'IC 持有期衰减',
    category: '预测能力',
    description: '固定同一组信号日期，比较 H=20 至 252 日未来收益对应的平均 Rank IC。',
    formula: String.raw`\begin{aligned}
\mathcal H &= \{20,21,\ldots,252\} \\
R_{i,t}^{(H)} &= \frac{\operatorname{open}_{i,t+1+H}}{\operatorname{open}_{i,t+1}}-1 \\
IC_t^{(H)} &= \operatorname{Corr}\!\left(\operatorname{rank}_i(f_{i,t}),\operatorname{rank}_i(R_{i,t}^{(H)})\right) \\
\mathcal T &= \bigcap_{H\in\mathcal H}\left\{t:IC_t^{(H)}\text{ 有限}\right\} \\
\overline{IC}^{(H)} &= \frac{1}{|\mathcal T|}\sum_{t\in\mathcal T}IC_t^{(H)}
\end{aligned}`,
    definition:
      '直接读取当前工作因子和复权开盘价，不依赖单一期限的 rank_ic。因子在 t 日收盘后形成，并于 t+1 日开盘进入；H 日标签为 open[t+1+H]/open[t+1]−1。先按最大 H=252 限制可用信号日期，再计算 H=20...252 的逐日横截面 Spearman IC；只有在所有 233 个期限上都能形成有限 IC 的日期才进入共同样本 T，因此曲线每一点使用完全相同的日期集合。每个横截面至少需要 3 对有限样本，页面同时报告结构可用日期数、最终共同日期数和被共同样本规则剔除的日期数。',
    interpretation:
      '横轴是未来收益持有期 H，纵轴是共同日期样本上的平均 Rank IC；它衡量预测能力随持有期变化，而不是 IC 在历史日期上的趋势。不同 H 的累计未来收益高度重叠，曲线点并不独立；本模块只输出衰减曲线，不计算或定义半衰期。',
  },
  ic_peak_decay: {
    label: '60 日滚动 Mean IC（50 日重叠）',
    category: '预测能力',
    description: '使用 60 日滚动窗口、每次前移 10 日，观察窗口平均 IC；相邻窗口重叠 50 日。',
    formula: String.raw`\overline{IC}_{W_k}=\frac{1}{60}\sum_{j=0}^{59}IC_{t_{10k+j}},\qquad W_k=\{t_{10k},\ldots,t_{10k+59}\}`,
    definition:
      '读取最近一次 rank_ic 的有效日度 IC，按时间顺序使用 60 个有效观测组成窗口，相邻窗口每次前移 10 个有效观测，因此重叠 50 个观测；最后不足 60 个观测的尾部不纳入。折线横轴使用每个窗口的结束日期。当前只展示 Mean IC，不定义或计算半衰期。',
    interpretation: '用于观察较长阶段内 IC 的方向和水平变化；由于窗口高度重叠，相邻点并非独立观测，不代表不同持有期的 IC 衰减。',
  },
  ic_trend_filter: {
    label: 'IC 趋势滤波',
    category: '预测能力',
    description: '并排比较三种 IC 趋势，并输出 10 日窗口、5 日重叠的滤波后 Mean IC。',
    formula: String.raw`\begin{aligned}
x_t &= x_{t-1} + w_t, & w_t &\sim \mathcal{N}(0,Q) \\
IC_t &= x_t + v_t, & v_t &\sim \mathcal{N}(0,R) \\
K_t &= \frac{P_t^-}{P_t^-+R}, & \hat{x}_t &= \hat{x}_{t-1}+K_t\left(IC_t-\hat{x}_{t-1}\right)
\\[4pt]
a &= \frac{2}{31}, &
Z_t &= 2(1-a)Z_{t-1}-(1-a)^2Z_{t-2} \\
&&&\quad +(a-\tfrac14a^2)M_{t-1}+\tfrac12a^2M_{t-2}-(a-\tfrac34a^2)M_{t-3}
\\[4pt]
\widetilde{M}^{\mathrm{FFT}}
&=\mathcal{F}^{-1}\!\left(\mathcal{F}(M)_k\,\mathbf{1}\!\left\{f_k\leq\frac{1}{31}\right\}\right)
\\[4pt]
\overline{F}^{(m)}_k
&=\frac{1}{10}\sum_{j=1}^{10}F^{(m)}_{5k+j}, &
W_k&=\{t_{5k+1},\ldots,t_{5k+10}\}
\end{aligned}`,
    definition:
      '依赖最近一次 rank_ic；完整输出至少需要 10 个有效日度 IC。卡尔曼曲线固定 Q/R=0.01，并以稳态后验方差初始化。二阶低通严格使用上式，周期参数为 31，前三个状态取对应的前三个有效 IC；两者在日期 t 都只读取此前或当期可用信息。傅里叶曲线对按有效日期排序的完整 IC 序列做实数 FFT，删除频率高于 1/31 的分量后逆变换，因此会使用样本后部信息。在三条滤波曲线上分别计算 10 个有效观测的均值，窗口每次前移 5 个观测：首窗为 t1 至 t10，次窗为 t6 至 t15；尾部不足 10 个观测不输出，横轴为窗口结束日期。',
    interpretation:
      '10 日 Mean IC 图进一步压低短期波动，可用于观察滤波后趋势是否存在阶段性起伏；相邻窗口重叠 5 个观测，因此相邻点并不独立。卡尔曼和二阶低通的窗口均值在窗口结束日可获得；傅里叶结果仍是事后回看型，不能作为实时信号。',
  },
  newey_west_ic_significance: {
    label: 'IC 显著性',
    category: '统计检验',
    description: '使用 Newey-West HAC 标准误检验 IC 均值是否显著异于零。',
    formula: String.raw`\begin{aligned}
\widehat{LRV} &= \hat{\gamma}_0 + 2\sum_{\ell=1}^{L}\left(1-\frac{\ell}{L+1}\right)\hat{\gamma}_{\ell} \\
SE_{\mathrm{HAC}} &= \sqrt{\frac{\widehat{LRV}}{T}} \\
t_{\mathrm{NW}} &= \frac{\overline{IC}}{SE_{\mathrm{HAC}}}
\end{aligned}`,
    definition:
      '从最新 IC 序列估计自协方差；滞后阶数由连续 3 个落入 ±2/√T 显著性带的自相关自动选择，且不低于 horizon−1。',
    interpretation: 'p 值越小，越有证据拒绝“平均 IC 为零”；HAC 修正了 IC 的时间序列相关。',
  },
  quantile_returns: {
    label: '分组毛收益',
    category: '组合回测',
    description: '按每日因子排序构建等权分位数组合，输出未扣交易成本的毛收益。',
    formula: String.raw`\begin{aligned}
g_{i,t} &= \left\lfloor\frac{(\operatorname{rank}_i(f_{i,t})-1)N}{n_t}\right\rfloor + 1 \\
r_{G_g,t} &= \frac{1}{|G_g|}\sum_{i\in G_g}R_{i,t}^{(H)}
\end{aligned}`,
    definition:
      '每个交易日按因子从低到高尽量均匀分成 N 组，各组使用有效股票未来收益的等权均值；至少需要 N 个有效样本。',
    interpretation: 'G1 是最低因子组，GN 是最高因子组；应检查完整组间排序和可做多头部组合表现。',
  },
  quantile_net_returns: {
    label: '分组净收益',
    category: '组合回测',
    description: '按分位组重新生成等权持仓路径，按单边万分之 7 的统一成本假设扣除交易成本。',
    formula: String.raw`\begin{aligned}
w_{i,g,t} &= \frac{\mathbf{1}(i\in G_{g,t})}{|G_{g,t}|} \\
B_{g,t} &= \sum_i \max(w_{i,g,t}-w_{i,g,t-1},0) \\
S_{g,t} &= \sum_i \max(w_{i,g,t-1}-w_{i,g,t},0) \\
c_{\mathrm{buy}} &= c_{\mathrm{sell}} = 0.0007 \\
r^{\mathrm{net}}_{G_g,t} &= r_{G_g,t}-B_{g,t}c_{\mathrm{buy}}-S_{g,t}c_{\mathrm{sell}}
\end{aligned}`,
    definition:
      '每个交易日使用与 quantile_returns 相同的因子排序和分组规则生成 G1 到 GN 等权目标权重；由相邻目标权重差计算买入换手、卖出换手和单边换手，并从当日分位组毛收益中扣除交易成本。买入和卖出均按单边万分之 7（0.0007）扣费；一单位完整买卖的成本为千分之 1.4（0.0014）。输出净收益到标准 group_returns 明细，另输出 quantile_turnover 与 quantile_transaction_cost 诊断明细。',
    interpretation: '净收益可直接交给 quantile_cumulative、quantile_plot 和滚动风险方法；该规则是统一单边成本假设，不按佣金、税费等项目拆分，也不随成交规模或价格变化。',
  },
  quantile_cumulative: {
    label: '分组累计收益',
    category: '组合回测',
    description: '将各分位组日收益按几何方式复利。',
    formula: String.raw`C_{G,t}=\prod_{s\le t}\left(1+r_{G,s}\right)-1`,
    definition:
      '读取最近一次 group_returns 结果，将缺失日收益按 0 处理后逐日复利，并记录最低组和最高组的期末累计收益。group_returns 可以来自 quantile_returns 的分组毛收益，也可以来自 quantile_net_returns 的分组净收益。',
    interpretation: '累计收益反映整段样本表现，但仍需结合日收益稳定性、回撤和换手率。',
  },
  quantile_plot: {
    label: '收益曲线图',
    category: '结果呈现',
    description: '绘制全部分位组的累计收益曲线。',
    formula: String.raw`\operatorname{plot}\!\left(\{C_{G_1,t},\ldots,C_{G_N,t}\}\right)`,
    definition:
      '读取最近一次 quantile_cumulative 明细，绘制 Q1-QN 曲线并保存 PNG；重复执行时使用版本化文件名。',
    interpretation: '用于观察组间排序、头部组表现与阶段失效，不产生新的收益估计。',
  },
  cycle_context: {
    label: '周期背景',
    category: '结果呈现',
    description: '在已有时间序列图中叠加 A 股大盘上升段和下跌段的背景色。',
    formula: String.raw`B(t)=\begin{cases}\mathrm{red},&t\in\mathcal{U}\\\mathrm{green},&t\in\mathcal{D}\\\varnothing,&\text{otherwise}\end{cases}`,
    definition:
      '依赖收益曲线图，以中证全指（000985）25% 反转阈值划分的市场段落作为背景：上升段为红色、下跌段为绿色。它原地重绘已有的分位组累计收益 PNG；运行详情页则把同一背景用于 IC、累计收益和分段风险等折线图。该步骤不读取因子或额外行情数据，也不产生指标、明细或新的 artifact 键。2024-02-05 之后仍是未确认结束的上升段，因此明确标为“暂定”。',
    interpretation:
      '用于把因子的阶段性表现放回市场环境中审阅，例如观察 IC 或分组收益是否只在某一轮上涨或下跌中有效；背景颜色不代表因子优劣，也不参与筛选或回测计算。',
  },
  future_data_perturbation: {
    label: '未来数据检验',
    category: '数据质量',
    description: '扰动检查点之后的数据，检测因子是否错误引用未来信息。',
    formula: String.raw`\max_i\left|f_t(D_{\le t},D_{>t})-f_t(D_{\le t},\widetilde{D}_{>t})\right|=0`,
    definition:
      '选择最多 4 个有效检查点，随机改变严格位于检查点之后的有限行情值并重算因子；检查点当日因子必须在容差内保持不变。',
    interpretation: '任一已比较值改变即判定失败；通过只能说明未在这些扰动中发现前视，不是形式化证明。',
  },
  prefix_truncation_consistency: {
    label: '前缀截断一致性',
    category: '数据质量',
    description: '只使用检查点及以前的数据重算因子，检测全量计算结果是否依赖未来数据。',
    formula: String.raw`\max_i\left|f_{i,t}(D_{\le t},D_{>t})-f_{i,t}(D_{\le t})\right|=0`,
    definition:
      '选择最多 4 个有效检查点；每个检查点 t 只保留所有输入矩阵中不晚于 t 的行，重算表达式，并将重算结果的最后一行与全量计算的 t 行因子在容差内比较。',
    interpretation: '任一已比较值改变即判定失败；通过说明这些检查点上批量计算与在线前缀计算一致，但不能替代数据发布时间审计。',
  },
  market_cap_neutralize: {
    label: '市值中性化',
    category: '因子变换',
    description: '每日对数市值横截面回归并以残差替换当前工作因子。',
    formula: String.raw`\begin{aligned}
f_{i,t} &= a_t+b_t\ln(\operatorname{cap}_{i,t})+\varepsilon_{i,t} \\
f_{i,t} &\leftarrow \varepsilon_{i,t}
\end{aligned}`,
    definition:
      '每天在市值为正且因子有限的股票上进行带截距 OLS，最少 3 个样本；残差保持原始宽表轴并替换 state.factor。',
    interpretation: '后续方法自动使用中性化因子；R² 越高说明原始因子的市值暴露越强。',
  },
  industry_neutralize: {
    label: '行业中性化',
    category: '因子变换',
    description: '每日按当日一级行业去均值，并以行业固定效应残差替换当前工作因子。',
    formula: String.raw`\begin{aligned}
\bar f_{g,t} &= \frac{1}{|\mathcal I_{g,t}|}\sum_{j\in\mathcal I_{g,t}}f_{j,t} \\
f_{i,t} &\leftarrow f_{i,t}-\bar f_{g(i,t),t}
\end{aligned}`,
    definition:
      '每天仅匹配因子暴露日 t 的一级行业标签；在因子有限且行业内至少有 3 只有效股票的样本上，按行业去均值。缺失分类和小行业样本置为 NaN，不以前后日期的行业标签补齐；残差保持原始宽表轴并替换 state.factor。',
    interpretation:
      '后续方法自动使用行业中性化因子；R² 表示当日行业固定效应解释的因子横截面方差比例。它不等于市值中性化，若要同时控制行业和市值，应使用联合回归而非随意串联。',
  },
  industry_market_cap_neutralize: {
    label: '行业＋市值联合中性化',
    category: '因子变换',
    description: '每日以行业固定效应和对数总市值联合 OLS，并以残差替换当前工作因子。',
    formula: String.raw`\begin{aligned}
f_{i,t} &= \alpha_t+\gamma_t\ln(\operatorname{cap}_{i,t})
+\sum_{g=1}^{G_t-1}\beta_{g,t}D_{i,g,t}+\varepsilon_{i,t} \\
f_{i,t} &\leftarrow \widehat{\varepsilon}_{i,t}
\end{aligned}`,
    definition:
      '每天仅使用因子暴露日 t 的同日一级行业标签；在因子有限、总市值为正且行业内至少有 3 只有效股票的样本上，一次拟合带截距的 OLS。使用 G−1 个行业 dummy；缺失分类、无效市值和小行业样本置为 NaN，不以前后日期补齐行业标签。残差保持原始宽表轴并替换 state.factor。',
    interpretation:
      '后续方法自动使用同时控制行业和总市值后的因子；联合 R² 表示两类暴露合计解释的当日横截面方差比例。它不同于先运行两个单独中性化模块：顺序残差会依赖顺序，第二次变换可能重新引入第一次控制的暴露。',
  },
  tradability_filter: {
    label: '可交易性过滤',
    category: '因子变换',
    description: '剔除下一开盘入场日触及涨跌停或处于 ST 状态的样本。',
    formula: String.raw`\begin{aligned}
g_{i,t+1} &= \frac{\operatorname{open}_{i,t+1}}{\operatorname{close}_{i,t}} - 1 \\
\left(|g_{i,t+1}| \ge L_{i,t+1}-0.002\right)\lor \operatorname{ST}_{i,t+1}
&\Longrightarrow f_{i,t}\leftarrow\mathrm{NaN}
\end{aligned}`,
    definition:
      '检查因子日 t 的下一交易日 t+1；用代码板块推导的涨跌幅限制矩阵 L 判断开盘是否触及涨跌停，并将触线或 ST 样本从当前工作因子中屏蔽。',
    interpretation: '覆盖开盘封板后盘中打开的入场不可成交情形；涨跌停价仍是基于复权开盘/收盘和板块限制的代理口径，不检查退出日可交易性。',
  },
  top_quantile_performance: {
    label: '最高组表现分析',
    category: '组合诊断',
    description: '比较最高分位组与其余分组，并统计 60 日非重叠窗口内最高组累计收益排名第一的比例。',
    formula: String.raw`\begin{aligned}
r_{\mathrm{top},t} &= r_{G_N,t} \\
r_{\mathrm{other},t} &= \frac{1}{N-1}\sum_{g=1}^{N-1}r_{G_g,t} \\
\operatorname{win}_t &= \mathbb{1}\!\left[
r_{\mathrm{top},t}\ge \max_{g\le N-1} r_{G_g,t}
\right] \\
\operatorname{WinRate}_{\mathrm{day}} &= \frac{1}{T}\sum_{t=1}^{T}\operatorname{win}_t \\
C_{g,W_k} &= \prod_{t\in W_k}(1+r_{G_g,t})-1 \\
C_{\mathrm{top},W_k} &= \prod_{t\in W_k}(1+r_{\mathrm{top},t})-1 \\
R_{\mathrm{top},\mathrm{ann}} &=
\left(1+C_{\mathrm{top},T}\right)^{252/T}-1 \\
\operatorname{win}_k &= \mathbb{1}\!\left[
C_{\mathrm{top},W_k}\ge \max_{g\le N-1} C_{g,W_k}
\right] \\
\operatorname{WinRate}_{60} &= \frac{1}{K}\sum_{k=1}^{K}\operatorname{win}_k
\end{aligned}`,
    definition:
      '读取最近一次分组日收益明细的 G1 到 GN；如果上游是 quantile_returns，则使用毛收益，如果上游是 quantile_net_returns，则使用扣费后的净收益。头部使用最高分位组 GN；日度跑赢要求 GN 当日收益不低于 G1 到 G(N−1) 的每个更低分组。其余组按 G1 到 G(N−1) 的简单平均作为日均收益对照。最高组年化收益以全样本复利累计收益为基础，按 T 个明细交易日和每年 252 个交易日计算 \( (1+C_{\\mathrm{top},T})^{252/T}-1 \)；缺失的最高组日收益沿用累计收益的既有规则按 0 计入。与此同时，按 60 个交易日切成不重叠窗口 W1={t1...t60}, W2={t61...t120}，尾部不足 60 日不纳入统计。每个窗口分别计算最高组与各组累计收益，判断最高组是否不低于 G1 到 G(N−1) 中的最高累计收益，并单独列出未跑赢窗口。页面展示口径统一为最高组。',
    interpretation: '60 日胜率衡量最高组在中期阶段里是否经常领先更低分组；未跑赢窗口用于定位阶段失效日期和当期领先分组。',
  },
  rolling_sharpe: {
    label: '分段夏普',
    category: '风险稳定性',
    description: '计算分位组在 20/60/252 日非重叠完整窗口内的年化夏普，并输出 60 日窗口折线图。',
    formula: String.raw`\operatorname{Sharpe}_{W_k}
=\frac{\operatorname{mean}(r_{W,t})}{\operatorname{std}(r_{W,t})}\sqrt{252}`,
    definition:
      '将最近一次分组日收益切成 20、60、252 日的完整非重叠窗口 W1={t1...tW}、W2={t(W+1)...t(2W)}。每个窗口使用样本标准差计算年化夏普，尾部不足窗口的数据不纳入；detail 以窗口结束日为索引。以结束日 t 标记的 60 日点使用含 t 在内的 [t-59,...,t] 共 60 个有效交易日，不按日历日回溯。输出最高组分段夏普的正值占比、最小值和中位数；同时把 60 日 detail 的全部 G1-GN 绘制为折线 PNG，便于观察分段稳定性和组间排序。',
    interpretation: '用于比较相互独立的市场阶段中的收益风险比，避免每日滑动窗口对同一批收益反复计数。',
  },
  rolling_drawdown: {
    label: '分段回撤',
    category: '风险稳定性',
    description: '计算分位组在 20/60/252 日非重叠完整窗口内的最大回撤。',
    formula: String.raw`\begin{aligned}
\operatorname{wealth}_k &= \prod_{j\le k}(1+r_j) \\
MDD_{W,t} &= \min_{k\in W}\left[
\frac{\operatorname{wealth}_k}{\max_{u\le k}(1,\operatorname{wealth}_u)}-1
\right]
\end{aligned}`,
    definition:
      '将最近一次分组日收益切成 20、60、252 日的完整非重叠窗口。每个窗口从净值 1 开始复利，计算窗口内最深回撤；尾部不足窗口的数据不纳入，detail 以窗口结束日为索引。汇总最高组的最差值与中位数。',
    interpretation: '越接近 0 表示回撤越浅；非重叠分段的分布更能反映不同阶段的独立风险。',
  },
}
