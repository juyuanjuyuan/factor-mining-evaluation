import ReactECharts from 'echarts-for-react'
import { useEffect, useState } from 'react'
import type { Curve, Detail, FactorCorrelationWindow, MarketCycleBackground } from '../../api/client'

// Saturated, high-contrast categorical colors. These stay legible on the
// warm chart background and keep G1-G10 recognisable across all detail views.
const palette = [
  '#1565C0', // blue
  '#E65100', // orange
  '#008A3E', // saturated emerald green
  '#7B1FA2', // purple
  '#C62828', // red
  '#6D4C41', // brown
  '#00838F', // teal
  '#D81B60', // magenta
  '#9A7800', // ochre
  '#37474F', // slate
]
const groupColors: Record<string, string> = Object.fromEntries(
  palette.map((color, index) => [`G${index + 1}`, color]),
)
const semanticColors: Record<string, string> = {
  'Long-Short': '#111827',
  '累计 IC': '#D81B60',
  '60日滚动 Mean IC（50日重叠）': '#1565C0',
  '日度 IC': '#7a8794',
  '卡尔曼滤波 IC 趋势': '#1565C0',
  '二阶低通 IC 趋势': '#E65100',
  '傅里叶低频重构 IC': '#00838F',
  '卡尔曼 10日 IC Mean': '#1565C0',
  '二阶低通 10日 IC Mean': '#E65100',
  '傅里叶 10日 IC Mean': '#00838F',
}
const axisColor = '#756d61'
const gridColor = '#dfd2bb'
const correlationColor = '#9a4f32'
const thresholdColor = '#c2413b'
const marketCycleColors: Record<MarketCycleBackground['direction'], string> = {
  up: 'rgba(217, 72, 65, 0.15)',
  down: 'rgba(58, 143, 99, 0.15)',
}

type ZoomRange = {
  start: number
  end: number
  startIndex: number
}

const FULL_ZOOM_RANGE: ZoomRange = { start: 0, end: 100, startIndex: 0 }

function clampZoomPercent(value: number): number {
  return Math.min(100, Math.max(0, value))
}

function resolvedZoomStartIndex(chart: any, pointCount: number, fallback: number): number {
  const startValue = chart
    ?.getModel?.()
    ?.getComponent?.('dataZoom', 0)
    ?.getValueRange?.()?.[0]
  if (typeof startValue !== 'number' || !Number.isFinite(startValue)) return fallback
  return Math.min(Math.max(0, pointCount - 1), Math.max(0, Math.ceil(startValue)))
}

function rebaseCumulativeSeries(
  values: Array<number | string | boolean | null>,
  startIndex: number,
): Array<number | string | boolean | null> {
  const baseValue = values.slice(startIndex).find(
    (value): value is number => typeof value === 'number' && Number.isFinite(value),
  )
  if (baseValue === undefined || Math.abs(1 + baseValue) < Number.EPSILON) return values
  return values.map((value) => (
    typeof value === 'number' && Number.isFinite(value)
      ? (1 + value) / (1 + baseValue) - 1
      : value
  ))
}

function seriesColor(name: string, index: number): string {
  return semanticColors[name] || groupColors[name] || palette[index % palette.length]
}

function detailBaseName(name: string): string {
  const match = name.match(/^(.*?)__(\d+)$/)
  return match ? match[1] : name
}

function marketCycleMarkAreas(
  index: string[],
  backgrounds: MarketCycleBackground[],
) {
  if (!index.length) return []
  return backgrounds.flatMap((background) => {
    const startDay = index.find((day) => day >= background.start_day)
    const endDay = [...index].reverse().find(
      (day) => background.end_day === null || day <= background.end_day,
    )
    if (!startDay || !endDay || startDay > endDay) return []
    return [[
      {
        name: background.label,
        xAxis: startDay,
        itemStyle: { color: marketCycleColors[background.direction] },
      },
      { xAxis: endDay },
    ]]
  })
}

export function DetailChart({
  detail,
  cycleBackgrounds = [],
}: {
  detail: Detail
  cycleBackgrounds?: MarketCycleBackground[]
}) {
  const baseName = detailBaseName(detail.name)
  const isIc = baseName === 'ic'
  const isCumulativeReturns = baseName === 'cumulative_returns'
  const isIcPeakDecay = baseName === 'ic_peak_decay'
  const isIcTrendFilter = baseName === 'ic_trend_filter'
  const isIcTrendMean = baseName === 'ic_trend_filter_mean_10'
  const isLag = detail.name.includes('autocovariances')
  const isDrawdown = detail.name.includes('drawdown')
  const rollingWindowMatch = baseName.match(/^rolling_(?:sharpe|drawdown)_(\d+)$/)
  const rollingWindowDays = rollingWindowMatch ? Number(rollingWindowMatch[1]) : null
  const [zoomRange, setZoomRange] = useState<ZoomRange>(FULL_ZOOM_RANGE)
  useEffect(() => {
    setZoomRange(FULL_ZOOM_RANGE)
  }, [detail.name, detail.index.length, detail.index[0], detail.index[detail.index.length - 1]])
  const selectedStartIndex = isCumulativeReturns ? zoomRange.startIndex : 0
  const series: any[] = detail.columns.map((column, columnIndex) => {
    const label = isIcPeakDecay && column === 'mean_ic'
      ? '60日滚动 Mean IC（50日重叠）'
      : isIcTrendFilter && column === 'daily_ic'
        ? '日度 IC'
        : isIcTrendFilter && column === 'filtered_ic'
          ? '卡尔曼滤波 IC 趋势'
          : isIcTrendFilter && column === 'second_order_low_pass_ic'
            ? '二阶低通 IC 趋势'
            : isIcTrendFilter && column === 'fourier_low_pass_ic'
              ? '傅里叶低频重构 IC'
              : isIcTrendMean && column === 'kalman_filtered_ic_mean'
                ? '卡尔曼 10日 IC Mean'
                : isIcTrendMean && column === 'second_order_low_pass_ic_mean'
                  ? '二阶低通 10日 IC Mean'
                  : isIcTrendMean && column === 'fourier_low_pass_ic_mean'
                    ? '傅里叶 10日 IC Mean'
          : column
    const color = seriesColor(label, columnIndex)
    const rawData = detail.data.map((row) => row[columnIndex])
    return {
      name: label,
      type: isLag ? 'bar' : 'line',
      showSymbol: isIc || isIcTrendFilter || isIcTrendMean ? false : undefined,
      connectNulls: false,
      smooth: false,
      itemStyle: { color },
      lineStyle: {
        color,
        width: isIcTrendFilter && column === 'daily_ic'
          ? 1
          : column.toLowerCase().includes('long') && column.toLowerCase().includes('short') ? 3 : 2.6,
        type: (isIcTrendFilter && column === 'second_order_low_pass_ic') ||
          (isIcTrendMean && column === 'second_order_low_pass_ic_mean')
          ? 'dashed'
          : (isIcTrendFilter && column === 'fourier_low_pass_ic') ||
            (isIcTrendMean && column === 'fourier_low_pass_ic_mean')
            ? 'dotted'
            : 'solid',
      },
      areaStyle: isDrawdown ? { color, opacity: 0.15 } : undefined,
      data: isCumulativeReturns
        ? rebaseCumulativeSeries(rawData, selectedStartIndex)
        : rawData,
    }
  })
  if (isIc && detail.columns.length === 1) {
    let total = 0
    series.push({
      name: '累计 IC',
      type: 'line',
      showSymbol: false,
      connectNulls: false,
      smooth: false,
      itemStyle: { color: semanticColors['累计 IC'] },
      lineStyle: { color: semanticColors['累计 IC'], width: 2.4 },
      areaStyle: undefined,
      yAxisIndex: 1,
      data: detail.data.map((row) => {
        const value = row[0]
        if (typeof value === 'number') total += value
        return total
      }),
    })
  }
  const cycleMarkAreas = !isLag ? marketCycleMarkAreas(detail.index, cycleBackgrounds) : []
  if (cycleMarkAreas.length && series.length) {
    series[0].markArea = {
      silent: true,
      label: { show: false },
      itemStyle: { borderWidth: 0 },
      data: cycleMarkAreas,
    }
  }
  const handleDataZoom = (event: any, chart: any) => {
    if (!isCumulativeReturns) return
    const payload = Array.isArray(event?.batch) ? event.batch[0] : event
    if (!payload) return
    const start = typeof payload.start === 'number'
      ? clampZoomPercent(payload.start)
      : zoomRange.start
    const end = typeof payload.end === 'number'
      ? clampZoomPercent(payload.end)
      : zoomRange.end
    const startIndex = resolvedZoomStartIndex(chart, detail.index.length, zoomRange.startIndex)
    setZoomRange((current) => (
      current.start === start && current.end === end && current.startIndex === startIndex
        ? current
        : { start, end, startIndex }
    ))
  }
  return (
    <ReactECharts
      style={{ height: 420 }}
      onEvents={isCumulativeReturns ? { datazoom: handleDataZoom } : undefined}
      option={{
        backgroundColor: 'transparent',
        color: palette,
        tooltip: rollingWindowDays
          ? {
              trigger: 'axis',
              formatter: (params: any[]) => {
                const endDay = String(params[0]?.axisValueLabel ?? '')
                const values = params
                  .map((item) => {
                    const value = typeof item.data === 'number' ? item.data.toFixed(4) : '暂无'
                    return `${item.marker}${item.seriesName}: ${value}`
                  })
                  .join('<br/>')
                return [
                  `<strong>窗口结束日：${endDay}</strong>`,
                  `计算区间：该日及前 ${rollingWindowDays - 1} 个有效交易日（共 ${rollingWindowDays} 日，含结束日）`,
                  values,
                ].join('<br/>')
              },
            }
          : { trigger: 'axis' },
        legend: { type: 'scroll', top: 4, textStyle: { color: '#665f55' } },
        grid: { left: 55, right: 24, top: 48, bottom: 70 },
        xAxis: { type: 'category', data: detail.index, axisLabel: { color: axisColor } },
        yAxis: isIc
          ? [
              { type: 'value', scale: true, axisLabel: { color: axisColor }, splitLine: { lineStyle: { color: gridColor } } },
              { type: 'value', scale: true, axisLabel: { color: axisColor }, splitLine: { show: false } },
            ]
          : { type: 'value', scale: true, axisLabel: { color: axisColor }, splitLine: { lineStyle: { color: gridColor } } },
        dataZoom: [
          { type: 'inside', start: zoomRange.start, end: zoomRange.end },
          { type: 'slider', bottom: 16, start: zoomRange.start, end: zoomRange.end },
        ],
        toolbox: { right: 16, feature: { saveAsImage: {} } },
        series,
      }}
    />
  )
}

export function SummaryBar({ detail }: { detail: Detail }) {
  const entries = Object.entries(detail.summary || {})
  return (
    <ReactECharts
      style={{ height: 340 }}
      option={{
        backgroundColor: 'transparent',
        color: ['#9a4f32'],
        tooltip: { trigger: 'axis' },
        grid: { left: 60, right: 24, top: 28, bottom: 55 },
        xAxis: { type: 'category', data: entries.map(([name]) => name), axisLabel: { color: axisColor, rotate: entries.length > 8 ? 25 : 0 } },
        yAxis: { type: 'value', scale: true, axisLabel: { color: axisColor }, splitLine: { lineStyle: { color: gridColor } } },
        toolbox: { feature: { saveAsImage: {} } },
        series: [{ name: '平均收益', type: 'bar', data: entries.map(([, value]) => value) }],
      }}
    />
  )
}

export function OverlayChart({ curves, title }: { curves: Curve[]; title: string }) {
  return (
    <ReactECharts
      style={{ height: 380 }}
      option={{
        backgroundColor: 'transparent',
        title: { text: title, textStyle: { color: '#2a2925', fontSize: 15, fontWeight: 650 } },
        color: palette,
        tooltip: { trigger: 'axis' },
        legend: { top: 28, textStyle: { color: '#665f55' } },
        grid: { left: 55, right: 24, top: 72, bottom: 65 },
        xAxis: { type: 'time', axisLabel: { color: axisColor } },
        yAxis: { type: 'value', scale: true, axisLabel: { color: axisColor }, splitLine: { lineStyle: { color: gridColor } } },
        dataZoom: [{ type: 'inside' }, { type: 'slider', bottom: 12 }],
        toolbox: { feature: { saveAsImage: {} } },
        series: curves.map((curve, index) => {
          const color = seriesColor(curve.name, index)
          return {
            name: curve.name,
            type: 'line',
            showSymbol: false,
            connectNulls: false,
            itemStyle: { color },
            lineStyle: { color, width: 2 },
            data: curve.data,
          }
        }),
      }}
    />
  )
}

export function CorrelationWindowChart({
  windows,
  threshold,
}: {
  windows: FactorCorrelationWindow[]
  threshold: number
}) {
  const validCorrelations = windows
    .map((window) => window.correlation)
    .filter((value): value is number => typeof value === 'number')
  const maxAbsCorrelation = validCorrelations.length
    ? Math.max(...validCorrelations.map((value) => Math.abs(value)))
    : 0
  const yExtent = Math.min(1, Math.max(threshold + 0.08, maxAbsCorrelation + 0.08, 0.25))
  const thresholdLabel = threshold.toFixed(2)

  return (
    <ReactECharts
      style={{ height: 330 }}
      option={{
        backgroundColor: 'transparent',
        animationDuration: 350,
        aria: {
          enabled: true,
          description: `非重叠窗口的分段相关性折线图，绝对值超过 ${thresholdLabel} 时标记为红色。`,
        },
        tooltip: {
          trigger: 'axis',
          axisPointer: { type: 'line', lineStyle: { color: '#9b9183', type: 'dashed' } },
          formatter: (params: any[]) => {
            const point = params.find((item) => item.seriesName === '分段相关性')
            if (!point) return ''
            const row = point.data
            const correlation = typeof row.value === 'number' ? row.value.toFixed(4) : '样本不足'
            return [
              `<strong>第 ${row.windowNumber} 段</strong>`,
              `${row.startDay} 至 ${row.endDay}`,
              `相关性：${correlation}`,
              `阈值：±${thresholdLabel}`,
            ].join('<br/>')
          },
        },
        grid: { left: 58, right: 24, top: 32, bottom: 78 },
        xAxis: {
          type: 'category',
          boundaryGap: false,
          data: windows.map((window) => window.end_day),
          axisLine: { lineStyle: { color: '#b9ae9e' } },
          axisTick: { show: false },
          axisLabel: { color: axisColor, hideOverlap: true, margin: 12 },
        },
        yAxis: {
          type: 'value',
          min: -yExtent,
          max: yExtent,
          name: '相关性',
          nameTextStyle: { color: axisColor, padding: [0, 0, 4, 0] },
          axisLabel: { color: axisColor, formatter: (value: number) => value.toFixed(2) },
          splitNumber: 5,
          splitLine: { lineStyle: { color: gridColor } },
        },
        dataZoom: [
          { type: 'inside', filterMode: 'none' },
          {
            type: 'slider',
            bottom: 8,
            height: 22,
            borderColor: '#d8ccba',
            backgroundColor: '#f5f1e9',
            fillerColor: 'rgba(154, 79, 50, 0.14)',
            handleStyle: { color: correlationColor, borderColor: correlationColor },
            textStyle: { color: axisColor },
          },
        ],
        series: [
          {
            name: '分段相关性',
            type: 'line',
            connectNulls: false,
            smooth: false,
            showSymbol: true,
            symbol: 'circle',
            symbolSize: (_value: unknown, params: any) => (params.data.exceedsThreshold ? 9 : 5),
            lineStyle: { color: correlationColor, width: 2.2 },
            itemStyle: { color: correlationColor, borderColor: '#fffdf9', borderWidth: 1.5 },
            emphasis: { focus: 'series', scale: 1.35 },
            data: windows.map((window) => ({
              value: window.correlation,
              windowNumber: window.window_number,
              startDay: window.start_day,
              endDay: window.end_day,
              exceedsThreshold: window.exceeds_threshold,
              itemStyle: window.exceeds_threshold
                ? { color: thresholdColor, borderColor: '#fffdf9', borderWidth: 2 }
                : undefined,
            })),
            markLine: {
              silent: true,
              symbol: ['none', 'none'],
              data: [
                {
                  yAxis: threshold,
                  label: { formatter: `+${thresholdLabel} 阈值`, color: thresholdColor, position: 'insideEndTop' },
                  lineStyle: { color: thresholdColor, type: 'dashed', width: 1.4 },
                },
                {
                  yAxis: -threshold,
                  label: { formatter: `-${thresholdLabel} 阈值`, color: thresholdColor, position: 'insideEndBottom' },
                  lineStyle: { color: thresholdColor, type: 'dashed', width: 1.4 },
                },
                {
                  yAxis: 0,
                  label: { show: false },
                  lineStyle: { color: '#aaa093', type: 'solid', width: 1 },
                },
              ],
            },
          },
        ],
      }}
    />
  )
}
