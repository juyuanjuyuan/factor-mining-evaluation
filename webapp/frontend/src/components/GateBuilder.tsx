import { DeleteOutlined, PlusOutlined } from '@ant-design/icons'
import { Button, InputNumber, Segmented, Select, Typography } from 'antd'
import { GateCondition, GateOp, GateSpec } from '../api/client'
import {
  DRAWDOWN_GATE_OPERATORS,
  GATE_OPERATORS,
  describeGateCondition,
  gateMetricFormat,
  gateMetricSelectGroups,
  isDrawdownMagnitude,
  thresholdFromInput,
  thresholdStep,
  thresholdSuffix,
  thresholdToInput,
} from '../lib/gate'

const DEFAULT_CONDITION: GateCondition = { metric: 'ir', op: 'gte', value: 0 }

function ConditionRow({
  condition,
  onChange,
  onRemove,
}: {
  condition: GateCondition
  onChange: (next: GateCondition) => void
  onRemove: () => void
}) {
  const format = gateMetricFormat(condition.metric)
  const isBool = format === 'bool'
  const isDrawdown = isDrawdownMagnitude(condition.metric)
  const isBetween = condition.op === 'between'

  const setThreshold = (raw: number | null) => onChange({ ...condition, value: raw ?? 0 })
  const setUpper = (raw: number | null) => onChange({ ...condition, value2: raw ?? 0 })

  return (
    <div className="gate-condition">
      <Select
        showSearch
        className="gate-condition-metric"
        placeholder="选择指标"
        value={condition.metric || undefined}
        optionFilterProp="label"
        options={gateMetricSelectGroups()}
        onChange={(metric) => {
          // Reset operator/value sensibly when the metric type changes.
          const nextFormat = gateMetricFormat(metric)
          if (nextFormat === 'bool') {
            onChange({ metric, op: 'gte', value: 1 })
          } else if (nextFormat === 'drawdown_pct') {
            // 回撤不超过 20%（内部存为 ≥ -0.20）。
            onChange({ metric, op: 'gte', value: -0.2 })
          } else {
            onChange({ metric, op: condition.op === 'between' ? 'gte' : condition.op, value: condition.value })
          }
        }}
      />

      {isBool ? (
        <Segmented
          className="gate-condition-bool"
          value={condition.op === 'lte' ? 'no' : 'yes'}
          options={[
            { label: '为是', value: 'yes' },
            { label: '为否', value: 'no' },
          ]}
          onChange={(choice) =>
            onChange(
              choice === 'yes'
                ? { metric: condition.metric, op: 'gte', value: 1 }
                : { metric: condition.metric, op: 'lte', value: 0 },
            )
          }
        />
      ) : isDrawdown ? (
        <>
          <Select
            className="gate-condition-op"
            value={condition.op === 'lte' ? 'lte' : 'gte'}
            options={DRAWDOWN_GATE_OPERATORS}
            onChange={(op: GateOp) => onChange({ metric: condition.metric, op, value: condition.value })}
          />
          <InputNumber
            className="gate-condition-value"
            min={0}
            value={thresholdToInput(format, condition.value)}
            step={thresholdStep(format)}
            suffix="%"
            onChange={(input) => setThreshold(thresholdFromInput(format, input))}
          />
        </>
      ) : (
        <>
          <Select
            className="gate-condition-op"
            value={condition.op}
            options={GATE_OPERATORS.map((item) => ({
              value: item.value,
              label: `${item.symbol}  ${item.label}`,
            }))}
            onChange={(op: GateOp) =>
              onChange({
                ...condition,
                op,
                value2: op === 'between' ? condition.value2 ?? condition.value : undefined,
              })
            }
          />
          <InputNumber
            className="gate-condition-value"
            value={thresholdToInput(format, condition.value)}
            step={thresholdStep(format)}
            suffix={thresholdSuffix(format) || undefined}
            onChange={(input) => setThreshold(thresholdFromInput(format, input))}
          />
          {isBetween && (
            <>
              <span className="gate-condition-tilde">~</span>
              <InputNumber
                className="gate-condition-value"
                value={thresholdToInput(format, condition.value2 ?? condition.value)}
                step={thresholdStep(format)}
                suffix={thresholdSuffix(format) || undefined}
                onChange={(input) => setUpper(thresholdFromInput(format, input))}
              />
            </>
          )}
        </>
      )}

      <Button
        type="text"
        className="gate-condition-remove"
        icon={<DeleteOutlined />}
        aria-label="删除条件"
        onClick={onRemove}
      />
    </div>
  )
}

export function GateBuilder({ value, onChange }: { value: GateSpec; onChange: (next: GateSpec) => void }) {
  const { conditions, match } = value

  const update = (index: number, next: GateCondition) =>
    onChange({ ...value, conditions: conditions.map((item, i) => (i === index ? next : item)) })
  const remove = (index: number) =>
    onChange({ ...value, conditions: conditions.filter((_, i) => i !== index) })
  const add = () => onChange({ ...value, conditions: [...conditions, { ...DEFAULT_CONDITION }] })

  return (
    <div className="gate-builder">
      {conditions.length > 1 && (
        <div className="gate-match-row">
          <Typography.Text type="secondary">当同一因子命中时判定为「过关」：</Typography.Text>
          <Segmented
            size="small"
            value={match}
            options={[
              { label: '全部满足', value: 'all' },
              { label: '满足任一', value: 'any' },
            ]}
            onChange={(next) => onChange({ ...value, match: next as 'all' | 'any' })}
          />
        </div>
      )}

      {conditions.length === 0 ? (
        <div className="gate-empty">
          <Typography.Text type="secondary">
            尚未设置门槛。回测完成后可用它一键筛出达标因子，例如「最高组分段回撤(60日)最差值 ≥ -0.40」。
          </Typography.Text>
        </div>
      ) : (
        <div className="gate-condition-list">
          {conditions.map((condition, index) => (
            <div key={index} className="gate-condition-wrap">
              <ConditionRow
                condition={condition}
                onChange={(next) => update(index, next)}
                onRemove={() => remove(index)}
              />
              {condition.metric && (
                <Typography.Text type="secondary" className="gate-condition-preview">
                  达标要求：{describeGateCondition(condition)}
                </Typography.Text>
              )}
            </div>
          ))}
        </div>
      )}

      <Button type="dashed" icon={<PlusOutlined />} onClick={add} className="gate-add-button">
        添加条件
      </Button>
    </div>
  )
}
