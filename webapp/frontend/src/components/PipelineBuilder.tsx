import {
  DeleteOutlined,
  HolderOutlined,
  PlusOutlined,
  WarningOutlined,
} from '@ant-design/icons'
import { DndContext, DragEndEvent, PointerSensor, closestCenter, useSensor, useSensors } from '@dnd-kit/core'
import {
  SortableContext,
  arrayMove,
  useSortable,
  verticalListSortingStrategy,
} from '@dnd-kit/sortable'
import { CSS } from '@dnd-kit/utilities'
import { useQuery } from '@tanstack/react-query'
import { Button, Empty, Flex, Tag, Tooltip, Typography } from 'antd'
import { useEffect, useMemo, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { api, Method } from '../api/client'
import { METHOD_INFO, methodContractLabel, methodLabel } from '../lib/metrics'

/**
 * 拖拽式流水线编排：
 * - 左侧方法面板点击「+」加入流水线（自动在其前面补齐缺失的依赖）
 * - 右侧流水线内可拖拽排序、删除单步
 * - 同一方法可以出现多次（如 IC → 市值中性化 → 再算 IC）
 * - 步骤依赖未被前面步骤满足时显示警告，可一键补全
 */

type Step = { id: number; name: string }

let nextStepId = 1
const makeStep = (name: string): Step => ({ id: nextStepId++, name })

/** 递归展开 name 的缺失依赖（按依赖先行的顺序返回，不含 name 本身）。 */
function expandMissingDeps(
  name: string,
  satisfied: Set<string>,
  catalog: Map<string, Method>,
): string[] {
  const result: string[] = []
  const visit = (current: string) => {
    for (const requirement of catalog.get(current)?.requires || []) {
      if (satisfied.has(requirement) || result.includes(requirement)) continue
      visit(requirement)
      result.push(requirement)
    }
  }
  visit(name)
  return result
}

function markProvided(name: string, satisfied: Set<string>, catalog: Map<string, Method>) {
  const method = catalog.get(name)
  const provided = method?.provides?.length ? method.provides : [name]
  for (const capability of provided) satisfied.add(capability)
}

function SortableStep({
  step,
  index,
  occurrence,
  missing,
  onRemove,
  onFixDeps,
}: {
  step: Step
  index: number
  occurrence: number
  missing: string[]
  onRemove: () => void
  onFixDeps: () => void
}) {
  const { attributes, listeners, setNodeRef, transform, transition, isDragging } = useSortable({
    id: step.id,
  })
  return (
    <div
      ref={setNodeRef}
      className={`pipeline-step${isDragging ? ' dragging' : ''}${missing.length ? ' invalid' : ''}`}
      style={{ transform: CSS.Transform.toString(transform), transition }}
    >
      <span className="pipeline-step-handle" {...attributes} {...listeners}>
        <HolderOutlined />
      </span>
      <span className="pipeline-step-index">{index + 1}</span>
      <span className="pipeline-step-body">
        <span>
          <Typography.Text strong>{methodLabel(step.name)}</Typography.Text>
          {occurrence > 1 && (
            <Tag color="orange" style={{ marginLeft: 6 }}>
              第 {occurrence} 次
            </Tag>
          )}
        </span>
        <Typography.Text type="secondary" className="pipeline-step-code">
          {step.name}
        </Typography.Text>
      </span>
      {missing.length > 0 && (
        <Tooltip title={`需要先执行 ${missing.map(methodContractLabel).join('、')}`}>
          <Button
            size="small"
            type="text"
            icon={<WarningOutlined style={{ color: '#f59e0b' }} />}
            onClick={onFixDeps}
          >
            补全依赖
          </Button>
        </Tooltip>
      )}
      <Button size="small" type="text" icon={<DeleteOutlined />} onClick={onRemove} />
    </div>
  )
}

export function PipelineBuilder({
  value,
  onChange,
  onValidityChange,
}: {
  value: string[]
  onChange: (value: string[]) => void
  onValidityChange?: (valid: boolean) => void
}) {
  const methods = useQuery({ queryKey: ['methods'], queryFn: api.methods })
  const catalog = useMemo(
    () => new Map((methods.data || []).map((method) => [method.name, method])),
    [methods.data],
  )
  const [steps, setSteps] = useState<Step[]>([])
  // 用 ref 镜像最新步骤，保证快速连续操作（连点添加）不会读到过期状态
  const stepsRef = useRef<Step[]>([])
  const lastEmitted = useRef<string>('')
  const sensors = useSensors(useSensor(PointerSensor, { activationConstraint: { distance: 4 } }))

  // 外部（模板应用）修改 value 时重建步骤；自己发出的更新不重建，保持拖拽 id 稳定
  useEffect(() => {
    const incoming = JSON.stringify(value)
    if (incoming !== lastEmitted.current) {
      const rebuilt = value.map(makeStep)
      stepsRef.current = rebuilt
      setSteps(rebuilt)
      lastEmitted.current = incoming
    }
  }, [value])

  const update = (transform: (previous: Step[]) => Step[]) => {
    const next = transform(stepsRef.current)
    stepsRef.current = next
    setSteps(next)
    const names = next.map((step) => step.name)
    lastEmitted.current = JSON.stringify(names)
    onChange(names)
  }

  const appendWithDeps = (name: string) =>
    update((previous) => {
      const satisfied = new Set<string>()
      for (const step of previous) markProvided(step.name, satisfied, catalog)
      const inserted = expandMissingDeps(name, satisfied, catalog).map(makeStep)
      return [...previous, ...inserted, makeStep(name)]
    })

  const fixDepsAt = (stepId: number) =>
    update((previous) => {
      const index = previous.findIndex((step) => step.id === stepId)
      if (index === -1) return previous
      const satisfied = new Set<string>()
      for (const step of previous.slice(0, index)) markProvided(step.name, satisfied, catalog)
      const inserted = expandMissingDeps(previous[index].name, satisfied, catalog).map(makeStep)
      return [...previous.slice(0, index), ...inserted, ...previous.slice(index)]
    })

  const removeStep = (stepId: number) =>
    update((previous) => previous.filter((step) => step.id !== stepId))

  const onDragEnd = ({ active, over }: DragEndEvent) => {
    if (!over || active.id === over.id) return
    update((previous) =>
      arrayMove(
        previous,
        previous.findIndex((step) => step.id === active.id),
        previous.findIndex((step) => step.id === over.id),
      ),
    )
  }

  // 每步的未满足依赖与重复序号
  const analysis = useMemo(() => {
    const satisfied = new Set<string>()
    const executed: string[] = []
    return steps.map((step) => {
      const requires = catalog.get(step.name)?.requires || []
      const missing = requires.filter((requirement) => !satisfied.has(requirement))
      const occurrence = executed.filter((name) => name === step.name).length + 1
      executed.push(step.name)
      markProvided(step.name, satisfied, catalog)
      return { missing, occurrence }
    })
  }, [steps, catalog])
  const isValid = steps.length > 0 && analysis.every((item) => item.missing.length === 0)

  useEffect(() => {
    onValidityChange?.(isValid)
  }, [isValid, onValidityChange])

  const defaults = (methods.data || []).filter((method) => method.is_default)
  const extended = (methods.data || []).filter((method) => !method.is_default)
  const renderPalette = (title: string, items: Method[]) => (
    <div className="method-group">
      <Typography.Text type="secondary" className="method-group-title">
        {title}
      </Typography.Text>
      {items.map((method) => (
        <div key={method.name} className="palette-item">
          <span className="palette-item-body">
            <span>
              <Typography.Text strong>{methodLabel(method.name)}</Typography.Text>
              <Typography.Text type="secondary" className="pipeline-step-code">
                {' '}
                {method.name}
              </Typography.Text>
            </span>
            <span className="palette-item-desc">
              {METHOD_INFO[method.name]?.description}
              {method.requires.length > 0 && ` · 依赖 ${method.requires.map(methodContractLabel).join('、')}`}
            </span>
          </span>
          <Tooltip title="加入流水线（自动补齐依赖）">
            <Button size="small" icon={<PlusOutlined />} onClick={() => appendWithDeps(method.name)} />
          </Tooltip>
          <Link className="palette-item-link" to={`/methods?method=${encodeURIComponent(method.name)}`}>
            定义
          </Link>
        </div>
      ))}
    </div>
  )

  return (
    <div className="pipeline-builder">
      <div className="pipeline-palette">
        {renderPalette('默认流水线', defaults)}
        {renderPalette('扩展方法', extended)}
      </div>
      <div className="pipeline-lane">
        <Flex justify="space-between" align="center" style={{ marginBottom: 8 }}>
          <Typography.Text type="secondary" className="method-group-title" style={{ margin: 0 }}>
            执行顺序（拖动排序，可重复添加）
          </Typography.Text>
          {steps.length > 0 && (
            <Button size="small" type="text" onClick={() => update(() => [])}>
              清空
            </Button>
          )}
        </Flex>
        {steps.length === 0 ? (
          <Empty
            image={Empty.PRESENTED_IMAGE_SIMPLE}
            description="从左侧点击「+」按执行顺序添加方法"
          />
        ) : (
          <DndContext sensors={sensors} collisionDetection={closestCenter} onDragEnd={onDragEnd}>
            <SortableContext items={steps.map((step) => step.id)} strategy={verticalListSortingStrategy}>
              {steps.map((step, index) => (
                <SortableStep
                  key={step.id}
                  step={step}
                  index={index}
                  occurrence={analysis[index].occurrence}
                  missing={analysis[index].missing}
                  onRemove={() => removeStep(step.id)}
                  onFixDeps={() => fixDepsAt(step.id)}
                />
              ))}
            </SortableContext>
          </DndContext>
        )}
        {analysis.some((item) => item.missing.length > 0) && (
          <Typography.Text type="warning" className="hint-line">
            <WarningOutlined /> 有步骤缺少前置依赖，提交会被拒绝。点击对应步骤上的「补全依赖」。
          </Typography.Text>
        )}
      </div>
    </div>
  )
}
