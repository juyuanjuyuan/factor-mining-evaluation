import { Tag } from 'antd'

const colors: Record<string, string> = {
  queued: 'default',
  running: 'processing',
  succeeded: 'success',
  failed: 'error',
  cancelled: 'warning',
  skipped: 'default',
  passed: 'success',
  eliminated: 'error',
  not_passed: 'error',
  completed: 'success',
}

const labels: Record<string, string> = {
  queued: '排队中',
  running: '运行中',
  succeeded: '成功',
  failed: '失败',
  cancelled: '已取消',
  skipped: '已跳过',
  passed: '通过',
  eliminated: '淘汰',
  not_passed: '未过关',
  completed: '完成',
}

export function StatusTag({ status }: { status?: string }) {
  if (!status) return <Tag>暂无</Tag>
  return <Tag color={colors[status]}>{labels[status] || status}</Tag>
}
