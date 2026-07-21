import { ArrowLeftOutlined } from '@ant-design/icons'
import { Button } from 'antd'
import { useNavigate } from 'react-router-dom'

/**
 * 有来路时按浏览器历史返回，保留上一页的筛选、分页和滚动位置；
 * 直接打开详情页（无历史）时退到 fallback 列表。
 */
export function BackButton({ fallback, label = '返回' }: { fallback: string; label?: string }) {
  const navigate = useNavigate()
  const canGoBack = (window.history.state?.idx ?? 0) > 0
  return (
    <Button
      type="text"
      icon={<ArrowLeftOutlined />}
      className="back-button"
      onClick={() => (canGoBack ? navigate(-1) : navigate(fallback, { replace: true }))}
    >
      {label}
    </Button>
  )
}
