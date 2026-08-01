import { Segmented } from 'antd'
import { useLocation, useNavigate } from 'react-router-dom'

type SectionTab = { label: string; path: string }

/**
 * 板块内的二级导航。顶层只保留四个板块，同板块下的相邻页面（如流水线模板与评价模块库）
 * 在这里切换，路由地址保持不变。
 */
export function SectionTabs({ tabs }: { tabs: SectionTab[] }) {
  const navigate = useNavigate()
  const location = useLocation()
  const current = tabs.find((tab) => location.pathname.startsWith(tab.path))?.path ?? tabs[0].path
  return (
    <Segmented
      className="section-subnav"
      value={current}
      onChange={(value) => navigate(String(value))}
      options={tabs.map((tab) => ({ label: tab.label, value: tab.path }))}
    />
  )
}
