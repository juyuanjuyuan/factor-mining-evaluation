import { ReactNode, useEffect, useRef, useState } from 'react'

/**
 * 滚动到视口附近才挂载子节点，避免长结果页一次性初始化十几个 ECharts
 * 实例把主线程卡死。占位高度接近真实卡片高度，减少滚动跳动。
 * 用滚动事件 + getBoundingClientRect 判断而非 IntersectionObserver，
 * 保证在合成器不活跃的嵌入式 WebView 里也能触发。
 */
export function LazyRender({ minHeight = 480, children }: { minHeight?: number; children: ReactNode }) {
  const ref = useRef<HTMLDivElement>(null)
  const [visible, setVisible] = useState(false)
  useEffect(() => {
    if (visible) return
    let ticking = false
    const check = () => {
      ticking = false
      const node = ref.current
      if (!node) return
      const rect = node.getBoundingClientRect()
      if (rect.top < window.innerHeight + 600 && rect.bottom > -600) {
        setVisible(true)
        detach()
      }
    }
    const onScroll = () => {
      if (ticking) return
      ticking = true
      // setTimeout 而非 requestAnimationFrame：后者在部分嵌入式 WebView 里
      // 会随合成器一起挂起，导致懒加载永远不触发
      window.setTimeout(check, 120)
    }
    // 低频轮询兜底：部分嵌入式 WebView 会挂起后台标签的 scroll 事件，
    // 只靠事件驱动会导致图表永远不挂载
    const timer = window.setInterval(check, 600)
    const detach = () => {
      window.removeEventListener('scroll', onScroll)
      window.removeEventListener('resize', onScroll)
      window.clearInterval(timer)
    }
    window.addEventListener('scroll', onScroll, { passive: true })
    window.addEventListener('resize', onScroll)
    check()
    return detach
  }, [visible])
  return (
    <div ref={ref} style={visible ? undefined : { minHeight }}>
      {visible ? children : null}
    </div>
  )
}
