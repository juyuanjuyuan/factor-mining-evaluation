import { useEffect, useLayoutEffect, useRef } from 'react'
import { useLocation, useNavigationType } from 'react-router-dom'

// 浏览器自带的恢复会和 SPA 路由抢滚动位置，统一交给这里管理
if ('scrollRestoration' in window.history) {
  window.history.scrollRestoration = 'manual'
}

const positions = new Map<string, number>()
// scroll 事件是异步派发的：路由切换瞬间的 scrollTo 会在切换后才触发事件。
// 用模块级的“当前路由 key”而不是闭包里的旧 key 记录位置，避免把上一页
// 已保存的位置覆盖成 0。
let activeKey = 'default'

// 懒加载图表会逐步撑高页面，一次 scrollTo 常常够不到目标位置：
// 分几次重试，直到落点稳定或超时为止
function restoreScroll(target: number) {
  let attempts = 0
  const tryScroll = () => {
    window.scrollTo(0, target)
    attempts += 1
    if (attempts < 8 && Math.abs(window.scrollY - target) > 2) {
      window.setTimeout(tryScroll, 140)
    }
  }
  tryScroll()
}

/**
 * 点击链接前进时回到页面顶部；浏览器后退/前进时恢复离开时的滚动位置；
 * 筛选条件写入 URL（replace 导航）时保持原位不动。配合列表页的 URL
 * 筛选状态，实现「从详情出来回到原来的位置」。
 */
export function ScrollManager() {
  const location = useLocation()
  const navigationType = useNavigationType()
  const previousPath = useRef(location.pathname)

  useLayoutEffect(() => {
    activeKey = location.key
    const pathChanged = previousPath.current !== location.pathname
    previousPath.current = location.pathname
    if (navigationType === 'POP') {
      const saved = positions.get(location.key)
      if (saved !== undefined) {
        restoreScroll(saved)
        return
      }
      window.scrollTo(0, 0)
      return
    }
    // REPLACE 且路径不变 = 页内筛选变化，保持滚动位置
    if (navigationType === 'REPLACE' && !pathChanged) return
    window.scrollTo(0, 0)
  }, [location.key, location.pathname, navigationType])

  useEffect(() => {
    const save = () => positions.set(activeKey, window.scrollY)
    window.addEventListener('scroll', save, { passive: true })
    // 低频轮询兜底：部分嵌入式 WebView 不派发 scroll 事件（与 LazyRender 同因），
    // 只靠事件驱动会导致位置永远记不下来
    const timer = window.setInterval(save, 400)
    return () => {
      window.removeEventListener('scroll', save)
      window.clearInterval(timer)
    }
  }, [])

  return null
}
