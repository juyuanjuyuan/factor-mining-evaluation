import { Alert, Button } from 'antd'
import { Component, type ErrorInfo, type ReactNode } from 'react'

export class ErrorBoundary extends Component<
  { children: ReactNode },
  { error: Error | null }
> {
  state = { error: null as Error | null }

  static getDerivedStateFromError(error: Error) {
    return { error }
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error(error, info)
  }

  render() {
    if (this.state.error) {
      return (
        <div style={{ padding: 48 }}>
          <Alert
            type="error"
            showIcon
            message="页面渲染失败"
            description={this.state.error.message}
            action={<Button onClick={() => window.location.reload()}>重新加载</Button>}
          />
        </div>
      )
    }
    return this.props.children
  }
}
