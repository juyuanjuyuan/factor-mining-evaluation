import { BranchesOutlined, ReloadOutlined } from '@ant-design/icons'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { Button, Skeleton, Space, Tag, Typography } from 'antd'
import { useState } from 'react'
import { api } from '../api/client'
import { GeneticMiningPanel } from '../components/GeneticMiningPanel'

export default function GeneticMining() {
  const queryClient = useQueryClient()
  const [createOpen, setCreateOpen] = useState(false)
  const campaigns = useQuery({
    queryKey: ['genetic-campaigns'],
    queryFn: api.geneticCampaigns,
    refetchInterval: (query) => (query.state.data?.some((item) => item.status === 'running') ? 2500 : 10000),
  })
  const running = (campaigns.data || []).filter((campaign) => campaign.status === 'running')
  const finished = (campaigns.data || []).filter((campaign) => campaign.status !== 'running').length

  return (
    <>
      <div className="page-heading page-heading-row">
        <div className="page-heading-copy">
          <Typography.Title level={2}>遗传规划挖掘</Typography.Title>
          <Typography.Text type="secondary">
            用遗传规划在训练区间搜索因子表达式，通过测试筛选的候选自动送入因子库做相关性检验。
            挖掘跑在独立进程，不占用普通评价的任务队列。
          </Typography.Text>
        </div>
        <Space wrap>
          {running.length ? (
            <Tag color="blue">{running.map((campaign) => campaign.campaign).join('、')} 运行中</Tag>
          ) : (
            <Tag>当前无运行任务</Tag>
          )}
          <Button
            icon={<ReloadOutlined />}
            onClick={() => queryClient.invalidateQueries({ queryKey: ['genetic-campaigns'] })}
          >
            刷新
          </Button>
          <Button type="primary" icon={<BranchesOutlined />} onClick={() => setCreateOpen(true)}>
            新建挖掘任务
          </Button>
        </Space>
      </div>

      <div className="mining-overview" aria-label="挖掘任务概览">
        <div className="mining-overview-item">
          <span>运行中</span>
          <strong>{running.length}</strong>
        </div>
        <div className="mining-overview-item">
          <span>已结束</span>
          <strong>{finished}</strong>
        </div>
        <div className="mining-overview-item">
          <span>全部任务</span>
          <strong>{campaigns.data?.length ?? 0}</strong>
        </div>
      </div>

      {campaigns.isLoading ? (
        <Skeleton active paragraph={{ rows: 8 }} />
      ) : (
        <GeneticMiningPanel
          variant="page"
          open={createOpen}
          onOpen={() => setCreateOpen(true)}
          onClose={() => setCreateOpen(false)}
        />
      )}
    </>
  )
}
