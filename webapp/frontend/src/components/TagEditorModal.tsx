import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Button, Form, Modal, Select, Typography, message } from 'antd'
import { useEffect } from 'react'
import { api, Factor } from '../api/client'

export function TagEditorModal({
  factor,
  library,
  open,
  onClose,
}: {
  factor: Factor | null
  library: 'test' | 'factor'
  open: boolean
  onClose: () => void
}) {
  const [form] = Form.useForm<{ tags: string[] }>()
  const queryClient = useQueryClient()
  const suggestions = useQuery({
    queryKey: ['factor-tags', library],
    queryFn: () => api.factorTags(library),
  })
  const update = useMutation({
    mutationFn: (tags: string[]) => api.updateFactorTags(factor!.batch_id, factor!.factor_name, tags),
    onSuccess: (updated) => {
      message.success('标签已保存')
      queryClient.setQueryData(['factor', updated.batch_id, updated.factor_name], updated)
      queryClient.invalidateQueries({ queryKey: ['test-factors'] })
      queryClient.invalidateQueries({ queryKey: ['factors'] })
      queryClient.invalidateQueries({ queryKey: ['factor-tags'] })
      onClose()
    },
    onError: (error) => message.error(error.message),
  })

  useEffect(() => {
    if (open && factor) form.setFieldsValue({ tags: factor.tags || [] })
  }, [factor, form, open])

  const save = async () => {
    const values = await form.validateFields()
    update.mutate(values.tags || [])
  }

  return (
    <Modal
      destroyOnHidden
      open={open}
      title={factor ? `设置研究标签：${factor.factor_name}` : '设置研究标签'}
      onCancel={onClose}
      footer={[
        <Button key="cancel" onClick={onClose}>取消</Button>,
        <Button key="save" type="primary" loading={update.isPending} onClick={save}>保存标签</Button>,
      ]}
    >
      <Typography.Paragraph type="secondary" className="tag-editor-hint">
        标签只用于研究分组和批量任务，不会修改因子公式或历史评价结果。
      </Typography.Paragraph>
      <Form form={form} layout="vertical">
        <Form.Item
          label="标签"
          name="tags"
          extra="输入后按 Enter，或用逗号分隔。每个因子最多 20 个标签，每个标签最多 32 个字符。"
        >
          <Select
            mode="tags"
            tokenSeparators={[',', '，']}
            placeholder="例如：回测组合、低波动、候选池"
            options={(suggestions.data || []).map((item) => ({
              value: item.tag,
              label: `${item.tag}（${item.count}）`,
            }))}
          />
        </Form.Item>
      </Form>
    </Modal>
  )
}
