import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { AutoComplete, Button, Form, Input, Modal, Typography, message } from 'antd'
import { useEffect, useMemo } from 'react'
import { api, Factor } from '../api/client'

export function ProjectEditorModal({
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
  const [form] = Form.useForm<{ project: string }>()
  const queryClient = useQueryClient()
  const catalog = useQuery({
    queryKey: [library === 'test' ? 'test-factors' : 'factors'],
    queryFn: library === 'test' ? api.testFactors : api.factors,
  })
  const options = useMemo(
    () => [...new Set((catalog.data || []).map((item) => item.project))]
      .sort((left, right) => left.localeCompare(right, 'zh-CN'))
      .map((project) => ({ value: project })),
    [catalog.data],
  )
  const update = useMutation({
    mutationFn: (project: string) => api.updateFactorProject(factor!.batch_id, factor!.factor_name, project),
    onSuccess: (updated) => {
      message.success('项目归属已更新')
      queryClient.setQueryData(['factor', updated.batch_id, updated.factor_name], updated)
      queryClient.invalidateQueries({ queryKey: ['test-factors'] })
      queryClient.invalidateQueries({ queryKey: ['factors'] })
      onClose()
    },
    onError: (error) => message.error(error.message),
  })

  useEffect(() => {
    if (open && factor) form.setFieldsValue({ project: factor.project })
  }, [factor, form, open])

  const save = async () => {
    const values = await form.validateFields()
    update.mutate(values.project)
  }

  return (
    <Modal
      destroyOnHidden
      open={open}
      title={factor ? `修改项目归属：${factor.factor_name}` : '修改项目归属'}
      onCancel={onClose}
      footer={[
        <Button key="cancel" onClick={onClose}>取消</Button>,
        <Button key="save" type="primary" loading={update.isPending} onClick={save}>保存项目</Button>,
      ]}
    >
      <Typography.Paragraph type="secondary" className="tag-editor-hint">
        项目用于分类、展开和筛选，不会修改因子表达式、研究标签或既有评价结果。
      </Typography.Paragraph>
      <Form form={form} layout="vertical">
        <Form.Item
          label="项目"
          name="project"
          rules={[{ required: true, whitespace: true, message: '请输入项目名称' }]}
        >
          <AutoComplete
            options={options}
            filterOption={(input, option) => String(option?.value || '').toLocaleLowerCase().includes(input.toLocaleLowerCase())}
          >
            <Input autoFocus placeholder="例如：Alpha101、华夏191" />
          </AutoComplete>
        </Form.Item>
      </Form>
    </Modal>
  )
}
