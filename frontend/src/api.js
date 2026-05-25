export async function detectImage(file) {
  const formData = new FormData()
  formData.append('file', file)

  const response = await fetch('/api/detect', {
    method: 'POST',
    body: formData,
  })

  if (!response.ok) {
    const message = await readErrorMessage(response)
    throw new Error(message || '识别请求失败')
  }

  return response.json()
}

export async function compareBoxes(originalBox, editedBox, detectionId = null, boxIndex = null) {
  const payload = {
    original_box: originalBox,
    edited_box: editedBox,
  }
  if (detectionId != null) payload.detection_id = detectionId
  if (boxIndex != null) payload.box_index = boxIndex

  const response = await fetch('/api/compare', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  })

  if (!response.ok) {
    const message = await readErrorMessage(response)
    throw new Error(message || '差异计算失败')
  }

  return response.json()
}

export async function fetchRecords() {
  const response = await fetch('/api/records')

  if (!response.ok) {
    const message = await readErrorMessage(response)
    throw new Error(message || '历史记录获取失败')
  }

  return response.json()
}

export async function fetchRecord(id) {
  const response = await fetch(`/api/records/${id}`)

  if (!response.ok) {
    const message = await readErrorMessage(response)
    throw new Error(message || '记录详情获取失败')
  }

  return response.json()
}

async function readErrorMessage(response) {
  try {
    const data = await response.json()
    return data.detail || data.message || ''
  } catch {
    return ''
  }
}
