import React, { useEffect, useRef, useState } from 'react'
import { detectImage, compareBoxes, createAddedBox, updateAddedBox, saveDeletedBox, fetchRecords } from './api'

const clamp = (v, min, max) => Math.min(Math.max(v, min), max)

export default function App() {
  const stageRef = useRef(null)
  const dragRef = useRef(null)

  const [status, setStatus] = useState('等待上传图片')
  const [image, setImage] = useState(null)
  const [originalBoxes, setOriginalBoxes] = useState([])
  const [editedBoxes, setEditedBoxes] = useState([])
  const [showEditedForModel, setShowEditedForModel] = useState({})
  const [boxTypes, setBoxTypes] = useState([])
  const [deletedBoxes, setDeletedBoxes] = useState([])
  const [activeBoxIndex, setActiveBoxIndex] = useState(0)
  const [boxMenuIndex, setBoxMenuIndex] = useState(null)

  const [confidenceMap, setConfidenceMap] = useState({})
  const [metricsMap, setMetricsMap] = useState({})
  const [detectionIds, setDetectionIds] = useState([])
  const [addedBoxIds, setAddedBoxIds] = useState([])
  const [isDetecting, setIsDetecting] = useState(false)

  const [records, setRecords] = useState([])
  const [historyStatus, setHistoryStatus] = useState('暂无历史记录')

  useEffect(() => {
    loadRecords()
  }, [])

  async function loadRecords(showStatus = true) {
    try {
      const data = await fetchRecords()
      setRecords(data.records || [])
      setHistoryStatus(data.records?.length ? `共 ${data.records.length} 条记录` : '暂无历史记录')
    } catch (err) {
      console.error(err)
      if (showStatus) setHistoryStatus('历史记录加载失败')
    }
  }

  async function handleUpload(e) {
    const file = e.target.files?.[0]
    if (!file) return

    const localUrl = URL.createObjectURL(file)
    setStatus('图片上传成功，正在调用后端识别...')
    setIsDetecting(true)

    try {
      const result = await detectImage(file)
      const detections = result.detections || []
      const modelBoxes = detections.map((det) => ({ ...det.bbox }))

      setImage({
        id: result.image.id,
        url: result.image.url || localUrl,
        width: result.image.width,
        height: result.image.height,
        name: result.image.name,
      })
      setOriginalBoxes(modelBoxes)
      setEditedBoxes(modelBoxes.map((b) => ({ ...b })))
      setShowEditedForModel({})
      setBoxTypes(modelBoxes.map(() => 'model'))
      setDeletedBoxes([])
      setActiveBoxIndex(0)
      setBoxMenuIndex(null)

      setConfidenceMap(Object.fromEntries(detections.map((det, idx) => [idx, det.confidence ?? null])))
      setMetricsMap({})
      setDetectionIds(detections.map((det) => det.id ?? null))
      setAddedBoxIds(detections.map(() => null))

      setStatus(detections.length ? `识别完成，共检测到 ${detections.length} 个模型框` : '未检测到目标区域，图片已保存')
      loadRecords(false)
    } catch (err) {
      console.error(err)
      setStatus(`识别失败：${err.message}`)
      setImage(null)
      setOriginalBoxes([])
      setEditedBoxes([])
      setShowEditedForModel({})
      setBoxTypes([])
      setDeletedBoxes([])
      setActiveBoxIndex(0)
      setBoxMenuIndex(null)
      setConfidenceMap({})
      setMetricsMap({})
      setDetectionIds([])
      setAddedBoxIds([])
    } finally {
      setIsDetecting(false)
    }
  }

  function openRecord(record) {
    const detections = record.detections || (record.detection ? [record.detection] : [])
    const corrections = record.corrections || []
    const addedBoxes = record.added_boxes || []
    const deletedIndexes = record.deleted_box_indexes || []

    const modelOriginal = detections.map((det) => ({ ...det.bbox }))
    const modelEdited = modelOriginal.map((box, idx) => {
      const correction = corrections.find((item) => item.box_index === idx)
      return correction?.bbox ? { ...correction.bbox } : { ...box }
    })

    const mergedEdited = [...modelEdited, ...addedBoxes.map((item) => ({ ...item.bbox }))]

    const showMap = {}
    corrections.forEach((item) => {
      if (typeof item.box_index === 'number') showMap[item.box_index] = true
    })

    setImage({
      id: record.id,
      url: record.url || record.image_url || '',
      width: record.width,
      height: record.height,
      name: record.filename,
    })
    setOriginalBoxes(modelOriginal)
    setEditedBoxes(mergedEdited)
    setShowEditedForModel(showMap)
    setBoxTypes([...modelOriginal.map(() => 'model'), ...addedBoxes.map(() => 'added')])
    setDeletedBoxes(deletedIndexes)
    setActiveBoxIndex(0)
    setBoxMenuIndex(null)

    setConfidenceMap(Object.fromEntries(modelOriginal.map((_, idx) => [idx, detections[idx]?.confidence ?? null])))
    setDetectionIds([...detections.map((det) => det.id ?? null), ...addedBoxes.map(() => null)])
    setAddedBoxIds([...detections.map(() => null), ...addedBoxes.map((item) => item.id ?? null)])
    setMetricsMap(Object.fromEntries(corrections.map((item) => [item.box_index ?? 0, item])))
    setStatus('已打开历史检测记录')
  }

  function getScale() {
    const rect = stageRef.current?.getBoundingClientRect()
    if (!rect || !image) return null
    return { sx: rect.width / image.width, sy: rect.height / image.height }
  }

  function canDrag(index) {
    if (deletedBoxes.includes(index)) return false
    if (boxTypes[index] === 'added') return true
    return !!showEditedForModel[index]
  }

  function startDrag(mode, e, index) {
    const box = editedBoxes[index]
    if (!box || !image || !canDrag(index)) return
    e.preventDefault()
    e.stopPropagation()
    setActiveBoxIndex(index)
    const scale = getScale()
    if (!scale) return

    dragRef.current = {
      mode,
      boxIndex: index,
      startX: e.clientX,
      startY: e.clientY,
      box: { ...box },
      ...scale,
    }

    window.addEventListener('pointermove', onMove)
    window.addEventListener('pointerup', stopDrag)
  }

  function onMove(e) {
    const s = dragRef.current
    if (!s || !image) return
    const dx = (e.clientX - s.startX) / s.sx
    const dy = (e.clientY - s.startY) / s.sy
    const next = { ...s.box }

    if (s.mode === 'move') {
      next.x = clamp(s.box.x + dx, 0, image.width - s.box.width)
      next.y = clamp(s.box.y + dy, 0, image.height - s.box.height)
    } else if (s.mode === 'se') {
      next.width = clamp(s.box.width + dx, 20, image.width - s.box.x)
      next.height = clamp(s.box.height + dy, 20, image.height - s.box.y)
    } else {
      const newX = clamp(s.box.x + dx, 0, s.box.x + s.box.width - 20)
      const newY = clamp(s.box.y + dy, 0, s.box.y + s.box.height - 20)
      next.width = s.box.width + (s.box.x - newX)
      next.height = s.box.height + (s.box.y - newY)
      next.x = newX
      next.y = newY
    }

    setEditedBoxes((prev) => prev.map((item, idx) => (idx === s.boxIndex ? next : item)))
    setStatus('已手动调整当前修正框，可点击“保存全部修正”')
  }

  function stopDrag() {
    dragRef.current = null
    window.removeEventListener('pointermove', onMove)
    window.removeEventListener('pointerup', stopDrag)
  }

  function onModelBoxClick(idx, e) {
    e.preventDefault()
    e.stopPropagation()
    setActiveBoxIndex(idx)
    setBoxMenuIndex((prev) => (prev === idx ? null : idx))
  }

  function chooseModify() {
    if (boxMenuIndex == null) return
    const idx = boxMenuIndex
    setShowEditedForModel((prev) => ({ ...prev, [idx]: true }))
    setDeletedBoxes((prev) => prev.filter((x) => x !== idx))
    setActiveBoxIndex(idx)
    setBoxMenuIndex(null)
    setStatus(`模型框 ${idx + 1} 已进入修改模式，可拖动修正框`)
  }

  function chooseDelete() {
    if (boxMenuIndex == null) return
    const idx = boxMenuIndex
    setDeletedBoxes((prev) => (prev.includes(idx) ? prev : [...prev, idx]))
    setShowEditedForModel((prev) => {
      const next = { ...prev }
      delete next[idx]
      return next
    })
    setBoxMenuIndex(null)
    setStatus(`模型框 ${idx + 1} 已标记删除（虚线显示）`)
  }

  async function addBox() {
    if (!image) return
    const base = editedBoxes[activeBoxIndex] || originalBoxes[activeBoxIndex] || { x: image.width * 0.2, y: image.height * 0.2, width: image.width * 0.24, height: image.height * 0.24 }
    const next = getNewBox(base, image)

    try {
      const created = await createAddedBox(image.id, next, getNextAddedIndex(boxTypes))
      setEditedBoxes((prev) => {
        const nextList = [...prev, next]
        setActiveBoxIndex(nextList.length - 1)
        return nextList
      })
      setBoxTypes((prev) => [...prev, 'added'])
      setDetectionIds((prev) => [...prev, null])
      setAddedBoxIds((prev) => [...prev, created?.added_box?.id ?? null])
      setStatus('已新增修正框并保存，可继续拖动')
      loadRecords(false)
    } catch (err) {
      console.error(err)
      setStatus(`新增修正框失败：${err.message}`)
    }
  }

  async function saveAllCorrections() {
    if (!editedBoxes.length) return
    try {
      const modelSaves = await Promise.all(
        editedBoxes.map((box, idx) => {
          if (boxTypes[idx] !== 'model') return Promise.resolve(null)
          if (deletedBoxes.includes(idx)) return Promise.resolve(null)
          if (!showEditedForModel[idx]) return Promise.resolve(null)
          if (!originalBoxes[idx]) return Promise.resolve(null)
          return compareBoxes(originalBoxes[idx], box, detectionIds[idx], idx)
        })
      )

      const addedUpdates = await Promise.all(
        editedBoxes.map((box, idx) => {
          if (boxTypes[idx] !== 'added') return Promise.resolve(null)
          if (!addedBoxIds[idx]) return Promise.resolve(null)
          return updateAddedBox(addedBoxIds[idx], box)
        })
      )

      const deletedSaves = await Promise.all(
        deletedBoxes.map((idx) => {
          const detectionId = detectionIds[idx]
          if (boxTypes[idx] !== 'model' || !image?.id || detectionId == null) return Promise.resolve(null)
          return saveDeletedBox(image.id, detectionId, idx)
        })
      )

      setMetricsMap((prev) => {
        const next = { ...prev }
        modelSaves.forEach((item, idx) => {
          if (item) next[idx] = item
        })
        return next
      })

      if (addedUpdates.some(Boolean) || deletedSaves.some(Boolean)) {
        setStatus('全部修正已保存（包含新增修正框与删除标记）')
      } else {
        setStatus('全部修正已保存')
      }
      loadRecords(false)
    } catch (err) {
      console.error(err)
      setStatus(`保存失败：${err.message}`)
    }
  }

  const stageStyle = image ? { aspectRatio: `${image.width} / ${image.height}` } : { aspectRatio: '4 / 3' }

  return (
    <div className="app-shell">
      <header className="hero">
        <div>
          <h1>口腔影像异常区域辅助识别系统</h1>
          <p className="hero-copy">上传口腔图片后，系统会自动识别异常区域，并支持人工修正与历史记录保存。</p>
        </div>
      </header>

      <main className="workspace">
        <section className="panel panel-upload">
          <div className="panel-title-row">
            <h2>图片上传</h2>
          </div>
          <label className="upload-card">
            <input type="file" accept="image/png,image/jpeg,image/jpg" onChange={handleUpload} />
            <span>点击上传口腔图片</span>
            <small>{status}</small>
          </label>

          <div className="button-row button-row-stack">
            <button onClick={addBox} disabled={!image}>新增修正框</button>
            <button onClick={saveAllCorrections} disabled={!image}>保存全部修正</button>
          </div>

          <HistoryPanel records={records} status={historyStatus} onOpen={openRecord} onRefresh={() => loadRecords()} />
        </section>

        <section className="panel panel-canvas">
          <div className="panel-title-row">
            <h2>识别与交互标注区</h2>
            <span className="legend"><i className="legend-original" />模型框<i className="legend-edited" />模型对应修正框<i className="legend-added" />新增修正框</span>
          </div>
          <div className="canvas-stage" ref={stageRef} style={stageStyle} onClick={() => setBoxMenuIndex(null)}>
            {image ? <img src={image.url} alt={image.name} className="preview-image" /> : <div className="empty-stage">上传图片后将在这里显示识别结果</div>}
            {(isDetecting || !image) && <div className="stage-message"><div className="stage-message-card">{isDetecting ? '图片上传成功，正在调用后端识别...' : '上传图片后将在这里显示识别结果'}</div></div>}

            {image && originalBoxes.map((box, idx) => box ? (
              <Box
                key={`model-${idx}`}
                box={box}
                image={image}
                className={`box-original ${deletedBoxes.includes(idx) ? 'deleted' : ''} ${activeBoxIndex === idx ? 'active' : ''}`}
                label={`模型框 ${idx + 1}`}
                onClick={(e) => onModelBoxClick(idx, e)}
              />
            ) : null)}

            {image && editedBoxes.map((box, idx) => {
              if (!box) return null
              if (boxTypes[idx] === 'model' && !showEditedForModel[idx]) return null
              return (
                <EditBox
                  key={`edit-${idx}`}
                  box={box}
                  image={image}
                  onStart={(mode, e) => startDrag(mode, e, idx)}
                  active={idx === activeBoxIndex}
                  index={idx}
                  deleted={deletedBoxes.includes(idx)}
                  boxType={boxTypes[idx]}
                  displayLabel={boxTypes[idx] === 'added' ? `新增修正框 ${getAddedDisplayIndex(boxTypes, idx)}` : `修正框 ${idx + 1}`}
                />
              )
            })}

            {boxMenuIndex != null && (
              <div className="box-action-menu" onClick={(e) => e.stopPropagation()}>
                <button type="button" onClick={() => setBoxMenuIndex(null)}>取消</button>
                <button type="button" onClick={chooseModify}>修改</button>
                <button type="button" onClick={chooseDelete}>删除</button>
              </div>
            )}
          </div>
        </section>

        <section className="panel panel-metrics">
          <h2>数值专区</h2>
          <MetricSection title="模型框" box={originalBoxes[activeBoxIndex]} />
          <MetricSection title={boxTypes[activeBoxIndex] === 'added' ? '新增修正框' : '修正框'} box={editedBoxes[activeBoxIndex]} />
          <div className="correction-list">
            {editedBoxes.map((box, idx) => {
              const isAdded = boxTypes[idx] === 'added'
              const title = isAdded
                ? `新增修正框 ${getAddedDisplayIndex(boxTypes, idx)}`
                : (deletedBoxes.includes(idx)
                  ? `已删除模型框 ${idx + 1}`
                  : (showEditedForModel[idx] ? `修正框 ${idx + 1}` : `模型框 ${idx + 1}`))
              const sub = box ? `x:${box.x.toFixed(0)} y:${box.y.toFixed(0)} w:${box.width.toFixed(0)} h:${box.height.toFixed(0)}` : '未修正'
              return (
                <button key={idx} type="button" className={`correction-item ${idx === activeBoxIndex ? 'active' : ''}`} onClick={() => setActiveBoxIndex(idx)}>
                  <strong>{title}</strong>
                  <span>{sub}</span>
                  <small>{isAdded ? '新增修正框索引从1开始' : `模型索引 ${idx}`}</small>
                </button>
              )
            })}
          </div>

          <div className="metric-section">
            <h3>模型输出信息</h3>
            <div className="metric-grid">
              <MetricItem label="confidence" value={confidenceMap[activeBoxIndex] != null ? Number(confidenceMap[activeBoxIndex]).toFixed(4) : '--'} />
              <MetricItem label="status" value={originalBoxes[activeBoxIndex] ? 'detected' : '--'} />
              <MetricItem label="image" value={image?.name || '--'} />
              <MetricItem label="record" value={image?.id ? `#${image.id}` : '--'} />
            </div>
          </div>

          <div className="stats-grid">
            <StatCard title="IoU" value={metricsMap[activeBoxIndex] ? Number(metricsMap[activeBoxIndex].iou).toFixed(4) : '--'} accent="cyan" />
            <StatCard title="差异比例" value={metricsMap[activeBoxIndex] ? `${(Number(metricsMap[activeBoxIndex].difference_ratio) * 100).toFixed(2)}%` : '--'} accent="amber" />
            <StatCard title="模型框面积" value={metricsMap[activeBoxIndex] ? Number(metricsMap[activeBoxIndex].original_area).toFixed(0) : '--'} accent="pink" />
            <StatCard title="修正框面积" value={metricsMap[activeBoxIndex] ? Number(metricsMap[activeBoxIndex].edited_area).toFixed(0) : '--'} accent="green" />
          </div>
        </section>
      </main>
    </div>
  )
}

function getAddedDisplayIndex(boxTypes, idx) {
  if (boxTypes[idx] !== 'added') return idx + 1
  let count = 0
  for (let i = 0; i <= idx; i += 1) {
    if (boxTypes[i] === 'added') count += 1
  }
  return count
}

function getNextAddedIndex(boxTypes) {
  return boxTypes.filter((item) => item === 'added').length + 1
}

function HistoryPanel({ records, status, onOpen, onRefresh }) {
  return (
    <div className="history-panel">
      <div className="history-head"><h3>历史记录</h3><button type="button" onClick={onRefresh}>刷新</button></div>
      <p>{status}</p>
      <div className="history-list">
        {records.slice(0, 8).map((record) => {
          const modelCount = record.detections?.length || (record.detection ? 1 : 0)
          const correctedCount = record.corrections?.length || 0
          const addedCount = record.added_boxes?.length || 0
          return (
            <button className="history-item" key={record.id} type="button" onClick={() => onOpen(record)}>
              <strong>#{record.id} {record.filename}</strong>
              <span>{new Date(record.upload_time).toLocaleString()}</span>
              <em>模型框: {modelCount} / 模型对应修正框: {correctedCount} / 新增修正框: {addedCount}</em>
              {record.url ? <small>图片已保存</small> : <small>图片已丢失/未保存</small>}
            </button>
          )
        })}
      </div>
    </div>
  )
}

function Box({ box, image, className, label, onClick }) {
  const style = {
    left: `${(box.x / image.width) * 100}%`,
    top: `${(box.y / image.height) * 100}%`,
    width: `${(box.width / image.width) * 100}%`,
    height: `${(box.height / image.height) * 100}%`,
  }
  return <div className={`box-overlay ${className}`} style={style} onClick={onClick}><span className="box-label">{label}</span></div>
}

function EditBox({ box, image, onStart, active, deleted, boxType, displayLabel }) {
  const style = {
    left: `${(box.x / image.width) * 100}%`,
    top: `${(box.y / image.height) * 100}%`,
    width: `${(box.width / image.width) * 100}%`,
    height: `${(box.height / image.height) * 100}%`,
  }
  return <div className={`box-overlay box-edited ${boxType === 'added' ? 'box-added' : ''} ${active ? 'active' : ''} ${deleted ? 'deleted' : ''}`} style={style} onPointerDown={(e) => onStart('move', e)}><span className="box-label">{displayLabel}</span><div className="resize-handle handle-se" style={{ width: 12, height: 12 }} onPointerDown={(e) => onStart('se', e)} /><div className="resize-handle handle-nw" style={{ width: 12, height: 12 }} onPointerDown={(e) => onStart('nw', e)} /></div>
}

function MetricSection({ title, box }) {
  return <div className="metric-section"><h3>{title}</h3><div className="metric-grid"><MetricItem label="x" value={box ? box.x.toFixed(0) : '--'} /><MetricItem label="y" value={box ? box.y.toFixed(0) : '--'} /><MetricItem label="width" value={box ? box.width.toFixed(0) : '--'} /><MetricItem label="height" value={box ? box.height.toFixed(0) : '--'} /></div></div>
}

function getNewBox(base, image) {
  const offset = 18
  const width = base ? base.width : Math.round(image.width * 0.24)
  const height = base ? base.height : Math.round(image.height * 0.24)
  return {
    x: clamp((base?.x ?? image.width * 0.2) + offset, 0, image.width - width),
    y: clamp((base?.y ?? image.height * 0.2) + offset, 0, image.height - height),
    width: clamp(width, 20, image.width),
    height: clamp(height, 20, image.height),
  }
}

function MetricItem({ label, value }) { return <div className="metric-item"><span>{label}</span><strong>{value}</strong></div> }
function StatCard({ title, value, accent }) { return <div className={`stat-card ${accent}`}><span>{title}</span><strong>{value}</strong></div> }
