import React, { useEffect, useRef, useState } from 'react'
import { detectImage, compareBoxes, createAddedBox, updateAddedBox, saveDeletedBox, fetchRecords } from './api'

const clamp = (v, min, max) => Math.min(Math.max(v, min), max)

export default function App() {
  const stageRef = useRef(null)
  const dragRef = useRef(null)
  const timerRef = useRef(null)

  const [status, setStatus] = useState('等待上传图片')
  const [image, setImage] = useState(null)
  const [originalBoxes, setOriginalBoxes] = useState([])
  const [editedBoxes, setEditedBoxes] = useState([])
  const [activeBoxIndex, setActiveBoxIndex] = useState(0)
  const [metricsMap, setMetricsMap] = useState({})
  const [confidenceMap, setConfidenceMap] = useState({})
  const [isDetecting, setIsDetecting] = useState(false)
  const [detectionIds, setDetectionIds] = useState([])
  const [addedBoxIds, setAddedBoxIds] = useState([])
  const [boxTypes, setBoxTypes] = useState([])
  const [deletedBoxes, setDeletedBoxes] = useState([])
  const [records, setRecords] = useState([])
  const [historyStatus, setHistoryStatus] = useState('暂无历史记录')

  useEffect(() => {
    loadRecords()
  }, [])

  useEffect(() => {
    return () => timerRef.current && clearTimeout(timerRef.current)
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
      setImage({
        id: result.image.id,
        url: result.image.url || localUrl,
        width: result.image.width,
        height: result.image.height,
        name: result.image.name,
      })
      setOriginalBoxes(detections.map((det) => ({ ...det.bbox })))
      setEditedBoxes(detections.map((det) => ({ ...det.bbox })))
      setActiveBoxIndex(0)
      setDeletedBoxes([])
      setMetricsMap({})
      setConfidenceMap(Object.fromEntries(detections.map((det, idx) => [idx, det.confidence ?? null])))
      setDetectionIds(detections.map((det) => det.id ?? null))
      setAddedBoxIds(detections.map(() => null))
      setBoxTypes(detections.map(() => 'model'))
      setStatus(detections.length ? `识别完成，共检测到 ${detections.length} 个区域，可在结果框内继续修正` : '未检测到目标区域，图片已保存')
      loadRecords(false)
    } catch (err) {
      console.error(err)
      setStatus(`识别失败：${err.message}`)
      setImage(null)
      setOriginalBoxes([])
      setEditedBoxes([])
      setActiveBoxIndex(0)
      setDeletedBoxes([])
      setMetricsMap({})
      setConfidenceMap({})
      setDetectionIds([])
      setAddedBoxIds([])
      setBoxTypes([])
    } finally {
      setIsDetecting(false)
    }
  }

  function openRecord(record) {
    const detections = record.detections || (record.detection ? [record.detection] : [])
    const corrections = record.corrections || []
    const addedBoxes = record.added_boxes || []
    const deletedIndexes = record.deleted_box_indexes || []
    const modelOriginal = detections.map((det) => det.bbox)
    const modelEdited = detections.map((det, idx) => corrections.find((item) => item.box_index === idx)?.bbox || det.bbox)
    const mergedOriginal = [...modelOriginal]
    const mergedEdited = [...modelEdited, ...addedBoxes.map((item) => item.bbox)]

    setImage({
      id: record.id,
      url: record.url || record.image_url || '',
      width: record.width,
      height: record.height,
      name: record.filename,
    })
    setOriginalBoxes(mergedOriginal)
    setEditedBoxes(mergedEdited)
    setActiveBoxIndex(0)
    setDeletedBoxes(deletedIndexes)
    setConfidenceMap(Object.fromEntries(modelOriginal.map((_, idx) => [idx, detections[idx]?.confidence ?? null])))
    setDetectionIds([...detections.map((det) => det.id ?? null), ...addedBoxes.map(() => null)])
    setAddedBoxIds([...detections.map(() => null), ...addedBoxes.map((item) => item.id ?? null)])
    setBoxTypes([...modelOriginal.map(() => 'model'), ...addedBoxes.map(() => 'added')])
    setMetricsMap(Object.fromEntries(corrections.map((item) => [item.box_index ?? 0, item])))
    setStatus('已打开历史检测记录')
  }

  function getScale() {
    const rect = stageRef.current?.getBoundingClientRect()
    if (!rect || !image) return null
    return { sx: rect.width / image.width, sy: rect.height / image.height }
  }

  function startDrag(mode, e, index) {
    const box = editedBoxes[index]
    if (!box || !image || deletedBoxes.includes(index)) return
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
    setStatus('已手动调整当前区域，修正结果将自动保存')
  }

  function stopDrag() {
    dragRef.current = null
    window.removeEventListener('pointermove', onMove)
    window.removeEventListener('pointerup', stopDrag)
  }

  async function addBox() {
    if (!image) return
    const base = originalBoxes[activeBoxIndex] || editedBoxes[activeBoxIndex] || { x: image.width * 0.2, y: image.height * 0.2, width: image.width * 0.24, height: image.height * 0.24 }
    const next = getNewBox(base, image)

    try {
      const created = await createAddedBox(image.id, next, editedBoxes.length)
      setOriginalBoxes((prev) => [...prev, null])
      setEditedBoxes((prev) => {
        const nextList = [...prev, next]
        setActiveBoxIndex(nextList.length - 1)
        return nextList
      })
      setConfidenceMap((prev) => ({ ...prev, [editedBoxes.length]: null }))
      setDetectionIds((prev) => [...prev, null])
      setAddedBoxIds((prev) => [...prev, created?.added_box?.id ?? null])
      setBoxTypes((prev) => [...prev, 'added'])
      setStatus('已新增修正框并单独保存，可继续拖动调整')
      loadRecords(false)
    } catch (err) {
      console.error(err)
      setStatus(`新增框失败：${err.message}`)
    }
  }

  function removeBox() {
    if (!originalBoxes[activeBoxIndex]) return
    setDeletedBoxes((prev) => (prev.includes(activeBoxIndex) ? prev : [...prev, activeBoxIndex]))
    setStatus('已删除当前模型框，当前框已变为虚线')
  }

  function resetBox() {
    if (!originalBoxes[activeBoxIndex]) return
    setDeletedBoxes((prev) => prev.filter((idx) => idx !== activeBoxIndex))
    setEditedBoxes((prev) => prev.map((item, idx) => (idx === activeBoxIndex ? { ...originalBoxes[activeBoxIndex] } : item)))
    setStatus('已重置当前框为模型原始框')
  }

  async function saveAllCorrections() {
    if (!editedBoxes.length) return
    try {
      const results = await Promise.all(editedBoxes.map((box, idx) => {
        if (deletedBoxes.includes(idx)) return Promise.resolve(null)
        if (boxTypes[idx] !== 'model') return Promise.resolve(null)
        if (!originalBoxes[idx]) return Promise.resolve(null)
        return compareBoxes(originalBoxes[idx], box, detectionIds[idx], idx)
      }))

      const addedUpdates = await Promise.all(editedBoxes.map((box, idx) => {
        if (boxTypes[idx] !== 'added') return Promise.resolve(null)
        if (!addedBoxIds[idx]) return Promise.resolve(null)
        return updateAddedBox(addedBoxIds[idx], box)
      }))

      const deletedSaves = await Promise.all(
        deletedBoxes.map((idx) => {
          const detectionId = detectionIds[idx]
          if (boxTypes[idx] !== 'model' || !image?.id || detectionId == null) return Promise.resolve(null)
          return saveDeletedBox(image.id, detectionId, idx)
        })
      )

      setMetricsMap((prev) => {
        const next = { ...prev }
        results.forEach((item, idx) => {
          if (item) next[idx] = item
        })
        return next
      })
      setStatus('全部修正已保存')
      if (addedUpdates.some(Boolean) || deletedSaves.some(Boolean)) {
        setStatus('全部修正已保存（包含新增修正框位置更新/删除模型框）')
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
            <small>支持 JPG / PNG，识别和修正结果会自动保存</small>
          </label>
          <div className="button-row button-row-stack">
            <button onClick={addBox} disabled={!image}>新增修正框</button>
            <button onClick={removeBox} disabled={!originalBoxes.length || boxTypes[activeBoxIndex] !== 'model'}>删除当前模型框</button>
            <button onClick={resetBox} disabled={!originalBoxes.length}>重置当前框</button>
          </div>
          <HistoryPanel records={records} status={historyStatus} onOpen={openRecord} onRefresh={() => loadRecords()} />
        </section>

        <section className="panel panel-canvas">
          <div className="panel-title-row">
            <h2>识别与交互标注区</h2>
            <span className="legend"><i className="legend-original" />模型原始框<i className="legend-edited" />人工修正框<i className="legend-added" />新增修正框</span>
          </div>
          <div className="canvas-stage" ref={stageRef} style={stageStyle}>
            {image ? <img src={image.url} alt={image.name} className="preview-image" /> : <div className="empty-stage">上传图片后将在这里显示识别结果</div>}
            {(isDetecting || !image) && <div className="stage-message"><div className="stage-message-card">{isDetecting ? '图片上传成功，正在调用后端识别...' : '上传图片后将在这里显示识别结果'}</div></div>}
            {image && originalBoxes.map((box, idx) => box ? <Box key={idx} box={box} image={image} className={`box-original ${deletedBoxes.includes(idx) ? 'deleted' : ''}`} label={`模型框 ${idx + 1}`} /> : null)}
            {image && editedBoxes.map((box, idx) => box ? <EditBox key={idx} box={box} image={image} onStart={(mode, e) => startDrag(mode, e, idx)} active={idx === activeBoxIndex} index={idx} deleted={deletedBoxes.includes(idx)} boxType={boxTypes[idx]} /> : null)}
          </div>
        </section>

        <section className="panel panel-metrics">
          <h2>数值专区</h2>
          <MetricSection title="模型框" box={originalBoxes[activeBoxIndex]} />
          <MetricSection title={boxTypes[activeBoxIndex] === 'added' ? '新增修正框' : '修正框'} box={editedBoxes[activeBoxIndex]} />
          <div className="correction-list">{editedBoxes.map((box, idx) => <button key={idx} type="button" className={`correction-item ${idx === activeBoxIndex ? 'active' : ''}`} onClick={() => setActiveBoxIndex(idx)}><strong>{boxTypes[idx] === 'added' ? `新增修正框 ${idx + 1}` : (deletedBoxes.includes(idx) ? `已删除模型框 ${idx + 1}` : `模型框 ${idx + 1}`)}</strong><span>{box ? `x:${box.x.toFixed(0)} y:${box.y.toFixed(0)} w:${box.width.toFixed(0)} h:${box.height.toFixed(0)}` : '未修正'}</span><small>{boxTypes[idx] === 'added' ? '新增修正框已独立保存' : (metricsMap[idx] ? `IoU ${Number(metricsMap[idx].iou).toFixed(3)}` : '未保存')}</small></button>)}</div>
          <div className="box-tabs">{editedBoxes.map((_, idx) => <button key={idx} className={idx === activeBoxIndex ? 'active' : ''} type="button" onClick={() => setActiveBoxIndex(idx)}>框 {idx + 1}</button>)}<button type="button" className="save-all" onClick={() => saveAllCorrections()}>保存全部修正</button></div>
          <div className="stats-grid">
            <StatCard title="IoU" value={metricsMap[activeBoxIndex] ? Number(metricsMap[activeBoxIndex].iou).toFixed(4) : '--'} accent="cyan" />
            <StatCard title="差异比例" value={metricsMap[activeBoxIndex] ? `${(Number(metricsMap[activeBoxIndex].difference_ratio) * 100).toFixed(2)}%` : '--'} accent="amber" />
            <StatCard title="模型框面积" value={metricsMap[activeBoxIndex] ? Number(metricsMap[activeBoxIndex].original_area).toFixed(0) : '--'} accent="pink" />
            <StatCard title="修正框面积" value={metricsMap[activeBoxIndex] ? Number(metricsMap[activeBoxIndex].edited_area).toFixed(0) : '--'} accent="green" />
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
        </section>
      </main>
    </div>
  )
}

function HistoryPanel({ records, status, onOpen, onRefresh }) {
  return <div className="history-panel"><div className="history-head"><h3>历史记录</h3><button type="button" onClick={onRefresh}>刷新</button></div><p>{status}</p><div className="history-list">{records.slice(0, 8).map((record) => <button className="history-item" key={record.id} type="button" onClick={() => onOpen(record)}><strong>#{record.id} {record.filename}</strong><span>{new Date(record.upload_time).toLocaleString()}</span><em>{record.detections?.length ? `检测到 ${record.detections.length} 个区域` : record.detection ? '检测到 1 个区域' : '未检测到目标'}</em><small>{record.corrections?.length ? `已保存 ${record.corrections.length} 个修正` : '暂无修正记录'}</small>{record.url ? <small>图片已保存</small> : <small>图片已丢失/未保存</small>}</button>)}</div></div>
}

function Box({ box, image, className, label }) {
  const style = {
    left: `${(box.x / image.width) * 100}%`,
    top: `${(box.y / image.height) * 100}%`,
    width: `${(box.width / image.width) * 100}%`,
    height: `${(box.height / image.height) * 100}%`,
  }
  return <div className={`box-overlay ${className}`} style={style}><span className="box-label">{label}</span></div>
}

function EditBox({ box, image, onStart, active, index, deleted, boxType }) {
  const style = {
    left: `${(box.x / image.width) * 100}%`,
    top: `${(box.y / image.height) * 100}%`,
    width: `${(box.width / image.width) * 100}%`,
    height: `${(box.height / image.height) * 100}%`,
  }
  return <div className={`box-overlay box-edited ${boxType === 'added' ? 'box-added' : ''} ${active ? 'active' : ''} ${deleted ? 'deleted' : ''}`} style={style} onPointerDown={(e) => onStart('move', e)}><span className="box-label">{boxType === 'added' ? `新增修正框 ${index + 1}` : `修正框 ${index + 1}`}</span><div className="resize-handle handle-se" style={{ width: 12, height: 12 }} onPointerDown={(e) => onStart('se', e)} /><div className="resize-handle handle-nw" style={{ width: 12, height: 12 }} onPointerDown={(e) => onStart('nw', e)} /></div>
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

