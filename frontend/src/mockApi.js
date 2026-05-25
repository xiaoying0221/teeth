export async function mockDetect(file) {
  await new Promise((resolve) => setTimeout(resolve, 900))

  const imageUrl = URL.createObjectURL(file)
  const imageSize = await getImageSize(imageUrl)

  const width = Math.round(imageSize.width * 0.34)
  const height = Math.round(imageSize.height * 0.28)
  const x = Math.round(imageSize.width * 0.31)
  const y = Math.round(imageSize.height * 0.36)

  return {
    success: true,
    image: {
      width: imageSize.width,
      height: imageSize.height,
      url: imageUrl,
      name: file.name,
    },
    detections: [
      {
        bbox: { x, y, width, height },
        confidence: 0.823,
        classId: 0,
        className: 'abnormal',
      },
    ],
  }
}

function getImageSize(src) {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.onload = () => resolve({ width: img.naturalWidth, height: img.naturalHeight })
    img.onerror = reject
    img.src = src
  })
}
