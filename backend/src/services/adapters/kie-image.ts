/**
 * kie.ai 图片生成 Adapter
 * 统一走 POST /api/v1/jobs/createTask { model, input }，异步轮询 /api/v1/jobs/recordInfo。
 * kie.ai 只接受公网 URL 的参考图，dataURL 在 prepareRecord 中先上传到 kie 临时存储。
 * 各模型家族的 input 字段不同，见下方 FAMILIES 映射表；文生图/编辑模型 id 按是否有参考图自动切换。
 */
import type {
  ImageProviderAdapter,
  ProviderRequest,
  AIConfig,
  ImageGenerationRecord,
  ImageGenResponse,
  ImagePollResponse,
} from './types'
import {
  buildKieCreateTaskRequest,
  buildKiePollRequest,
  parseKieCreateResponse,
  parseKieRecord,
  extractKieResultUrls,
  uploadAllToKie,
  parseUrlArray,
} from './kie-common'

interface KieImageFamily {
  /** 判断配置的模型 id 是否属于该家族 */
  match: (model: string) => boolean
  /** 文生图模型 id */
  t2i: string
  /** 带参考图时的模型 id（与 t2i 相同表示同一模型同时支持两种） */
  edit: string
  /** 参考图字段名 */
  imageField: 'image_urls' | 'image_input' | 'input_urls'
  /** 参考图数量上限 */
  maxImages: number
  /** 该家族支持的宽高比 */
  ratios: string[]
  /** 补充字段 */
  extra?: (record: ImageGenerationRecord) => Record<string, any>
  /** 宽高比未指定时的默认值 */
  defaultRatio: string
}

const STD_RATIOS = ['1:1', '9:16', '16:9', '3:4', '4:3', '3:2', '2:3', '5:4', '4:5', '21:9']

const FAMILIES: KieImageFamily[] = [
  {
    // google/nano-banana(-edit)：edit 用 image_urls，最多 10 张
    match: (m) => /^(google\/)?nano-banana(-edit)?$/.test(m),
    t2i: 'google/nano-banana',
    edit: 'google/nano-banana-edit',
    imageField: 'image_urls',
    maxImages: 10,
    ratios: STD_RATIOS,
    defaultRatio: '16:9',
    extra: () => ({ output_format: 'png' }),
  },
  {
    // nano-banana-2：单一模型，参考图字段 image_input（最多 14 张）
    match: (m) => /^(google\/)?nano-banana-2$/.test(m),
    t2i: 'nano-banana-2',
    edit: 'nano-banana-2',
    imageField: 'image_input',
    maxImages: 14,
    ratios: ['1:1', '2:3', '3:2', '1:4', '4:1', '3:4', '4:3', '4:5', '5:4', '1:8', '8:1', '9:16', '16:9', '21:9'],
    defaultRatio: '16:9',
    extra: () => ({ resolution: '2K', output_format: 'png' }),
  },
  {
    // seedream 4.5：文生图 / edit 分开，quality basic=2K high=4K
    match: (m) => /^seedream\/4\.5-(text-to-image|edit)$/.test(m),
    t2i: 'seedream/4.5-text-to-image',
    edit: 'seedream/4.5-edit',
    imageField: 'image_urls',
    maxImages: 14,
    ratios: ['1:1', '4:3', '3:4', '16:9', '9:16', '2:3', '3:2', '21:9'],
    defaultRatio: '16:9',
    extra: () => ({ quality: 'basic' }),
  },
  {
    // gpt-image-2：文生图 / image-to-image 分开，i2i 参考图字段为 input_urls（最多 16 张）
    match: (m) => /^gpt-image-2-(text-to-image|image-to-image)$/.test(m),
    t2i: 'gpt-image-2-text-to-image',
    edit: 'gpt-image-2-image-to-image',
    imageField: 'input_urls',
    maxImages: 16,
    ratios: ['auto', '1:1', '3:2', '2:3', '4:3', '3:4', '5:4', '4:5', '16:9', '9:16', '2:1', '1:2', '3:1', '1:3', '21:9', '9:21'],
    defaultRatio: '16:9',
    extra: () => ({ resolution: '2K' }),
  },
]

function gcd(a: number, b: number): number {
  return b === 0 ? a : gcd(b, a % b)
}

/** 把 "1920x1080" / "16:9" 解析为最接近的受支持宽高比；无法解析返回 null */
function pickAspectRatio(size: string | null | undefined, allowed: string[]): string | null {
  if (!size) return null
  const s = size.trim().toLowerCase()
  if (s === 'auto') return allowed.includes('auto') ? 'auto' : null
  const m = /^(\d+(?:\.\d+)?)\s*[x:×*]\s*(\d+(?:\.\d+)?)$/.exec(s)
  if (!m) return null
  const w = Number(m[1])
  const h = Number(m[2])
  if (!w || !h) return null
  const g = gcd(Math.round(w), Math.round(h))
  const exact = `${Math.round(w) / g}:${Math.round(h) / g}`
  if (allowed.includes(exact)) return exact
  // 选取数值最接近的比例（按 log 距离）
  const target = Math.log(w / h)
  let best: string | null = null
  let bestDiff = Infinity
  for (const r of allowed) {
    const [a, b] = r.split(':').map(Number)
    if (!a || !b) continue
    const diff = Math.abs(Math.log(a / b) - target)
    if (diff < bestDiff) { bestDiff = diff; best = r }
  }
  return best
}

export class KieImageAdapter implements ImageProviderAdapter {
  provider = 'kie'

  /** 参考图（dataURL）上传到 kie.ai，替换为公网 URL */
  async prepareRecord(config: AIConfig, record: ImageGenerationRecord): Promise<ImageGenerationRecord> {
    const refs = parseUrlArray(record.referenceImages)
    if (!refs.length) return record
    const urls = await uploadAllToKie(config, refs)
    return { ...record, referenceImages: JSON.stringify(urls) }
  }

  buildGenerateRequest(config: AIConfig, record: ImageGenerationRecord): ProviderRequest {
    const configured = (record.model || config.model || '').trim()
    if (!configured) throw new Error('kie.ai 图片模型未配置')
    const refs = parseUrlArray(record.referenceImages).filter((u) => /^https?:\/\//i.test(u))
    const prompt = record.prompt || ''
    const family = FAMILIES.find((f) => f.match(configured))

    if (!family) {
      // 未知模型：通用 input，原样使用配置的模型 id
      const input: Record<string, any> = { prompt }
      if (refs.length) input.image_urls = refs
      const ratio = pickAspectRatio(record.size, STD_RATIOS)
      if (ratio) input.aspect_ratio = ratio
      return buildKieCreateTaskRequest(config, configured, input)
    }

    // 有参考图用 edit 模型，无参考图用文生图模型（edit 模型必须带图）
    const model = refs.length ? family.edit : family.t2i

    const input: Record<string, any> = { prompt, ...(family.extra ? family.extra(record) : {}) }
    input.aspect_ratio = pickAspectRatio(record.size, family.ratios) || family.defaultRatio
    if (refs.length) input[family.imageField] = refs.slice(0, family.maxImages)

    return buildKieCreateTaskRequest(config, model, input)
  }

  parseGenerateResponse(result: any): ImageGenResponse {
    const { taskId } = parseKieCreateResponse(result)
    return { isAsync: true, taskId }
  }

  buildPollRequest(config: AIConfig, taskId: string): ProviderRequest {
    return buildKiePollRequest(config, taskId)
  }

  parsePollResponse(result: any): ImagePollResponse {
    const r = parseKieRecord(result)
    if (r.status === 'completed') {
      const imageUrl = r.resultUrls[0]
      if (!imageUrl) return { status: 'failed', error: 'kie.ai 任务成功但未返回图片地址' }
      return { status: 'completed', imageUrl }
    }
    if (r.status === 'failed') return { status: 'failed', error: r.error }
    return { status: r.status }
  }

  extractImageUrl(result: any): string | null {
    return extractKieResultUrls(result)[0] || null
  }

  extractImageBase64(_result: any): { data: string; mimeType: string } | null {
    return null
  }
}
