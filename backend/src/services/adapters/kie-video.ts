/**
 * kie.ai 视频生成 Adapter
 * 统一任务接口: POST /api/v1/jobs/createTask { model, input }（始终异步）
 * 轮询: GET /api/v1/jobs/recordInfo?taskId=（state + resultJson.resultUrls）
 *
 * kie.ai market 模型的 input 结构因模型族而异，这里按「配置的模型名 → 模型族」分发到显式的 builder 表:
 * - Seedance 2.0 / 2.0 Fast / 2.0 Mini / 2.5: first_frame_url / last_frame_url / reference_*_urls
 *   （首尾帧模式与多模态参考模式互斥，有参考素材时首尾帧降级为参考图）
 * - Seedance 1.5 Pro: input_urls（≤2 张图）
 * - Hailuo 2.3 Pro（图生视频）/ Hailuo 02 Pro（文生视频）
 * - Wan 2.6（文/图生视频）、Wan 2.7 R2V（参考生视频）
 * - Kling 3.0、Kling 2.6（文/图生视频）
 * - 未知模型: 通用 { prompt, image_urls?, duration?, aspect_ratio? }
 *
 * kie.ai 只接受公网 URL，dataURL 参考图在 prepareRecord 中先上传到 kie.ai 临时存储。
 * 需要同时支持文生/图生的族，按是否存在首帧自动选择 model id。
 */
import type {
  VideoProviderAdapter,
  ProviderRequest,
  AIConfig,
  VideoGenerationRecord,
  VideoGenResponse,
  VideoPollResponse,
} from './types'
import {
  buildKieCreateTaskRequest,
  buildKiePollRequest,
  parseKieCreateResponse,
  parseKieRecord,
  extractKieResultUrls,
  uploadDataUrlToKie,
  uploadAllToKie,
  parseUrlArray,
} from './kie-common'

// ============ 通用工具 ============

interface BuildContext {
  prompt: string
  firstFrame: string
  lastFrame: string
  refImages: string[]
  refVideos: string[]
  refAudios: string[]
  duration?: number | null
  resolution: string
  aspectRatio: string
  audio: boolean
  seed?: number | null
  promptExtend?: boolean
  watermark?: boolean
  rawModel: string
}

interface BuiltRequest {
  model: string
  input: Record<string, any>
}

type Builder = (ctx: BuildContext) => BuiltRequest

function toBool(v: unknown, fallback: boolean): boolean {
  if (v === null || v === undefined) return fallback
  return v === true || v === 1 || v === '1' || v === 'true'
}

function clampInt(value: number | null | undefined, min: number, max: number, fallback: number): number {
  const n = Math.round(Number(value))
  if (!Number.isFinite(n) || n <= 0) return fallback
  return Math.min(max, Math.max(min, n))
}

/** 从枚举列表中取最接近的值（用于只支持离散时长的模型） */
function snapDuration(value: number | null | undefined, allowed: number[], fallback: number): number {
  const n = Number(value)
  if (!Number.isFinite(n) || n <= 0) return fallback
  return allowed.reduce((best, cur) => (Math.abs(cur - n) < Math.abs(best - n) ? cur : best), allowed[0])
}

/** 项目侧分辨率: 480p / 720p / 1080p（兼容 2k/4k 写法） */
function normalizeResolution(raw: string | null | undefined): '480p' | '720p' | '1080p' | '4k' {
  const r = (raw || '').trim().toLowerCase()
  if (r === '4k') return '4k'
  if (r === '1080p' || r === '2k' || r === '1080') return '1080p'
  if (r === '480p' || r === '480' || r === '360p') return '480p'
  return '720p'
}

/** 把 resolution 限制在模型允许集合内（超出取最近的较低档，低于取最低档） */
function fitResolution(res: string, allowed: string[]): string {
  if (allowed.includes(res)) return res
  const order = ['480p', '720p', '1080p', '4k']
  const idx = order.indexOf(res)
  for (let i = idx; i >= 0; i--) if (allowed.includes(order[i])) return order[i]
  return allowed[0]
}

/** 宽高比归一：不在允许集合内时回退 fallback */
function fitRatio(raw: string, allowed: string[], fallback: string): string {
  const r = (raw || '').trim().toLowerCase()
  return allowed.includes(r) ? r : fallback
}

function assertPrompt(name: string, prompt: string, min: number, max: number) {
  if (prompt.length < min) throw new Error(`${name} 要求提示词至少 ${min} 个字符`)
  if (prompt.length > max) throw new Error(`提示词超长：${name} 上限 ${max} 字符，当前 ${prompt.length}`)
}

// ============ Seedance ============

interface SeedanceSpec {
  model: string
  label: string
  resolutions: string[]
  maxDuration: number
  maxPrompt: number
  limits: { images: number; videos: number; audios: number }
  /** 尾帧必须搭配首帧 */
  lastNeedsFirst: boolean
}

function seedance2Builder(spec: SeedanceSpec): Builder {
  return (ctx) => {
    assertPrompt(spec.label, ctx.prompt, 3, spec.maxPrompt)
    const hasMultiRef = ctx.refImages.length + ctx.refVideos.length + ctx.refAudios.length > 0

    let firstFrame = ctx.firstFrame
    let lastFrame = ctx.lastFrame
    let refImages = ctx.refImages
    // 首尾帧与多模态参考互斥：存在参考素材时，首/尾帧降级为参考图
    if (hasMultiRef) {
      refImages = [...[firstFrame, lastFrame].filter(Boolean), ...refImages]
      firstFrame = ''
      lastFrame = ''
    }
    if (lastFrame && !firstFrame && spec.lastNeedsFirst) {
      throw new Error(`${spec.label} 的尾帧必须搭配首帧使用`)
    }

    if (refImages.length > spec.limits.images || ctx.refVideos.length > spec.limits.videos || ctx.refAudios.length > spec.limits.audios) {
      throw new Error(`参考素材超限：${spec.label} 图片≤${spec.limits.images}、视频≤${spec.limits.videos}、音频≤${spec.limits.audios}`)
    }

    const input: Record<string, any> = {
      prompt: ctx.prompt,
      resolution: fitResolution(ctx.resolution, spec.resolutions),
      // 有首帧/参考素材时默认 adaptive 跟随素材比例；纯文生视频用指定比例（默认 16:9）
      aspect_ratio: (firstFrame || hasMultiRef) && !ctx.aspectRatio
        ? 'adaptive'
        : fitRatio(ctx.aspectRatio, ['1:1', '4:3', '3:4', '16:9', '9:16', '21:9', 'adaptive'], firstFrame || hasMultiRef ? 'adaptive' : '16:9'),
      duration: clampInt(ctx.duration, 4, spec.maxDuration, 5),
      generate_audio: ctx.audio,
    }
    if (firstFrame) input.first_frame_url = firstFrame
    if (lastFrame) input.last_frame_url = lastFrame
    if (refImages.length) input.reference_image_urls = refImages
    if (ctx.refVideos.length) input.reference_video_urls = ctx.refVideos
    if (ctx.refAudios.length) input.reference_audio_urls = ctx.refAudios
    return { model: spec.model, input }
  }
}

const buildSeedance1_5Pro: Builder = (ctx) => {
  assertPrompt('Seedance 1.5 Pro', ctx.prompt, 3, 20000)
  // 仅支持 ≤2 张图（首/尾帧）；无首尾帧时用参考图兜底
  let urls = [ctx.firstFrame, ctx.lastFrame].filter(Boolean)
  if (!urls.length) urls = ctx.refImages.slice(0, 2)
  const input: Record<string, any> = {
    prompt: ctx.prompt,
    // aspect_ratio 必填且不支持 adaptive
    aspect_ratio: fitRatio(ctx.aspectRatio, ['1:1', '4:3', '3:4', '16:9', '9:16', '21:9'], '16:9'),
    resolution: fitResolution(ctx.resolution, ['480p', '720p', '1080p']),
    duration: clampInt(ctx.duration, 4, 12, 5),
    generate_audio: ctx.audio,
  }
  if (urls.length) input.input_urls = urls
  return { model: 'bytedance/seedance-1.5-pro', input }
}

// ============ Hailuo ============

const buildHailuo: Builder = (ctx) => {
  const first = ctx.firstFrame
  if (!first) {
    // 文生视频（Hailuo 02 Pro）只有 prompt，时长/分辨率不可配
    assertPrompt('Hailuo 02 Pro 文生视频', ctx.prompt, 1, 1500)
    return { model: 'hailuo/02-text-to-video-pro', input: { prompt: ctx.prompt } }
  }
  assertPrompt('Hailuo 2.3 Pro', ctx.prompt, 1, 5000)
  const res = ctx.resolution === '1080p' || ctx.resolution === '4k' ? '1080P' : '768P'
  // 字符串枚举 '6' | '10'；1080P 官方通常仅支持 6 秒（文档未明确说明，保守处理）
  const duration = res === '1080P' ? 6 : snapDuration(ctx.duration, [6, 10], 6)
  return {
    model: 'hailuo/2-3-image-to-video-pro',
    input: { prompt: ctx.prompt, image_url: first, duration: String(duration), resolution: res },
  }
}

// ============ Wan ============

const buildWan26: Builder = (ctx) => {
  assertPrompt('Wan 2.6', ctx.prompt, 1, 5000)
  const input: Record<string, any> = {
    prompt: ctx.prompt,
    duration: String(snapDuration(ctx.duration, [5, 10, 15], 5)),
    resolution: fitResolution(ctx.resolution, ['720p', '1080p']),
    multi_shots: false,
  }
  // Wan 2.6 输入无宽高比字段：文生视频比例由模型决定，图生视频跟随图片
  if (ctx.firstFrame) {
    input.image_urls = [ctx.firstFrame]
    return { model: 'wan/2-6-image-to-video', input }
  }
  return { model: 'wan/2-6-text-to-video', input }
}

const buildWan27R2V: Builder = (ctx) => {
  assertPrompt('Wan 2.7 R2V', ctx.prompt, 1, 5000)
  const refImages = ctx.refImages
  if (!refImages.length && !ctx.refVideos.length && !ctx.firstFrame) {
    throw new Error('Wan 2.7 参考生视频至少需要 1 张参考图、1 段参考视频或首帧图')
  }
  if (refImages.length > 5 || ctx.refVideos.length > 5) {
    throw new Error('参考素材超限：Wan 2.7 R2V 参考图≤5、参考视频≤5')
  }
  const input: Record<string, any> = {
    prompt: ctx.prompt,
    resolution: fitResolution(ctx.resolution, ['720p', '1080p']),
    aspect_ratio: fitRatio(ctx.aspectRatio, ['16:9', '9:16', '1:1', '4:3', '3:4'], '16:9'),
    duration: clampInt(ctx.duration, 2, 10, 5),
  }
  if (refImages.length) input.reference_image = refImages
  else if (ctx.firstFrame && !ctx.refVideos.length) input.reference_image = [ctx.firstFrame]
  if (ctx.refVideos.length) input.reference_video = ctx.refVideos
  if (ctx.firstFrame) input.first_frame = ctx.firstFrame
  if (ctx.refAudios.length) input.reference_voice = ctx.refAudios[0]
  if (ctx.seed != null && Number(ctx.seed) >= 0 && Number(ctx.seed) <= 2147483647) input.seed = Math.round(Number(ctx.seed))
  if (ctx.promptExtend !== undefined) input.prompt_extend = ctx.promptExtend
  if (ctx.watermark !== undefined) input.watermark = ctx.watermark
  return { model: 'wan/2-7-r2v', input }
}

// ============ Kling ============

const buildKling30: Builder = (ctx) => {
  const input: Record<string, any> = {
    prompt: ctx.prompt,
    sound: ctx.audio,
    duration: String(clampInt(ctx.duration, 3, 15, 5)),
    aspect_ratio: fitRatio(ctx.aspectRatio, ['16:9', '9:16', '1:1'], '16:9'),
    // mode 表示画质档位: std ≈ 720p，pro ≈ 1080p，4K
    mode: ctx.resolution === '4k' ? '4K' : ctx.resolution === '1080p' ? 'pro' : 'std',
    multi_shots: false,
  }
  assertPrompt('Kling 3.0', ctx.prompt, 1, 2500)
  // image_urls: 首/尾帧
  const frames = [ctx.firstFrame, ctx.lastFrame].filter(Boolean)
  if (frames.length) input.image_urls = frames
  return { model: 'kling-3.0/video', input }
}

const buildKling26: Builder = (ctx) => {
  assertPrompt('Kling 2.6', ctx.prompt, 1, 1000)
  const duration = String(snapDuration(ctx.duration, [5, 10], 5))
  if (ctx.firstFrame) {
    return {
      model: 'kling-2.6/image-to-video',
      input: { prompt: ctx.prompt, image_urls: [ctx.firstFrame], sound: ctx.audio, duration },
    }
  }
  return {
    model: 'kling-2.6/text-to-video',
    input: {
      prompt: ctx.prompt,
      sound: ctx.audio,
      aspect_ratio: fitRatio(ctx.aspectRatio, ['1:1', '16:9', '9:16'], '16:9'),
      duration,
    },
  }
}

// ============ 通用兜底 ============

function buildGeneric(model: string): Builder {
  return (ctx) => {
    const input: Record<string, any> = { prompt: ctx.prompt }
    const images = [ctx.firstFrame, ctx.lastFrame, ...ctx.refImages].filter(Boolean)
    if (images.length) input.image_urls = images
    if (ctx.duration) input.duration = Math.round(Number(ctx.duration))
    if (ctx.aspectRatio) input.aspect_ratio = ctx.aspectRatio
    return { model, input }
  }
}

// ============ 模型族分发表 ============

const SEEDANCE_2: SeedanceSpec = {
  model: 'bytedance/seedance-2',
  label: 'Seedance 2.0',
  resolutions: ['480p', '720p', '1080p'],
  maxDuration: 15,
  maxPrompt: 20000,
  limits: { images: 9, videos: 3, audios: 3 },
  lastNeedsFirst: false,
}

const FAMILY_BUILDERS: Array<{ name: string; match: RegExp; build: (rawModel: string) => Builder }> = [
  { name: 'seedance-1.5-pro', match: /seedance[-.]?1[-.]5/, build: () => buildSeedance1_5Pro },
  {
    name: 'seedance-2.5',
    match: /seedance[-.]?2[-.]5/,
    build: () => seedance2Builder({
      model: 'bytedance/seedance-2-5',
      label: 'Seedance 2.5',
      resolutions: ['480p', '720p', '1080p'],
      maxDuration: 30,
      maxPrompt: 30000,
      limits: { images: 30, videos: 10, audios: 10 },
      lastNeedsFirst: true,
    }),
  },
  {
    name: 'seedance-2-mini',
    match: /seedance[-.]?2[-.]?mini/,
    build: () => seedance2Builder({ ...SEEDANCE_2, model: 'bytedance/seedance-2-mini', label: 'Seedance 2.0 Mini', resolutions: ['480p', '720p'] }),
  },
  {
    name: 'seedance-2-fast',
    match: /seedance[-.]?2[-.]?fast/,
    build: () => seedance2Builder({ ...SEEDANCE_2, model: 'bytedance/seedance-2-fast', label: 'Seedance 2.0 Fast', resolutions: ['480p', '720p'] }),
  },
  { name: 'seedance-2', match: /seedance[-.]?2/, build: () => seedance2Builder(SEEDANCE_2) },
  { name: 'hailuo', match: /hailuo/, build: () => buildHailuo },
  { name: 'wan-2.7-r2v', match: /wan[-/.]?2[-.]7|r2v/, build: () => buildWan27R2V },
  { name: 'wan-2.6', match: /wan/, build: () => buildWan26 },
  { name: 'kling-3.0', match: /kling[-/.]?3/, build: () => buildKling30 },
  { name: 'kling-2.6', match: /kling/, build: () => buildKling26 },
]

function resolveBuilder(model: string): Builder {
  const key = model.trim().toLowerCase().replace(/[\s_]+/g, '-')
  for (const fam of FAMILY_BUILDERS) {
    if (fam.match.test(key)) return fam.build(model)
  }
  return buildGeneric(model)
}

// ============ Adapter ============

export class KieVideoAdapter implements VideoProviderAdapter {
  provider = 'kie'

  /** kie.ai 只接受公网 URL：把 dataURL 参考图上传到 kie.ai 临时存储并改写记录 */
  async prepareRecord(config: AIConfig, record: VideoGenerationRecord): Promise<VideoGenerationRecord> {
    const next: VideoGenerationRecord = { ...record }
    const single = async (v?: string | null) => (v && v.startsWith('data:') ? uploadDataUrlToKie(config, v) : v)
    next.imageUrl = await single(record.imageUrl)
    next.firstFrameUrl = await single(record.firstFrameUrl)
    next.lastFrameUrl = await single(record.lastFrameUrl)

    const refs = parseUrlArray(record.referenceImageUrls)
    if (refs.some((u) => u.startsWith('data:'))) {
      next.referenceImageUrls = JSON.stringify(await uploadAllToKie(config, refs))
    }
    return next
  }

  buildGenerateRequest(config: AIConfig, record: VideoGenerationRecord): ProviderRequest {
    const rawModel = (record.model || config.model || '').trim()
    if (!rawModel) throw new Error('kie.ai 视频生成需要指定模型（如 bytedance/seedance-2）')

    const ctx: BuildContext = {
      prompt: (record.prompt || '').trim(),
      firstFrame: (record.firstFrameUrl || record.imageUrl || '').trim(),
      lastFrame: (record.lastFrameUrl || '').trim(),
      refImages: parseUrlArray(record.referenceImageUrls),
      refVideos: parseUrlArray(record.referenceVideoUrls),
      refAudios: parseUrlArray(record.referenceAudioUrls),
      duration: record.duration,
      resolution: normalizeResolution(record.resolution),
      aspectRatio: (record.aspectRatio || '').trim().toLowerCase(),
      audio: toBool(record.generateAudio, true),
      seed: record.seed,
      promptExtend: record.promptExtend == null ? undefined : toBool(record.promptExtend, true),
      watermark: record.watermark == null ? undefined : toBool(record.watermark, false),
      rawModel,
    }
    if (!ctx.prompt) throw new Error('kie.ai 视频生成要求必须提供提示词')

    // 兜底：dataURL 未经 prepareRecord 上传会被 kie.ai 拒绝，提前给出明确错误
    const allUrls = [ctx.firstFrame, ctx.lastFrame, ...ctx.refImages]
    if (allUrls.some((u) => u.startsWith('data:'))) {
      throw new Error('kie.ai 不支持 base64 参考图，请确认 prepareRecord 已执行上传')
    }

    const { model, input } = resolveBuilder(rawModel)(ctx)
    return buildKieCreateTaskRequest(config, model, input)
  }

  parseGenerateResponse(result: any): VideoGenResponse {
    const { taskId } = parseKieCreateResponse(result)
    return { isAsync: true, taskId }
  }

  buildPollRequest(config: AIConfig, taskId: string): ProviderRequest {
    return buildKiePollRequest(config, taskId)
  }

  parsePollResponse(result: any): VideoPollResponse {
    const r = parseKieRecord(result)
    if (r.status === 'completed') {
      const videoUrl = r.resultUrls[0]
      if (!videoUrl) return { status: 'failed', error: 'kie.ai 任务成功但未返回视频地址' }
      return { status: 'completed', videoUrl }
    }
    if (r.status === 'failed') return { status: 'failed', error: r.error }
    return { status: r.status }
  }

  extractVideoUrl(result: any): string | null {
    return extractKieResultUrls(result)[0] || null
  }
}
