/**
 * kie.ai 通用工具（图片/视频 Adapter 共用）
 * 统一任务接口:
 *   POST {base}/api/v1/jobs/createTask   { model, input } -> { code, msg, data: { taskId } }
 *   GET  {base}/api/v1/jobs/recordInfo?taskId=  -> { code, data: { state, resultJson, failCode, failMsg } }
 *   state: waiting | queuing | generating | success | fail
 * 注意: kie.ai 业务错误常以 HTTP 200 + code != 200 返回，必须检查 code。
 * 文件上传: POST {base}/api/file-base64-upload -> { data: { downloadUrl } }（3 天后过期）
 */
import type { AIConfig, ProviderRequest } from './types'
import { joinProviderUrl } from './url'
import { logTaskWarn } from '../../utils/task-logger.js'

export const KIE_DEFAULT_BASE_URL = 'https://api.kie.ai'
/** 部分文档/账号的上传服务在独立域名，主域名失败时回退 */
const KIE_UPLOAD_FALLBACK_BASE_URL = 'https://kieai.redpandaai.co'

function baseOf(config: AIConfig) {
  return (config.baseUrl || '').trim() || KIE_DEFAULT_BASE_URL
}

export function kieHeaders(config: AIConfig, withJson = false): Record<string, string> {
  const headers: Record<string, string> = { 'Authorization': `Bearer ${config.apiKey}` }
  if (withJson) headers['Content-Type'] = 'application/json'
  return headers
}

export function buildKieCreateTaskRequest(config: AIConfig, model: string, input: Record<string, any>): ProviderRequest {
  return {
    url: joinProviderUrl(baseOf(config), '', '/api/v1/jobs/createTask'),
    method: 'POST',
    headers: kieHeaders(config, true),
    body: { model, input },
  }
}

export function buildKiePollRequest(config: AIConfig, taskId: string): ProviderRequest {
  const url = new URL(joinProviderUrl(baseOf(config), '', '/api/v1/jobs/recordInfo'))
  url.searchParams.set('taskId', taskId)
  return { url: url.toString(), method: 'GET', headers: kieHeaders(config), body: undefined }
}

/** 业务 code 非 200 时抛错（HTTP 层 200 也可能是失败） */
export function assertKieOk(result: any) {
  if (result && typeof result.code === 'number' && result.code !== 200) {
    throw new Error(`kie.ai error [${result.code}]: ${result.msg || JSON.stringify(result)}`)
  }
}

export function parseKieCreateResponse(result: any): { taskId: string } {
  assertKieOk(result)
  const taskId = result?.data?.taskId
  if (!taskId) throw new Error(`kie.ai createTask 未返回 taskId: ${JSON.stringify(result)}`)
  return { taskId: String(taskId) }
}

export interface KiePollResult {
  status: 'pending' | 'processing' | 'completed' | 'failed'
  resultUrls: string[]
  error?: string
}

export function parseKieRecord(result: any): KiePollResult {
  assertKieOk(result)
  const data = result?.data || {}
  const state = String(data.state || '').toLowerCase()
  if (state === 'success') {
    return { status: 'completed', resultUrls: extractKieResultUrls(result) }
  }
  if (state === 'fail') {
    const code = data.failCode ? `[${data.failCode}] ` : ''
    return { status: 'failed', resultUrls: [], error: `${code}${data.failMsg || 'kie.ai task failed'}` }
  }
  if (state === 'waiting' || state === 'queuing') return { status: 'pending', resultUrls: [] }
  return { status: 'processing', resultUrls: [] }
}

/** resultJson 是 JSON 字符串: { resultUrls: [...] } */
export function extractKieResultUrls(result: any): string[] {
  const raw = result?.data?.resultJson
  if (!raw) return []
  try {
    const parsed = typeof raw === 'string' ? JSON.parse(raw) : raw
    const urls = parsed?.resultUrls || parsed?.resultUrl || []
    return (Array.isArray(urls) ? urls : [urls]).filter((u: any) => typeof u === 'string' && u)
  } catch {
    return []
  }
}

/**
 * 把 dataURL 上传到 kie.ai 临时存储，返回公网 URL；http(s) URL 原样返回。
 * kie.ai 的 market 模型只接受可访问的 URL，不接受 base64。
 */
export async function uploadDataUrlToKie(config: AIConfig, value: string, uploadPath = 'huobao/refs'): Promise<string> {
  if (!value.startsWith('data:')) return value
  const mime = /^data:([^;]+);/.exec(value)?.[1] || 'image/jpeg'
  const ext = (mime.split('/')[1] || 'jpg').replace('jpeg', 'jpg')
  const body = JSON.stringify({ base64Data: value, uploadPath, fileName: `ref-${Date.now()}-${Math.random().toString(36).slice(2, 8)}.${ext}` })

  const bases = [baseOf(config), KIE_UPLOAD_FALLBACK_BASE_URL]
  let lastErr = ''
  for (const base of bases) {
    try {
      const resp = await fetch(joinProviderUrl(base, '', '/api/file-base64-upload'), {
        method: 'POST',
        headers: kieHeaders(config, true),
        body,
        signal: AbortSignal.timeout(120_000),
      })
      const text = await resp.text()
      if (!resp.ok) { lastErr = `HTTP ${resp.status}: ${text.slice(0, 300)}`; continue }
      const json = JSON.parse(text)
      const url = json?.data?.downloadUrl || json?.data?.fileUrl
      if (json?.success === false || !url) { lastErr = text.slice(0, 300); continue }
      return url
    } catch (err) {
      lastErr = (err as Error).message
    }
    logTaskWarn('KieUpload', 'upload-failed', { base, error: lastErr })
  }
  throw new Error(`kie.ai 参考素材上传失败: ${lastErr}`)
}

export function parseUrlArray(raw?: string | null): string[] {
  if (!raw) return []
  try {
    const arr = JSON.parse(raw)
    return Array.isArray(arr) ? arr.filter((u) => typeof u === 'string' && u.trim()) : []
  } catch {
    return []
  }
}

export async function uploadAllToKie(config: AIConfig, values: string[]): Promise<string[]> {
  return Promise.all(values.map((v) => uploadDataUrlToKie(config, v)))
}
