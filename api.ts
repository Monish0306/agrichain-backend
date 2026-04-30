// src/services/api.ts
// All API calls to AgriChain FastAPI backend

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

// ─── Helpers ───────────────────────────────────────────────

export function getToken(): string | null {
  if (typeof window === 'undefined') return null
  return localStorage.getItem('agrichain_token')
}

export function getRole(): string | null {
  if (typeof window === 'undefined') return null
  return localStorage.getItem('agrichain_role')
}

function authHeaders(): Record<string, string> {
  const token = getToken()
  return {
    'Content-Type': 'application/json',
    ...(token ? { Authorization: `Bearer ${token}` } : {}),
  }
}

async function apiCall<T = any>(path: string, options: RequestInit = {}): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      ...(options.headers || {}),
    },
  })
  if (!res.ok) {
    let detail = `HTTP ${res.status}`
    try {
      const err = await res.json()
      detail = err.detail || JSON.stringify(err)
    } catch {}
    throw new Error(detail)
  }
  return res.json()
}

// ─── AUTH ───────────────────────────────────────────────────

export async function farmerLogin(phone: string, name: string) {
  const data = await apiCall('/api/auth/farmer/login', {
    method: 'POST',
    body: JSON.stringify({ phone, name }),
  })
  if (data.access_token) {
    localStorage.setItem('agrichain_token', data.access_token)
    localStorage.setItem('agrichain_role', 'farmer')
    localStorage.setItem('agrichain_user', JSON.stringify(data))
  }
  return data
}

export async function merchantLogin(email: string, password: string, name: string) {
  const data = await apiCall('/api/auth/merchant/login', {
    method: 'POST',
    body: JSON.stringify({ email, password, name }),
  })
  if (data.access_token) {
    localStorage.setItem('agrichain_token', data.access_token)
    localStorage.setItem('agrichain_role', 'merchant')
    localStorage.setItem('agrichain_user', JSON.stringify(data))
  }
  return data
}

export async function monitorLogin(username: string, password: string) {
  const data = await apiCall('/api/auth/monitor/login', {
    method: 'POST',
    body: JSON.stringify({ username, password }),
  })
  if (data.access_token) {
    localStorage.setItem('agrichain_token', data.access_token)
    localStorage.setItem('agrichain_role', 'monitor')
    localStorage.setItem('agrichain_user', JSON.stringify(data))
  }
  return data
}

export async function getCurrentUser() {
  return apiCall('/api/auth/me', { headers: authHeaders() })
}

export function logout() {
  localStorage.removeItem('agrichain_token')
  localStorage.removeItem('agrichain_role')
  localStorage.removeItem('agrichain_user')
}

// ─── ADVISORY ───────────────────────────────────────────────

export interface AdvisoryInput {
  nitrogen: number
  phosphorous: number
  potassium: number
  temperature: number
  humidity: number
  ph: number
  rainfall: number
  soil_type: string
  crop_type?: string
  gps_lat?: number
  gps_lon?: number
  language?: string
}

export async function getCropRecommendation(data: AdvisoryInput) {
  return apiCall('/api/advisory/recommend', {
    method: 'POST',
    headers: authHeaders(),
    body: JSON.stringify(data),
  })
}

export async function getCropList() {
  return apiCall('/api/advisory/crops')
}

export async function getSoilTypes() {
  return apiCall('/api/advisory/soil-types')
}

export async function getCropTypesForFertilizer() {
  return apiCall('/api/advisory/crop-types-for-fertilizer')
}

// ─── MARKETPLACE ────────────────────────────────────────────

export interface ListingFilters {
  crop_type?: string
  district?: string
  state?: string
}

export async function getListings(filters?: ListingFilters) {
  const params = new URLSearchParams()
  if (filters?.crop_type) params.set('crop_type', filters.crop_type)
  if (filters?.district) params.set('district', filters.district)
  if (filters?.state) params.set('state', filters.state)
  const query = params.toString() ? `?${params.toString()}` : ''
  return apiCall(`/api/marketplace/listings${query}`)
}

export async function getListing(listing_id: string) {
  return apiCall(`/api/marketplace/listings/${listing_id}`)
}

export interface CreateListingData {
  crop_type: string
  quantity_kg: number
  asking_price: number
  quality_grade: string
  description?: string
  district: string
  state: string
  location_lat?: number
  location_lon?: number
}

export async function createListing(data: CreateListingData) {
  return apiCall('/api/marketplace/listings', {
    method: 'POST',
    headers: authHeaders(),
    body: JSON.stringify(data),
  })
}

export async function placeOrder(listing_id: string, quantity_kg: number, offered_price: number) {
  return apiCall(`/api/marketplace/listings/${listing_id}/order`, {
    method: 'POST',
    headers: authHeaders(),
    body: JSON.stringify({ quantity_kg, offered_price }),
  })
}

export async function confirmTransaction(transaction_id: string) {
  return apiCall(`/api/marketplace/transactions/${transaction_id}/confirm`, {
    method: 'POST',
    headers: authHeaders(),
  })
}

export async function getMyListings() {
  return apiCall('/api/marketplace/my-listings', { headers: authHeaders() })
}

export async function getMyOrders() {
  return apiCall('/api/marketplace/my-orders', { headers: authHeaders() })
}

export async function getRoute(
  origin_lat: number,
  origin_lon: number,
  dest_lat: number,
  dest_lon: number
) {
  return apiCall(
    `/api/marketplace/route?origin_lat=${origin_lat}&origin_lon=${origin_lon}&dest_lat=${dest_lat}&dest_lon=${dest_lon}`
  )
}

// ─── PRICES ─────────────────────────────────────────────────

export async function getCommodities() {
  return apiCall('/api/prices/commodities')
}

export async function getPricePrediction(commodity: string, days: number = 7) {
  return apiCall(`/api/prices/predict/${encodeURIComponent(commodity)}?days=${days}`)
}

export async function getCurrentPrice(commodity: string) {
  return apiCall(`/api/prices/current/${encodeURIComponent(commodity)}`)
}

// ─── FINANCE ────────────────────────────────────────────────

export interface FinanceInput {
  crop_type: string
  land_acres: number
  state: string
  category?: string
}

export async function calculateFinance(data: FinanceInput) {
  return apiCall('/api/finance/calculate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data),
  })
}

export async function getSchemes(state?: string, crop_type?: string, category?: string) {
  const params = new URLSearchParams()
  if (state) params.set('state', state)
  if (crop_type) params.set('crop_type', crop_type)
  if (category) params.set('category', category)
  const query = params.toString() ? `?${params.toString()}` : ''
  return apiCall(`/api/finance/schemes${query}`)
}

export async function calculateEMI(principal: number, annual_rate_percent: number, months: number) {
  return apiCall(
    `/api/finance/emi?principal=${principal}&annual_rate_percent=${annual_rate_percent}&months=${months}`
  )
}

// ─── MONITOR ────────────────────────────────────────────────

export async function getMonitorOverview() {
  return apiCall('/api/monitor/overview', { headers: authHeaders() })
}

export async function getMonitorTransactions(status_filter?: string) {
  const query = status_filter ? `?status_filter=${status_filter}` : ''
  return apiCall(`/api/monitor/transactions${query}`, { headers: authHeaders() })
}

export async function getMonitorListings() {
  return apiCall('/api/monitor/listings', { headers: authHeaders() })
}