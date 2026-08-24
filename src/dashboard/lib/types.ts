export interface EngineStatus {
  loaded: boolean;
  vram_allocated_gb: number;
  total_vram_used_gb?: number;
  has_residual_vram?: boolean;
  residual_vram_gb?: number;
  active_team: string[];
  model_id?: string;
  spec_decoder_active?: boolean;
  spec_decode_enabled?: boolean;
  ring_buffer_mode?: string;
  ring_buffer_wired?: boolean;
  ring_buffer_options?: string[];
  spec_k?: number;
  spec_k_options?: number[];
  scale_mode?: string;
  prefold_enabled?: boolean;
  state_handoff_enabled?: boolean;
  state_handoff_mb?: number;
  w4a16_enabled?: boolean;
  intra_team_dr?: number;
  last_morph_ms?: number;
}

export interface CausalEdge {
  source: string;
  target: string;
  weight: number;
}

export interface CausalDagResponse {
  nodes: string[];
  edges: CausalEdge[];
  fitted: boolean;
  hit_rate: number;
  telemetry?: {
    prefold_triggers?: number;
    prefold_hits?: number;
    prefold_misses?: number;
    prefold_saved_ms?: number;
    wasted_morph_ms?: number;
    last_prediction?: string;
    last_confidence?: number;
  };
}

export interface TrainingRun {
  run_id: string;
  domain: string;
  adapter_version?: string;
  adapter_size_mb?: number;
  created_at?: string;
  stopped_at_step?: number;
  max_steps?: number;
  final_loss?: number;
  final_dw_w?: number;
  predicted_merge_err?: number;
  airm_scale?: number;
  gate_passed?: boolean;
  r?: number;
  alpha?: number;
}

export interface TrainingStatus {
  active: boolean;
  domain?: string;
  run_id?: string;
  start_time?: number;
  recent_logs?: string[];
  error?: string;
}

export interface CalibrateAlphaResponse {
  domain: string;
  run_id: string;
  adapter_name: string;
  trained_dw_over_w: number;
  alpha_opt: number;
  alpha_min: number;
  optimal_point: {
    alpha: number;
    scaling: number;
    dw_over_w: number;
    merge_err_pct: number;
    in_target_band: boolean;
    below_floor: boolean;
  };
  curve: Array<{
    alpha: number;
    scaling: number;
    dw_over_w: number;
    merge_err_pct: number;
    in_target_band: boolean;
    below_floor: boolean;
  }>;
}
