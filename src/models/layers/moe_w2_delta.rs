//! FP4 delta pool + confidence gate for W2 MoE experts.
//!
//! Keeps a bounded GPU pool of hot experts at higher precision; when router
//! confidence is low (`max_prob <= tau`), callers can request a re-forward
//! with promoted experts. Persistence of FP4 planes is left to the MoE
//! loader; this module tracks residency / promote / gate policy only.

use std::collections::HashMap;
use std::sync::Mutex;

/// Default confidence threshold (Moet: 0.60).
pub const DEFAULT_GATE_TAU: f32 = 0.60;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeltaPolicy {
    /// Promote by router frequency.
    Freq,
    /// LRU among cached slots.
    Lru,
    /// Only promote when confidence gate fires.
    Need,
}

impl DeltaPolicy {
    pub fn from_env() -> Self {
        match std::env::var("XINFER_MOE_W2_DELTA_POLICY")
            .unwrap_or_else(|_| "need".into())
            .to_lowercase()
            .as_str()
        {
            "freq" => Self::Freq,
            "lru" => Self::Lru,
            _ => Self::Need,
        }
    }
}

#[derive(Debug, Clone)]
struct Slot {
    layer: usize,
    expert: usize,
    hits: u64,
    last_tick: u64,
}

/// Process-wide delta / gate manager (one per engine recommended; global for now).
pub struct MoeW2DeltaManager {
    capacity_slots: usize,
    policy: DeltaPolicy,
    gate_tau: f32,
    gate_enabled: bool,
    tick: u64,
    /// (layer, expert) → slot index
    map: HashMap<(usize, usize), usize>,
    slots: Vec<Option<Slot>>,
}

impl MoeW2DeltaManager {
    pub fn from_env() -> Self {
        let gb: f64 = std::env::var("XINFER_MOE_W2_DELTA_GB")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.0);
        // ~12 MiB per FP4 expert slot (Moet w13+w2) → slots ≈ GB * 1024 / 12
        let capacity_slots = if gb <= 0.0 {
            0
        } else {
            ((gb * 1024.0) / 12.0).floor().max(1.0) as usize
        };
        let gate_enabled = std::env::var("XINFER_MOE_W2_GATE")
            .map(|v| v != "0" && !v.eq_ignore_ascii_case("false"))
            .unwrap_or(false);
        let gate_tau = std::env::var("XINFER_MOE_W2_GATE_TAU")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(DEFAULT_GATE_TAU);
        Self {
            capacity_slots,
            policy: DeltaPolicy::from_env(),
            gate_tau,
            gate_enabled,
            tick: 0,
            map: HashMap::new(),
            slots: vec![None; capacity_slots],
        }
    }

    pub fn capacity(&self) -> usize {
        self.capacity_slots
    }

    pub fn gate_enabled(&self) -> bool {
        self.gate_enabled
    }

    pub fn should_replay(&self, max_router_prob: f32) -> bool {
        self.gate_enabled && max_router_prob <= self.gate_tau
    }

    pub fn is_promoted(&self, layer: usize, expert: usize) -> bool {
        self.map.contains_key(&(layer, expert))
    }

    /// Record a routing hit; may promote under Freq/Lru policies.
    pub fn observe(&mut self, layer: usize, experts: &[usize]) {
        self.tick = self.tick.wrapping_add(1);
        if self.capacity_slots == 0 {
            return;
        }
        for &expert in experts {
            if let Some(&idx) = self.map.get(&(layer, expert)) {
                if let Some(slot) = self.slots[idx].as_mut() {
                    slot.hits = slot.hits.wrapping_add(1);
                    slot.last_tick = self.tick;
                }
                continue;
            }
            if self.policy == DeltaPolicy::Need {
                continue;
            }
            self.promote(layer, expert);
        }
    }

    /// Force-promote experts for a gate replay.
    pub fn force_promote(&mut self, layer: usize, experts: &[usize]) {
        for &e in experts {
            self.promote(layer, e);
        }
    }

    fn promote(&mut self, layer: usize, expert: usize) {
        if self.capacity_slots == 0 || self.map.contains_key(&(layer, expert)) {
            return;
        }
        let idx = if let Some(free) = self.slots.iter().position(|s| s.is_none()) {
            free
        } else {
            // Evict
            let victim = match self.policy {
                DeltaPolicy::Lru | DeltaPolicy::Need => self
                    .slots
                    .iter()
                    .enumerate()
                    .filter_map(|(i, s)| s.as_ref().map(|s| (i, s.last_tick)))
                    .min_by_key(|(_, t)| *t)
                    .map(|(i, _)| i),
                DeltaPolicy::Freq => self
                    .slots
                    .iter()
                    .enumerate()
                    .filter_map(|(i, s)| s.as_ref().map(|s| (i, s.hits)))
                    .min_by_key(|(_, h)| *h)
                    .map(|(i, _)| i),
            };
            let Some(v) = victim else { return };
            if let Some(old) = self.slots[v].take() {
                self.map.remove(&(old.layer, old.expert));
            }
            v
        };
        self.slots[idx] = Some(Slot {
            layer,
            expert,
            hits: 1,
            last_tick: self.tick,
        });
        self.map.insert((layer, expert), idx);
    }
}

static GLOBAL: Mutex<Option<MoeW2DeltaManager>> = Mutex::new(None);

pub fn global_delta_manager() -> std::sync::MutexGuard<'static, Option<MoeW2DeltaManager>> {
    let mut g = GLOBAL.lock().unwrap();
    if g.is_none() {
        *g = Some(MoeW2DeltaManager::from_env());
    }
    g
}
