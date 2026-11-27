use crate::error::ErrorExt;
use crate::automata::lexer::Lexer;
use std::collections::BTreeMap;

use range_set_blaze::RangeSetBlaze;

use crate::compiler::glr::accumulator::TerminalsDisallowed;
use crate::compiler::glr::parser::{ParserGSS, close_control_stacks};
use crate::grammar::flat::TerminalID;

use super::artifact::Constraint;


/// Scanner-state coordinate used by parser weights and possible matches.
/// A compiler partition can derive its inverse exactly; composed observations
/// may instead use an inverse over a different scope, which must be retained.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct TokenizerTsidMap {
    primary: Vec<u32>,
    groups: TsidGroups,
    #[serde(skip)]
    rows: TsidRows,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
enum TsidGroups {
    Partition {
        count: usize,
        #[serde(skip)]
        inverse: std::sync::OnceLock<Vec<Vec<u32>>>,
    },
    Explicit(Vec<Vec<u32>>),
}

#[derive(Debug, Clone, Default)]
enum TsidRows {
    #[default]
    Unbuilt,
    Primary,
    Csr {
        offsets: Vec<u32>,
        values: Vec<u32>,
    },
}

impl Default for TokenizerTsidMap {
    fn default() -> Self {
        Self::partition(Vec::new(), Vec::new())
    }
}

impl TokenizerTsidMap {
    /// A ManyToOneIdMap already contains the exact partition. Move its inverse
    /// into the cache rather than reconstructing it during compiler linking.
    pub(crate) fn partition(primary: Vec<u32>, inverse: Vec<Vec<u32>>) -> Self {
        let count = inverse.len();
        debug_assert!(
            inverse.iter().enumerate().all(|(id, states)| {
                states.windows(2).all(|pair| pair[0] < pair[1])
                    && states.iter().all(|&state| {
                        primary.get(state as usize).is_some_and(|&mapped| {
                            mapped != u32::MAX && mapped as usize == id
                        })
                    })
            }) && inverse.iter().map(Vec::len).sum::<usize>()
                == primary.iter().filter(|&&id| id != u32::MAX).count(),
            "compiler TSID partition must have its canonical exact inverse",
        );
        let groups = if count <= primary.len() {
            TsidGroups::Partition {
                count,
                inverse: std::sync::OnceLock::from(inverse),
            }
        } else {
            // Preserve even unused class IDs in an explicitly supplied domain.
            TsidGroups::Explicit(inverse)
        };
        Self {
            primary,
            groups,
            rows: TsidRows::Primary,
        }
    }

    pub(crate) fn identity(count: usize) -> Self {
        Self {
            primary: (0..count as u32).collect(),
            groups: TsidGroups::Partition {
                count,
                inverse: Default::default(),
            },
            rows: TsidRows::Primary,
        }
    }

    pub(crate) fn scoped(primary: Vec<u32>, inverse: Vec<Vec<u32>>) -> Self {
        Self {
            primary,
            groups: TsidGroups::Explicit(inverse),
            rows: TsidRows::Primary,
        }
    }

    pub(crate) fn with_relation(
        primary: Vec<u32>,
        inverse: Vec<Vec<u32>>,
        offsets: Vec<u32>,
        values: Vec<u32>,
    ) -> Self {
        Self {
            primary,
            groups: TsidGroups::Explicit(inverse),
            rows: TsidRows::Csr { offsets, values },
        }
    }

    pub(crate) fn primary(&self) -> &[u32] {
        &self.primary
    }

    pub(crate) fn is_unmapped(&self) -> bool {
        self.primary.is_empty() && self.count() == 0
    }

    pub(crate) fn is_singleton(&self) -> bool {
        matches!(self.rows, TsidRows::Primary)
            || matches!(self.groups, TsidGroups::Partition { .. })
    }

    pub(crate) fn has_csr(&self) -> bool {
        matches!(self.rows, TsidRows::Csr { .. })
    }

    pub(crate) fn is_partition(&self) -> bool {
        matches!(self.groups, TsidGroups::Partition { .. })
    }

    pub(crate) fn count(&self) -> usize {
        match &self.groups {
            TsidGroups::Partition { count, .. } => *count,
            TsidGroups::Explicit(inverse) => inverse.len(),
        }
    }

    pub(crate) fn groups(&self) -> &[Vec<u32>] {
        match &self.groups {
            TsidGroups::Explicit(inverse) => inverse,
            TsidGroups::Partition { count, inverse } => inverse.get_or_init(|| {
                let mut groups = vec![Vec::new(); *count];
                for (state, &id) in self.primary.iter().enumerate() {
                    if let Some(group) = groups.get_mut(id as usize) {
                        group.push(state as u32);
                    }
                }
                groups
            }),
        }
    }

    /// Validate the compact inverse domain before any lazy allocation on load.
    pub(crate) fn validate(&self) -> Result<(), String> {
        if let TsidGroups::Partition { count, .. } = &self.groups {
            if *count > self.primary.len()
                || self
                    .primary
                    .iter()
                    .any(|&id| id != u32::MAX && id as usize >= *count)
            {
                return Err("invalid tokenizer TSID partition".to_owned());
            }
        }
        Ok(())
    }

    #[inline]
    pub(crate) fn row(&self, state: usize) -> &[u32] {
        if let TsidRows::Csr { offsets, values } = &self.rows
            && let (Some(&start), Some(&end)) = (offsets.get(state), offsets.get(state + 1))
            && let Some(row) = values.get(start as usize..end as usize)
        {
            return row;
        }
        self.primary
            .get(state)
            .map(std::slice::from_ref)
            .unwrap_or(&[])
    }

    /// Only scoped/non-functional maps need to reconstruct a CSR. A partition
    /// always has precisely its primary row, including unmapped sentinels.
    pub(crate) fn prepare_rows(&mut self, expected_count: usize, raw_count: usize) {
        if self.is_partition() {
            self.rows = TsidRows::Primary;
            return;
        }
        let ready = match &self.rows {
            TsidRows::Primary => self.primary.len() == expected_count,
            TsidRows::Unbuilt => {
                self.primary.len() == expected_count
                    && self.primary.iter().all(|&id| id != u32::MAX)
            }
            TsidRows::Csr { offsets, values } => {
                offsets.len() == expected_count + 1
                    && offsets
                        .last()
                        .is_some_and(|&end| end as usize == values.len())
            }
        };
        if ready {
            // A loaded scoped inverse is not certified as a local partition
            // merely because its primary row covers the current tokenizer.
            return;
        }
        let mut relation = vec![smallvec::SmallVec::<[u32; 2]>::new(); raw_count];
        for (id, states) in self.groups().iter().enumerate() {
            for &state in states {
                if let Some(row) = relation.get_mut(state as usize) {
                    row.push(id as u32);
                }
            }
        }
        let mut offsets = Vec::with_capacity(raw_count + 1);
        let mut values = Vec::new();
        offsets.push(0);
        for (state, mut row) in relation.into_iter().enumerate() {
            if let Some(&id) = self.primary.get(state) {
                row.push(id);
            }
            row.sort_unstable();
            row.dedup();
            values.extend_from_slice(&row);
            offsets.push(values.len() as u32);
        }
        self.rows = TsidRows::Csr { offsets, values };
    }
}

/// The final token coordinate shared by parser weights and mask fragments.
/// Identity has no quotient; empty explicit quotients remain distinct from it.
#[derive(Debug, Clone, Default)]
pub(crate) struct TokenClasses(TokenClassStorage);

#[derive(Debug, Clone, Default)]
enum TokenClassStorage {
    #[default]
    Identity,
    Owned {
        forward: Vec<u32>,
        inverse: Vec<Vec<u32>>,
    },
    Packed {
        map: PackedTokenMap,
        expanded: std::sync::OnceLock<Vec<u32>>,
        inverse: std::sync::OnceLock<Vec<Vec<u32>>>,
    },
}

#[derive(Debug, Clone)]
struct PackedTokenMap {
    backing: std::sync::Arc<Vec<u8>>,
    start: usize,
    count: usize,
    class_count: usize,
    width: usize,
}

const TOKEN_MAP_MAGIC: &[u8; 4] = b"OTM3";
const TOKEN_MAP_HEADER: usize = 13;

impl PackedTokenMap {
    fn bytes(&self) -> &[u8] {
        &self.backing[self.start..self.start + TOKEN_MAP_HEADER + self.count * self.width]
    }

    #[inline]
    fn get(&self, index: usize) -> Option<u32> {
        if index >= self.count {
            return None;
        }
        let start = self.start + TOKEN_MAP_HEADER + index * self.width;
        let bytes = &self.backing[start..start + self.width];
        Some(match self.width {
            1 => {
                if bytes[0] == u8::MAX {
                    u32::MAX
                } else {
                    u32::from(bytes[0])
                }
            }
            2 => {
                let id = u16::from_le_bytes([bytes[0], bytes[1]]);
                if id == u16::MAX {
                    u32::MAX
                } else {
                    u32::from(id)
                }
            }
            4 => u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
            _ => unreachable!(),
        })
    }
}

impl TokenClasses {
    pub(crate) fn mapped(forward: Vec<u32>, inverse: Vec<Vec<u32>>) -> Self {
        debug_assert!(inverse.len() <= forward.len());
        debug_assert!(
            forward
                .iter()
                .all(|&id| id == u32::MAX || (id as usize) < inverse.len())
        );
        Self(TokenClassStorage::Owned { forward, inverse })
    }

    #[inline]
    fn count(&self) -> Option<usize> {
        match &self.0 {
            TokenClassStorage::Identity => None,
            TokenClassStorage::Owned { inverse, .. } => Some(inverse.len()),
            TokenClassStorage::Packed { map, .. } => Some(map.class_count),
        }
    }

    fn forward(&self) -> &[u32] {
        match &self.0 {
            TokenClassStorage::Identity => &[],
            TokenClassStorage::Owned { forward, .. } => forward,
            TokenClassStorage::Packed { map, expanded, .. } => expanded.get_or_init(|| {
                (0..map.count)
                    .map(|index| map.get(index).unwrap())
                    .collect()
            }),
        }
    }

    #[inline]
    fn get(&self, index: usize) -> Option<u32> {
        match &self.0 {
            TokenClassStorage::Identity => None,
            TokenClassStorage::Owned { forward, .. } => forward.get(index).copied(),
            TokenClassStorage::Packed { map, .. } => map.get(index),
        }
    }

    pub(crate) fn cached_groups(&self) -> Option<&[Vec<u32>]> {
        match &self.0 {
            TokenClassStorage::Identity => None,
            TokenClassStorage::Owned { inverse, .. } => Some(inverse),
            TokenClassStorage::Packed { inverse, .. } => inverse.get().map(Vec::as_slice),
        }
    }

    fn groups(&self) -> Option<&[Vec<u32>]> {
        match &self.0 {
            TokenClassStorage::Identity => None,
            TokenClassStorage::Owned { inverse, .. } => Some(inverse),
            TokenClassStorage::Packed { map, inverse, .. } => Some(inverse.get_or_init(|| {
                let forward = self.forward();
                let mut counts = vec![0usize; map.class_count];
                for &id in forward {
                    if id != u32::MAX {
                        counts[id as usize] += 1;
                    }
                }
                let mut groups = counts
                    .into_iter()
                    .map(Vec::<u32>::with_capacity)
                    .collect::<Vec<_>>();
                for (original, &id) in forward.iter().enumerate() {
                    if id != u32::MAX {
                        groups[id as usize].push(original as u32);
                    }
                }
                groups
            })),
        }
    }

    pub(crate) fn encode(&self) -> Vec<u8> {
        let (forward, class_count) = match &self.0 {
            TokenClassStorage::Identity => return Vec::new(),
            TokenClassStorage::Packed { map, .. } => return map.bytes().to_vec(),
            TokenClassStorage::Owned { forward, inverse } => (forward, inverse.len()),
        };
        let max_id = forward
            .iter()
            .copied()
            .filter(|&id| id != u32::MAX)
            .max()
            .unwrap_or(0);
        let width = if max_id < u8::MAX as u32 {
            1
        } else if max_id < u16::MAX as u32 {
            2
        } else {
            4
        };
        let mut bytes = Vec::with_capacity(TOKEN_MAP_HEADER + forward.len() * width);
        bytes.extend_from_slice(TOKEN_MAP_MAGIC);
        bytes.push(width as u8);
        bytes.extend_from_slice(
            &u32::try_from(forward.len())
                .expect("token extent fits u32")
                .to_le_bytes(),
        );
        bytes.extend_from_slice(
            &u32::try_from(class_count)
                .expect("class count fits u32")
                .to_le_bytes(),
        );
        for &id in forward {
            match width {
                1 => bytes.push(id as u8),
                2 => bytes.extend_from_slice(&(id as u16).to_le_bytes()),
                4 => bytes.extend_from_slice(&id.to_le_bytes()),
                _ => unreachable!(),
            }
        }
        bytes
    }

    pub(crate) fn decode_backed(
        backing: std::sync::Arc<Vec<u8>>,
        start: usize,
        len: usize,
    ) -> Result<Self, String> {
        let end = start
            .checked_add(len)
            .ok_or("token class section range overflow")?;
        let input = backing
            .get(start..end)
            .ok_or("token class section outside artifact")?;
        if input.is_empty() {
            return Ok(Self::default());
        }
        if input.len() < TOKEN_MAP_HEADER || !input.starts_with(TOKEN_MAP_MAGIC) {
            return Err("invalid token class map header".into());
        }
        let width = input[4] as usize;
        let count = u32::from_le_bytes(input[5..9].try_into().unwrap()) as usize;
        let class_count = u32::from_le_bytes(input[9..13].try_into().unwrap()) as usize;
        if !matches!(width, 1 | 2 | 4)
            || count
                .checked_mul(width)
                .and_then(|n| n.checked_add(TOKEN_MAP_HEADER))
                != Some(len)
        {
            return Err("invalid token class map length or width".into());
        }
        // A quotient can retain empty classes when sentinels are hidden.
        // Their former original IDs still occupy the forward extent.
        if class_count > count {
            return Err("token class cardinality exceeds original extent".into());
        }
        let map = PackedTokenMap {
            backing,
            start,
            count,
            class_count,
            width,
        };
        if (0..count).any(|index| {
            let id = map.get(index).unwrap();
            id != u32::MAX && id as usize >= class_count
        }) {
            return Err("token class ID outside class cardinality".into());
        }
        Ok(Self(TokenClassStorage::Packed {
            map,
            expanded: Default::default(),
            inverse: Default::default(),
        }))
    }
}

impl Constraint {
    /// Visit original-ID members of the final static class without expanding a
    /// loaded vocabulary map or producing a complete mask. The already-stored
    /// sparse output fragments are just another representation of the same class.
    pub(crate) fn visit_internal_token_members(
        &self,
        class: usize,
        visit: &mut impl FnMut(u32) -> crate::Result<bool>,
    ) -> crate::Result<bool> {
        if let Some(ids) = self
            .token_classes
            .cached_groups()
            .and_then(|v| v.get(class))
        {
            for &id in ids {
                if !visit(id)? {
                    return Ok(false);
                }
            }
            return Ok(true);
        }
        let mut word = |index: u16, mut bits: u32| -> crate::Result<bool> {
            while bits != 0 {
                let id = u32::from(index) * 32 + bits.trailing_zeros();
                bits &= bits - 1;
                if !visit(id)? {
                    return Ok(false);
                }
            }
            Ok(true)
        };
        if let Some(result) = self.token_projection.visit_row(class, &mut word) {
            return result;
        }
        // An artifact may not have restored its mask caches yet. The packed map
        // still defines the members exactly.
        if let Some(ids) = self
            .internal_token_groups()
            .and_then(|groups| groups.get(class))
        {
            for &id in ids {
                if !visit(id)? {
                    return Ok(false);
                }
            }
            return Ok(true);
        }
        Err(crate::Error::Internal(
            "bounded token proof: final class has no original-ID members".into(),
        ))
    }

    #[inline]
    pub(crate) fn has_original_token_map(&self) -> bool {
        self.token_classes.count().is_some()
    }

    pub(crate) fn original_token_map(&self) -> &[u32] {
        self.token_classes.forward()
    }

    #[inline]
    pub(crate) fn original_token_internal_at(&self, token_id: u32) -> Option<u32> {
        self.token_classes.get(token_id as usize)
    }

    #[inline]
    pub(crate) fn internal_token_count(&self) -> usize {
        self.token_classes
            .count()
            .unwrap_or_else(|| self.max_original_token_id().map_or(0, |id| id as usize + 1))
    }

    pub(crate) fn internal_token_groups(&self) -> Option<&[Vec<u32>]> {
        self.token_classes.groups()
    }

    pub(crate) fn runtime_commit_initial_state(&self) -> u32 {
		self.tokenizer.initial_state()
	}

	pub(crate) fn possible_matches_for_state(
		&self,
		tokenizer_state: u32,
	) -> BTreeMap<TerminalID, RangeSetBlaze<u32>> {
		self.runtime_possible_match_terminals()
			.filter_map(|terminal| {
				let weight = self.runtime_possible_match_weight(terminal)?;
				let mut tokens = RangeSetBlaze::new();
				for &internal_tsid in self.internal_tsids_for_state(tokenizer_state) {
					if let Some(token_set) = weight.token_set_for_tsid(internal_tsid) {
						tokens |= token_set.to_range_set();
					}
				}
				if tokens.is_empty() {
					None
				} else {
					Some((terminal, self.expand_internal_token_set(&tokens)))
				}
			})
			.collect()
	}

	#[inline]
	fn static_observation_state(&self, tokenizer_state: u32) -> u32 {
		if !self.uses_dynamic_runtime()
			&& self.tokenizer.has_virtual_residual_runtime()
			&& self.dynamic_mask_vocab.mask_projection_tokenizer().is_some()
		{
			self.dynamic_mask_vocab.mask_projection_state(tokenizer_state)
		} else {
			tokenizer_state
		}
	}

	pub(crate) fn internal_tsid_for_state(&self, tokenizer_state: u32) -> u32 {
		let tokenizer_state = self.static_observation_state(tokenizer_state);
		self.tokenizer_tsids.primary()
			.get(tokenizer_state as usize)
			.copied()
			.unwrap_or(tokenizer_state)
	}

	pub(crate) fn internal_tsids_for_state(&self, tokenizer_state: u32) -> &[u32] {
		self.tokenizer_tsids.row(self.static_observation_state(tokenizer_state) as usize)
	}

	pub(crate) fn internal_token_for_original(&self, token_id: u32) -> u32 {
		self.original_token_internal_at(token_id)
			.filter(|internal_id| *internal_id != u32::MAX)
			.unwrap_or(token_id)
	}

	pub(crate) fn final_internal_token_for_original(&self, token_id: u32) -> Option<u32> {
		let internal = self.original_token_internal_at(token_id)?;

		if internal == u32::MAX {
			return None;
		}

		if internal as usize >= self.internal_token_count() {
			return None;
		}

		Some(internal)
	}

	/// Visit the original vocabulary tokens that can complete `terminal` from
	/// `tokenizer_state`. Returns `false` only for legacy artifacts that omitted
	/// possible matches.
	///
	/// IMPORTANT: callers must not treat this as an acceptable construction
	/// strategy. The dynamic possible-matches fallback is terrible, is planned
	/// for removal, and MUST NOT be used by new compiler paths. DO NOT REMOVE OR
	/// WEAKEN THIS COMMENT.
	pub(crate) fn visit_possible_match_original_tokens(
		&self,
		tokenizer_state: u32,
		terminal: TerminalID,
		mut visit: impl FnMut(u32),
	) -> bool {
		if !self.possible_matches_complete {
			return false;
		}
		let Some(weight) = self.runtime_possible_match_weight(terminal) else {
			return true;
		};
		let mut internal_tokens = RangeSetBlaze::new();
		for &internal_tsid in self.internal_tsids_for_state(tokenizer_state) {
			if let Some(token_set) = weight.token_set_for_tsid(internal_tsid) {
				internal_tokens |= token_set.to_range_set();
			}
		}
		let Some(groups) = self.internal_token_groups() else {
			for token in internal_tokens.iter() {
				visit(token);
			}
			return true;
		};
		for internal_token in internal_tokens.iter() {
			if let Some(originals) = groups.get(internal_token as usize) {
				for &original in originals {
					visit(original);
				}
			}
		}
		true
	}

	/// Return `Some(true)` when the exact possible-match relation proves that
	/// no public/original vocabulary token can complete `terminal` from
	/// `tokenizer_state`. This deliberately accounts for internal token classes
	/// whose original-token group is empty (for example linker-only placeholder
	/// classes), so callers can skip an otherwise expensive full expansion.
	/// Legacy artifacts without complete possible matches return `None`.
	pub(crate) fn possible_match_original_tokens_definitely_empty(
		&self,
		tokenizer_state: u32,
		terminal: TerminalID,
	) -> Option<bool> {
		// Dynamic transfer intentionally omits the heavyweight possible-match
		// table. Before falling back to a vocabulary scan, prove the stronger
		// vocabulary-independent fact that this terminal cannot match after any
		// byte string whose length fits in one model token. This exact BFS only
		// handles ordinary scalar lexer states; epsilon/virtual paths conservatively
		// decline and retain the existing fallback.
		let ordinary_impossible = self.ordinary_terminal_match_impossible_within(
			tokenizer_state,
			terminal,
			self.max_token_byte_len(),
		);
		if crate::env::var_os("GLRMASK_PROFILE_DYNAMIC_MASK").is_some() {
			eprintln!(
				"[glrmask/profile][pending_guard_bounded_reachability] state={} terminal={} impossible={:?} epsilon={} virtual={} max_token_len={}",
				tokenizer_state,
				terminal,
				ordinary_impossible,
				self.tokenizer.state_has_epsilon_transitions(tokenizer_state),
				self.tokenizer.state_is_virtual_runtime(tokenizer_state),
				self.max_token_byte_len(),
			);
		}
		if let Some(impossible) = ordinary_impossible
			&& impossible {
				return Some(true);
			}
		if !self.possible_matches_complete {
			return None;
		}
		let Some(weight) = self.runtime_possible_match_weight(terminal) else {
			return Some(true);
		};
		let internal_count = self.internal_token_count();
        if internal_count == 0 { return Some(true); }
        if !self.token_projection.has_rows(internal_count) {
            return None;
        }		for &internal_tsid in self.internal_tsids_for_state(tokenizer_state) {
			let Some(tokens) = weight.token_set_for_tsid(internal_tsid) else {
				continue;
			};
			if tokens.is_empty() {
				continue;
			}
			let mut has_public = false;
			tokens.for_each_range(|start, end| {
				if has_public {
					return;
				}
				let start = start as usize;
				let end = (end as usize).min(internal_count.saturating_sub(1));
				if start > end {
					return;
				}
                has_public = (start..=end).any(|internal| self.token_projection.row_len(internal) != 0);			});
			if has_public {
				return Some(false);
			}
		}
		Some(true)
	}

	fn ordinary_terminal_match_impossible_within(
		&self,
		start: u32,
		terminal: TerminalID,
		horizon: usize,
	) -> Option<bool> {
		let tokenizer = &self.tokenizer;
		if horizon == 0 || start >= tokenizer.num_states() {
			return Some(true);
		}
		if tokenizer.state_has_epsilon_transitions(start)
			|| tokenizer.state_is_virtual_runtime(start)
		{
			return None;
		}
		if !tokenizer
			.possible_future_terminals(start)
			.contains(terminal as usize)
		{
			return Some(true);
		}

		let mut distance = vec![usize::MAX; tokenizer.num_states() as usize];
		let mut queue = std::collections::VecDeque::new();
		distance[start as usize] = 0;
		queue.push_back(start);
		let mut ambiguous = false;
		while let Some(state) = queue.pop_front() {
			let depth = distance[state as usize];
			if depth >= horizon {
				continue;
			}
			for (_, target) in tokenizer.transitions_from(state) {
				if tokenizer
					.matched_terminal_bitset(target)
					.contains(terminal as usize)
				{
					return Some(false);
				}
				if !tokenizer
					.possible_future_terminals(target)
					.contains(terminal as usize)
				{
					continue;
				}
				if tokenizer.state_has_epsilon_transitions(target)
					|| tokenizer.state_is_virtual_runtime(target)
				{
					ambiguous = true;
					continue;
				}
				let next_depth = depth + 1;
				if next_depth < distance[target as usize] {
					distance[target as usize] = next_depth;
					queue.push_back(target);
				}
			}
		}
		if ambiguous { None } else { Some(true) }
	}

	pub(crate) fn internal_token_universe(&self) -> RangeSetBlaze<u32> {
        let count = self.internal_token_count();
        if count == 0 {
            RangeSetBlaze::new()
        } else {
            RangeSetBlaze::from_iter(std::iter::once(0..=(count - 1) as u32))
        }
    }

    pub(crate) fn expand_internal_token_set(
		&self,
		internal_tokens: &RangeSetBlaze<u32>,
	) -> RangeSetBlaze<u32> {
		let Some(groups) = self.internal_token_groups() else {
			return internal_tokens.clone();
		};

		let all_ids = Self::collect_original_token_ids(groups, internal_tokens);
		Self::range_set_from_sorted_ids(&all_ids)
	}

	pub(crate) fn initial_state_map(&self) -> crate::runtime::state::ParserStateMap {
		let parser_gss = ParserGSS::from_stacks(&[(vec![0u32], TerminalsDisallowed::new())]);
		let parser_gss = if let Some(closed) = self.close_compact_segmented_parser(&parser_gss) {
			closed
		} else if !self.parser_has_controls() {
			parser_gss
		} else {
			close_control_stacks(&self.table, &parser_gss)
		};
		if self.uses_compact_segmented_parser_runtime() {
			let partitions = self
				.partition_recursive_parser_gss_by_active_leaf(&parser_gss)
				.expect("recursive initial parser state must partition by active leaf");
			let mut state = crate::runtime::state::ParserStateMap::default();
			for (leaf_index, partition) in partitions {
				let reset = self
					.recursive_tokenizer_reset_state(leaf_index)
					.expect("recursive initial active leaf must expose a tokenizer reset");
				state.merge_insert(reset, partition);
			}
			return state;
		}
		crate::runtime::state::ParserStateMap::singleton(self.tokenizer.initial_state(), parser_gss)
	}

	fn collect_original_token_ids(
		groups: &[Vec<u32>],
		internal_tokens: &RangeSetBlaze<u32>,
	) -> Vec<u32> {
		let total_estimate: usize = internal_tokens
			.iter()
			.filter_map(|token| groups.get(token as usize))
			.map(Vec::len)
			.sum();
		let mut all_ids = Vec::with_capacity(total_estimate);
		for internal_token in internal_tokens.iter() {
			if let Some(originals) = groups.get(internal_token as usize) {
				all_ids.extend_from_slice(originals);
			}
		}
		all_ids.sort_unstable();
		all_ids.dedup();
		all_ids
	}

	fn range_set_from_sorted_ids(ids: &[u32]) -> RangeSetBlaze<u32> {
		let Some((&first, rest)) = ids.split_first() else {
			return RangeSetBlaze::new();
		};

		let mut ranges = Vec::new();
		let mut start = first;
		let mut end = first;
		for &id in rest {
			if id == end + 1 {
				end = id;
			} else {
				ranges.push(start..=end);
				start = id;
				end = id;
			}
		}
		ranges.push(start..=end);
		RangeSetBlaze::from_iter(ranges)
	}
}

#[cfg(test)]
mod token_class_tests {
    use super::*;

    fn load(bytes: Vec<u8>) -> Result<TokenClasses, String> {
        let len = bytes.len();
        TokenClasses::decode_backed(std::sync::Arc::new(bytes), 0, len)
    }

    #[test]
    fn identity_and_empty_explicit_quotient_are_distinct() {
        let identity = load(TokenClasses::default().encode()).unwrap();
        let empty = load(TokenClasses::mapped(vec![], vec![]).encode()).unwrap();
        assert_eq!(identity.count(), None);
        assert_eq!(empty.count(), Some(0));
        assert_eq!(identity.groups(), None);
        assert_eq!(empty.groups(), Some(&[][..]));
    }

    #[test]
    fn fixed_width_quotient_round_trips_with_sparse_and_empty_classes() {
        for (id, width) in [(12, 1), (4096, 2), (65535, 4)] {
            let count = id as usize + 2;
            let mut forward = vec![u32::MAX; count];
            forward[0] = id;
            forward[1] = 0;
            let mut inverse = vec![vec![]; count];
            inverse[id as usize].push(0);
            inverse[0].push(1);
            let wire = TokenClasses::mapped(forward.clone(), inverse.clone()).encode();
            assert_eq!(wire[4], width);
            let packed = load(wire.clone()).unwrap();
            assert_eq!(packed.count(), Some(count));
            assert!(packed.cached_groups().is_none());
            assert_eq!(packed.get(0), Some(id));
            assert_eq!(packed.get(2), Some(u32::MAX));
            assert_eq!(packed.get(count), None);
            assert_eq!(packed.encode(), wire);
            assert!(packed.cached_groups().is_none());
            assert_eq!(packed.forward(), forward);
            assert_eq!(packed.groups().unwrap(), inverse);
        }
    }

    #[test]
    fn malformed_packed_class_ids_are_rejected() {
        let mut wire = TokenClasses::mapped(vec![0], vec![vec![0]]).encode();
        wire[TOKEN_MAP_HEADER] = 1;
        assert!(load(wire).is_err());
    }
}
