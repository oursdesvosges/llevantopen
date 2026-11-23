//! Parallel interning of encoded labels.
//!
//! Each worker interns the labels of one chunk of queries into a [`Chunk`]:
//! local ids, and the distinct keys in order of first occurrence, stored flat
//! (one allocation per chunk, not per key). [`merge`] then gives global ids
//! numbered by first occurrence over all chunks in order, exactly as a
//! sequential pass would, so results do not depend on the thread count.
//!
//! The merge spreads key positions over shards by hash; each shard finds, in
//! parallel, the first position of each of its keys. A key's global id is
//! the number of distinct keys first seen before it. Temporary buffers are
//! few and reused: freeing many large allocations is slow with many threads.

use std::hash::{Hash, Hasher};

use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHasher};

const NONE: u32 = u32::MAX;
const SHARDS: usize = 256;

fn hash(key: &[u32]) -> u64 {
    let mut h = FxHasher::default();
    key.hash(&mut h);
    h.finish()
}

/// The labels of one chunk of queries.
#[derive(Default)]
pub(crate) struct Chunk {
    /// Local id of each query.
    pub ids: Vec<u32>,
    words: Vec<u32>,
    /// End of each key in `words`.
    ends: Vec<u32>,
    hashes: Vec<u64>,
    /// First key with each hash, and the next key with the same hash.
    heads: FxHashMap<u64, u32>,
    next: Vec<u32>,
}

#[inline]
fn slice_key<'a>(words: &'a [u32], ends: &[u32], i: usize) -> &'a [u32] {
    let start = if i == 0 { 0 } else { ends[i - 1] as usize };
    &words[start..ends[i] as usize]
}

impl Chunk {
    /// Record a query with label `key`.
    pub fn push(&mut self, key: &[u32]) {
        let h = hash(key);
        match self.heads.entry(h) {
            std::collections::hash_map::Entry::Occupied(mut e) => {
                let head = *e.get();
                let mut i = head;
                while i != NONE {
                    if slice_key(&self.words, &self.ends, i as usize) == key {
                        self.ids.push(i);
                        return;
                    }
                    i = self.next[i as usize];
                }
                let id = self.ends.len() as u32;
                self.words.extend_from_slice(key);
                self.ends.push(self.words.len() as u32);
                self.hashes.push(h);
                self.next.push(head);
                *e.get_mut() = id;
                self.ids.push(id);
            }
            std::collections::hash_map::Entry::Vacant(e) => {
                let id = self.ends.len() as u32;
                self.words.extend_from_slice(key);
                self.ends.push(self.words.len() as u32);
                self.hashes.push(h);
                self.next.push(NONE);
                e.insert(id);
                self.ids.push(id);
            }
        }
    }

    /// Key of local id `i`.
    pub fn key(&self, i: usize) -> &[u32] {
        slice_key(&self.words, &self.ends, i)
    }

    /// Hash of the key of local id `i`.
    pub fn hash_of(&self, i: usize) -> u64 {
        self.hashes[i]
    }

    /// Keep only the keys (`key`, `len`): no more queries, ids or hashes.
    pub fn keys_only(&mut self) {
        *self = Chunk {
            words: std::mem::take(&mut self.words),
            ends: std::mem::take(&mut self.ends),
            ..Chunk::default()
        };
        self.words.shrink_to_fit();
        self.ends.shrink_to_fit();
    }

    /// Number of distinct keys.
    pub fn len(&self) -> usize {
        self.ends.len()
    }
}

/// Global ids of all queries, chunk after chunk, and the distinct keys
/// indexed by global id.
pub(crate) fn merge(chunks: &[Chunk]) -> (Vec<u32>, Vec<&[u32]>) {
    let mut offsets = Vec::with_capacity(chunks.len() + 1);
    offsets.push(0);
    for c in chunks {
        offsets.push(offsets.last().unwrap() + c.len());
    }
    let n = *offsets.last().unwrap();
    let key = |p: usize| {
        let c = offsets.partition_point(|&o| o <= p) - 1;
        chunks[c].key(p - offsets[c])
    };

    // Positions grouped by shard, in increasing order within each shard.
    let shard_of: Vec<u8> = chunks
        .par_iter()
        .flat_map_iter(|c| c.hashes.iter().map(|h| (h >> 56) as u8))
        .collect();
    let mut order: Vec<u32> = (0..n as u32).into_par_iter().collect();
    order.par_sort_unstable_by_key(|&p| (shard_of[p as usize], p));
    let mut parts = Vec::with_capacity(SHARDS);
    let mut buf = vec![0u32; n];
    let mut rest: &mut [u32] = &mut buf;
    let mut lo = 0;
    for s in 0..SHARDS {
        let len = order[lo..].partition_point(|&p| shard_of[p as usize] as usize == s);
        let (part, tail) = rest.split_at_mut(len);
        parts.push((&order[lo..lo + len], part));
        rest = tail;
        lo += len;
    }

    // First position of each key, aligned with `order`: positions arrive in
    // increasing order, so the first one inserted is the smallest.
    parts.into_par_iter().for_each(|(pos, out)| {
        let mut seen: FxHashMap<&[u32], u32> = FxHashMap::default();
        for (&p, o) in pos.iter().zip(out) {
            *o = *seen.entry(key(p as usize)).or_insert(p);
        }
    });
    let mut first = vec![0u32; n];
    for (&p, &f) in order.iter().zip(&buf) {
        first[p as usize] = f;
    }

    // Number distinct keys by first position (into `buf`), then give every
    // position its key's number (into `order`).
    let mut distinct = 0u32;
    for (p, &f) in first.iter().enumerate() {
        if f == p as u32 {
            buf[p] = distinct;
            distinct += 1;
        }
    }
    order
        .par_iter_mut()
        .zip(&first)
        .for_each(|(o, &f)| *o = buf[f as usize]);
    let global = &order;

    let ids = chunks
        .par_iter()
        .zip(&offsets)
        .flat_map_iter(|(c, &off)| c.ids.iter().map(move |&i| global[off + i as usize]))
        .collect();
    let table = (0..n)
        .into_par_iter()
        .filter(|&p| first[p] == p as u32)
        .map(key)
        .collect();
    (ids, table)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The sequential merge this replaces.
    fn reference(queries: &[Vec<Vec<u32>>]) -> (Vec<u32>, Vec<Vec<u32>>) {
        let mut global: FxHashMap<Vec<u32>, u32> = FxHashMap::default();
        let mut table = Vec::new();
        let mut ids = Vec::new();
        for chunk in queries {
            for k in chunk {
                ids.push(*global.entry(k.clone()).or_insert_with(|| {
                    table.push(k.clone());
                    table.len() as u32 - 1
                }));
            }
        }
        (ids, table)
    }

    #[test]
    fn matches_the_sequential_merge() {
        let mut r = 0x9e3779b97f4a7c15u64;
        let mut next = move |m: u64| {
            r ^= r << 13;
            r ^= r >> 7;
            r ^= r << 17;
            r % m
        };
        for nkeys in [1, 7, 300, 20_000] {
            let queries: Vec<Vec<Vec<u32>>> = (0..40)
                .map(|_| {
                    (0..next(500))
                        .map(|_| {
                            let v = next(nkeys);
                            vec![v as u32; 1 + (v % 5) as usize]
                        })
                        .collect()
                })
                .collect();
            let chunks: Vec<Chunk> = queries
                .iter()
                .map(|q| {
                    let mut c = Chunk::default();
                    for k in q {
                        c.push(k);
                    }
                    c
                })
                .collect();
            let (ids, table) = merge(&chunks);
            let table: Vec<Vec<u32>> = table.into_iter().map(|k| k.to_vec()).collect();
            assert_eq!((ids, table), reference(&queries));
        }
    }

    #[test]
    fn hash_collisions_within_a_chunk_are_resolved() {
        // Force a collision: make [3] start its chain at the key [1, 2].
        let mut c = Chunk::default();
        c.push(&[1, 2]);
        c.heads.insert(hash(&[3]), 0);
        c.push(&[3]);
        c.push(&[1, 2]);
        c.push(&[3]);
        assert_eq!(c.ids, [0, 1, 0, 1]);
    }
}
