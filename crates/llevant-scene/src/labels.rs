//! Cells by their labels: the inline `$` comments of their cards
//! (`$ NB CELL - PBS53 - HNB - BSV`, `$/ttl/BHC11` in CAD conversions), for
//! the rule `label "BSV"` ([`Query::Label`](crate::query::Query::Label)).
//!
//! The compiled model keeps no text (it is geometry only, and is read from
//! the cache without the input), so labels are read from the input when a
//! rule needs them and set on the [`ModelIndex`](crate::index::ModelIndex)
//! ([`ModelIndex::set_labels`](crate::index::ModelIndex::set_labels)).
//! They map to MCNP cell numbers, so one `Labels` serves the model, a
//! universe opened on its own and the model compiled again after an edit
//! that kept the comments.
//!
//! Each `$` comment of a cell card is a label of its own; `c` comment lines
//! (a header above a group of cells) are not labels.

use std::path::Path;

use llevant::{Error, Result};
use rayon::prelude::*;
use rustc_hash::FxHashMap;

/// The distinct labels of an input, each with the cells that have it.
#[derive(Clone, Debug, Default)]
pub struct Labels {
    /// Distinct labels, lowercased, one after another, each followed by a
    /// `\n` (no label holds one): one buffer to search, not 100 k strings.
    text: String,
    /// By label: where it ends in `text` (its `\n`).
    ends: Vec<u32>,
    /// By label: the MCNP numbers of its cells, sorted.
    cells: Vec<Vec<i64>>,
}

/// Labels searched as one piece of `text` by each task.
const CHUNK: usize = 4096;

impl Labels {
    /// The labels of the cell cards of `tree`.
    pub fn of(tree: &migjorn::Model) -> Labels {
        let per_cell: Vec<(i64, Vec<String>)> = tree
            .cst()
            .order()
            .par_iter()
            .filter_map(|&slot| {
                let c = tree.cell_at(slot)?;
                let id = c.id()?;
                let labels: Vec<String> = c
                    .comments()
                    .into_iter()
                    .filter(|s| !s.is_empty())
                    .map(str::to_lowercase)
                    .collect();
                (!labels.is_empty()).then_some((id, labels))
            })
            .collect();
        let mut at: FxHashMap<String, usize> = FxHashMap::default();
        let mut out = Labels::default();
        for (id, labels) in per_cell {
            for l in labels {
                let k = *at.entry(l).or_insert_with_key(|l| {
                    // A `\n` cannot be in a comment, but keep the buffer's
                    // separators whatever the parser gives.
                    out.text
                        .extend(l.chars().map(|c| if c == '\n' { ' ' } else { c }));
                    out.text.push('\n');
                    out.ends.push(out.text.len() as u32 - 1);
                    out.cells.push(Vec::new());
                    out.cells.len() - 1
                });
                out.cells[k].push(id);
            }
        }
        for c in &mut out.cells {
            c.sort_unstable();
            c.dedup();
        }
        out
    }

    /// The labels of the input at `path` (parsed again: the compiled model
    /// has no text). Supports MCNP decks and OpenMC XML models/directories.
    pub fn read(path: impl AsRef<Path>) -> Result<Labels> {
        let path = path.as_ref();
        let bytes = llevant_openmc::read_input_bytes(path).map_err(|e| match e {
            llevant_openmc::OpenMcError::Io { path, source } => Error::Io { path, source },
            other => Error::Invalid(other.to_string()),
        })?;
        Ok(Labels::of_text(&String::from_utf8_lossy(&bytes)))
    }

    /// The labels of input text (MCNP or OpenMC XML).
    pub fn of_text(text: &str) -> Labels {
        match llevant_openmc::maybe_translate_bytes(text.as_bytes()) {
            Ok(std::borrow::Cow::Owned(bytes)) => {
                Labels::of(&migjorn::Model::parse(&String::from_utf8_lossy(&bytes)))
            }
            _ => Labels::of(&migjorn::Model::parse(text)),
        }
    }

    /// The MCNP numbers of the cells with a label containing `text`, any
    /// case; sorted, each once.
    pub fn matching(&self, text: &str) -> Vec<i64> {
        let mut out: Vec<i64> = self.matching_unsorted(text).collect();
        out.par_sort_unstable();
        out.dedup();
        out
    }

    /// [`Labels::matching`] in no order, a cell with several such labels
    /// once per label: for a set of cells, which keeps each once anyway.
    pub fn matching_unsorted(&self, text: &str) -> impl Iterator<Item = i64> + '_ {
        self.matching_labels(text)
            .into_iter()
            .flat_map(move |k| self.cells(k).iter().copied())
    }

    /// The labels containing `text`, any case, in order.
    pub fn matching_labels(&self, text: &str) -> Vec<u32> {
        let needle = text.to_lowercase();
        if needle.contains('\n') {
            return Vec::new();
        }
        let finder = memchr::memmem::Finder::new(needle.as_bytes());
        let bytes = self.text.as_bytes();
        let n = self.ends.len();
        (0..n.div_ceil(CHUNK))
            .into_par_iter()
            .flat_map_iter(|chunk| {
                let (first, last) = (chunk * CHUNK, ((chunk + 1) * CHUNK).min(n));
                let mut k = first;
                let mut pos = self.start(first);
                let end = self.ends[last - 1] as usize;
                let mut hits = Vec::new();
                // Each match names the label it starts in; then on from the
                // end of that label.
                while k < last {
                    let Some(at) = finder.find(&bytes[pos..end]) else {
                        break;
                    };
                    let at = (pos + at) as u32;
                    k += self.ends[k..last].partition_point(|&e| e < at);
                    hits.push(k as u32);
                    pos = self.ends[k] as usize + 1;
                    k += 1;
                }
                hits
            })
            .collect()
    }

    /// Where label `k` starts in `text`.
    fn start(&self, k: usize) -> usize {
        if k == 0 {
            0
        } else {
            self.ends[k - 1] as usize + 1
        }
    }

    /// The MCNP numbers of the cells with label `k`, sorted.
    pub fn cells(&self, k: u32) -> &[i64] {
        &self.cells[k as usize]
    }

    /// How many distinct labels there are.
    pub fn len(&self) -> usize {
        self.ends.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ends.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Several labels on one cell, a CRLF line, a header (not a label), an
    /// `FC` card with a `$` in its text (not a cell label).
    const DECK: &str = "labels\r
c HNB vessel\r
1 3 -7.8 -1 imp:n=1 $ NB CELL - HNB - BSV - STAINLESS STEEL\r
2 3 -7.8 -2 imp:n=1  $ NB CELL - HNB - BLV\r
     $ second label of cell 2\r
3 0 1 2 -9 imp:n=1 $/ttl/BHC11\r
9 0 9 imp:n=0\r
\r
1 so 5\r
2 s 20 0 0 5\r
9 so 100\r
\r
m3 26000 1\r
fc4 the $ in a tally comment, HNB\r
";

    #[test]
    fn labels_are_dollar_comments_any_case() {
        let l = Labels::of_text(DECK);
        assert_eq!(l.len(), 4);
        assert_eq!(l.matching("hnb"), [1, 2]);
        assert_eq!(l.matching("HNB - BSV"), [1]);
        assert_eq!(l.matching("Second Label"), [2]);
        assert_eq!(l.matching("/ttl/"), [3]);
        // A header and a tally comment are not labels.
        assert!(l.matching("vessel").is_empty());
        assert!(l.matching("tally").is_empty());
        // No `\r` kept at the end of a label.
        assert_eq!(l.matching("steel"), [1]);
        assert!(l.matching("steel\r").is_empty());
        assert_eq!(l.matching(""), [1, 2, 3]);
        assert!(l.matching("hnb\nbsv").is_empty());
    }

    /// Over several chunks of labels: as testing each label on its own.
    #[test]
    fn matching_spans_chunks() {
        let n = 2 * CHUNK + 300;
        let label = |k: usize| {
            format!(
                "part {k} {}",
                if k.is_multiple_of(7) { "Alpha" } else { "beta" }
            )
        };
        let mut deck = String::from("many\n");
        for k in 1..=n {
            deck.push_str(&format!("{k} 0 -1 imp:n=1 $ {}\n", label(k)));
        }
        deck.push_str("\n1 so 5\n\n");
        let l = Labels::of_text(&deck);
        assert_eq!(l.len(), n);
        for text in [
            "alpha", "BETA", "part 1", "9 b", "a", "", "4095 ", "nothing",
        ] {
            let expect: Vec<i64> = (1..=n)
                .filter(|&k| label(k).to_lowercase().contains(&text.to_lowercase()))
                .map(|k| k as i64)
                .collect();
            assert_eq!(l.matching(text), expect, "{text:?}");
        }
    }

    /// Verify OpenMC cell `name` attributes are extracted as searchable labels from text and files.
    #[test]
    fn labels_from_openmc_text_and_file() {
        let xml = r#"<model>
  <materials>
    <material id="1" name="Tungsten"><density units="g/cm3" value="19.25"/><nuclide ao="1.0" name="W184"/></material>
  </materials>
  <geometry>
    <surface id="1" type="sphere" coeffs="0 0 0 10" boundary="vacuum"/>
    <cell id="7" material="1" region="-1" name="Tungsten First Wall Armor"/>
  </geometry>
</model>"#;
        let l = Labels::of_text(xml);
        assert_eq!(l.matching("tungsten"), [7]);
        assert_eq!(l.matching("FIRST WALL"), [7]);
        assert!(l.matching("nonexistent").is_empty());

        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../docs/examples/openmc/fusion_hts_blanket.xml");
        let file_labels = Labels::read(&path).expect("read OpenMC XML labels");
        assert_eq!(file_labels.matching("Tungsten"), [2]);
        assert_eq!(file_labels.matching("Cryostat"), [10]);
        assert_eq!(file_labels.matching("Breeder"), [4]);
    }
}
