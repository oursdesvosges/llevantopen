//! The input as written: the text and line of each cell, surface,
//! material and transform (`TRn`) card, by MCNP number, for frontends that show where something
//! comes from and edit it. Separate from [`Model`](llevant::Model), which a
//! cache can give without reading the input: a frontend loads a `Source`
//! when it wants one (one more parse of the input).
//!
//! A `Source` keeps migjorn's lossless tree of the input: [`Source::edit`]
//! replaces one card's text in place (nothing is parsed again; everything
//! else stays byte for byte as written), and
//! [`Model::recompile_parsed`](llevant::Model::recompile_parsed) compiles
//! the tree as it is. [`edit_card`] makes one edit to input text.

use std::io;
use std::path::Path;
use std::sync::{RwLock, RwLockReadGuard};

use llevant::{Error, Result};

mod fields;
mod many;
pub use fields::{
    coefficient_names, general_forms, geometry_words, wrap_long_lines, CellChange, CellFields,
    Entry, Flag, MaterialFields, Param, Spanned, Word,
};
pub use many::{IdKind, Renumbered, HIGHEST_NUMBER};

/// MCNP lines may hold at most this many columns.
pub const MAX_COLUMNS: usize = 128;

/// The kinds of card [`Source`] shows and edits.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CardKind {
    Cell,
    Surface,
    Material,
    /// `TRn` and `*TRn`.
    Transform,
    /// Another data card (a tally, a card per cell…): one a renumbering
    /// changed, found by its slot only.
    Data,
}

impl CardKind {
    pub fn name(self) -> &'static str {
        match self {
            CardKind::Cell => "cell",
            CardKind::Surface => "surface",
            CardKind::Material => "material",
            CardKind::Transform => "transform",
            CardKind::Data => "data card",
        }
    }
}

/// `input` with the card of `kind` and number `id` written as `text`
/// instead, and the card's text before: [`Source::edit`] on text.
pub fn edit_card(
    input: &str,
    kind: CardKind,
    id: i64,
    text: &str,
) -> std::result::Result<(String, String), String> {
    let src = Source::from_str(input);
    let e = src.edit(kind, id, text)?;
    Ok((src.to_source(), e.old))
}

/// One card of the input.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Card {
    /// 1-based line of its first token.
    pub line: usize,
    /// Its text as written: continuation lines, inline comments and the
    /// line terminator included.
    pub text: String,
}

/// One card edited by [`Source::edit`]: where it is and what it was.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Edited {
    pub kind: CardKind,
    /// The card's place in the tree, which every edit of it (a renumbering
    /// too) keeps: what [`Source::restore`] finds it by.
    pub slot: u32,
    /// Its text before.
    pub old: String,
    /// Its text now (a line terminator added if the text given had none).
    pub new: String,
    /// A card added (its text before is empty): its number, by which it is
    /// found to take it out (added again, it has another slot).
    pub added: Option<i64>,
}

/// The cards of an input, by number, in migjorn's lossless tree. The first
/// card wins when a number is defined twice (as in the compiled model).
///
/// Shared between threads: reading takes a lock that an edit holds only
/// while it splices one card in.
pub struct Source {
    tree: RwLock<migjorn::Model>,
    /// The line of each card (by slot), as of the last edit.
    lines: RwLock<CardLines>,
}

/// The line of every card, by slot: what [`migjorn::Model::line_of`] gives,
/// kept as of the last edit that moved lines. migjorn drops its own table at
/// every edit and counts the whole input again at the next lookup (0.25 s on
/// the 378 MB reference model), which a lookup on the viewer's UI thread
/// must not pay; this one is counted again only when an edit adds or removes
/// lines ([`Source::refresh_lines`]), off the UI thread.
#[derive(Debug, Default)]
pub(crate) struct CardLines(Vec<u32>);

impl CardLines {
    pub(crate) fn of(tree: &migjorn::Model) -> CardLines {
        let order = tree.cst().order();
        let mut lines = vec![0u32; order.iter().max().map_or(0, |&s| s as usize + 1)];
        for &s in order {
            if let Some(l) = tree.line_of(s) {
                lines[s as usize] = u32::try_from(l).unwrap_or(u32::MAX);
            }
        }
        CardLines(lines)
    }

    /// The line of the card at `slot`; `None` for a slot not in the input
    /// when the table was made.
    fn line(&self, slot: u32) -> Option<usize> {
        match self.0.get(slot as usize) {
            Some(&0) | None => None,
            Some(&l) => Some(l as usize),
        }
    }
}

impl std::fmt::Debug for Source {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Source").finish_non_exhaustive()
    }
}

impl Source {
    /// Read the input at `path` (supports MCNP decks and OpenMC XML models/directories).
    pub fn load(path: impl AsRef<Path>) -> Result<Source> {
        let path = path.as_ref();
        let bytes = llevant_openmc::read_input_bytes(path).map_err(|e| match e {
            llevant_openmc::OpenMcError::Io { path, source } => Error::Io { path, source },
            other => Error::Invalid(other.to_string()),
        })?;
        Ok(Source::from_str(&String::from_utf8_lossy(&bytes)))
    }

    /// The cards of input text (MCNP or OpenMC XML).
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(text: &str) -> Source {
        match llevant_openmc::maybe_translate_bytes(text.as_bytes()) {
            Ok(std::borrow::Cow::Owned(bytes)) => {
                Source::from_tree(migjorn::Model::parse(&String::from_utf8_lossy(&bytes)))
            }
            _ => Source::from_tree(migjorn::Model::parse(text)),
        }
    }

    /// The cards of a parsed input.
    pub fn from_tree(tree: migjorn::Model) -> Source {
        let lines = CardLines::of(&tree);
        Source {
            tree: RwLock::new(tree),
            lines: RwLock::new(lines),
        }
    }

    /// The tree as it is now (to compile, or to read other cards of),
    /// edits waiting while it is held.
    pub fn tree(&self) -> RwLockReadGuard<'_, migjorn::Model> {
        self.tree.read().unwrap_or_else(|e| e.into_inner())
    }

    pub fn cell(&self, id: i64) -> Option<Card> {
        self.card(CardKind::Cell, id)
    }

    pub fn surface(&self, id: i64) -> Option<Card> {
        self.card(CardKind::Surface, id)
    }

    pub fn material(&self, id: i64) -> Option<Card> {
        self.card(CardKind::Material, id)
    }

    pub fn transform(&self, id: i64) -> Option<Card> {
        self.card(CardKind::Transform, id)
    }

    /// The `TRn` numbers the card of `kind` and number `id` names: a
    /// surface's transform; a cell's `TRCL` and `FILL` transforms (by
    /// number, not written inline).
    pub fn transforms_of(&self, kind: CardKind, id: i64) -> Vec<i64> {
        let tree = self.tree();
        match kind {
            CardKind::Surface => tree
                .surface(id)
                .and_then(|s| s.transform())
                .into_iter()
                .collect(),
            CardKind::Cell => {
                let Some(c) = tree.cell(id) else {
                    return Vec::new();
                };
                let mut n = Vec::new();
                if let Ok(Some(migjorn::TransformSpec::Number(t))) = c.trcl() {
                    n.push(t);
                }
                if let Ok(Some(migjorn::Fill::Single {
                    transform: Some(migjorn::TransformSpec::Number(t)),
                    ..
                })) = c.fill()
                {
                    n.push(t);
                }
                n.dedup();
                n
            }
            CardKind::Material | CardKind::Transform | CardKind::Data => Vec::new(),
        }
    }

    /// The card of `kind` and number `id`.
    pub fn card(&self, kind: CardKind, id: i64) -> Option<Card> {
        let (slot, text) = find(&self.tree(), kind, id)?;
        let lines = self.lines.read().unwrap_or_else(|e| e.into_inner());
        Some(Card {
            line: lines.line(slot).unwrap_or(0),
            text,
        })
    }

    /// Write the card of `kind` and number `id` as `text` instead (a line
    /// terminator is added if it has none). Only that card changes;
    /// comments, spacing and every other card stay as written. An error,
    /// changing nothing, when the input has no such card (one in a `READ`
    /// file is not in the input), when `text` does not read as a card of
    /// that kind, or when a line is over [`MAX_COLUMNS`].
    pub fn edit(&self, kind: CardKind, id: i64, text: &str) -> std::result::Result<Edited, String> {
        let what = format!("{} {id}", kind.name());
        if let Some((n, line)) = text
            .lines()
            .enumerate()
            .find(|(_, l)| l.chars().count() > MAX_COLUMNS)
        {
            return Err(format!(
                "{what}: line {} has {} columns, over MCNP's {MAX_COLUMNS}",
                n + 1,
                line.chars().count()
            ));
        }
        let mut tree = self.tree.write().unwrap_or_else(|e| e.into_inner());
        let (slot, old) = find(&tree, kind, id)
            .ok_or_else(|| format!("{what} is not in the input (in a READ file?)"))?;
        let mut text = text.to_string();
        if !text.ends_with('\n') {
            text.push_str(if old.ends_with("\r\n") { "\r\n" } else { "\n" });
        }
        set_text(&mut tree, kind, slot, &text).map_err(|e| format!("{what}: {e}"))?;
        drop(tree);
        self.refresh_lines(&old, &text);
        Ok(Edited {
            kind,
            slot,
            old,
            new: text,
            added: None,
        })
    }

    /// Surface `id`'s type and coefficients as written (`None` if it is
    /// not in the input or does not read).
    pub fn surface_coeffs(&self, id: i64) -> Option<(String, Vec<f64>)> {
        let tree = self.tree();
        let s = tree.surface(id)?;
        Some((s.mnemonic()?.to_string(), s.coeffs().ok()?))
    }

    /// Surface `id`'s card with its coefficients written as `coeffs`
    /// instead, everything else as written (for [`Source::edit`]).
    pub fn surface_with_coeffs(
        &self,
        id: i64,
        coeffs: &[f64],
    ) -> std::result::Result<String, String> {
        let card = self
            .surface(id)
            .ok_or_else(|| format!("surface {id} is not in the input (in a READ file?)"))?;
        // The card alone in an input of its own, edited by migjorn.
        let mut m = migjorn::Model::parse(&format!("t\n1 0 -1\n\n{}", card.text));
        let mut s = m
            .surface_mut(id)
            .ok_or_else(|| format!("surface {id}: its card does not read alone"))?;
        s.set_coeffs(coeffs)
            .map_err(|e| format!("surface {id}: {e}"))?;
        let text = m
            .surface(id)
            .map(|s| s.text().to_string())
            .ok_or_else(|| format!("surface {id}: lost"))?;
        Ok(text)
    }

    /// Take an edit back: its card written as it was.
    pub fn restore(&self, e: &Edited) -> std::result::Result<(), String> {
        let mut tree = self.tree.write().unwrap_or_else(|e| e.into_inner());
        set_text(&mut tree, e.kind, e.slot, &e.old)
            .map_err(|err| format!("{} (slot {}): {err}", e.kind.name(), e.slot))?;
        drop(tree);
        self.refresh_lines(&e.new, &e.old);
        Ok(())
    }

    /// Make an edit taken back again: its card written as it was after it.
    pub fn redo(&self, e: &Edited) -> std::result::Result<(), String> {
        let mut tree = self.tree.write().unwrap_or_else(|e| e.into_inner());
        set_text(&mut tree, e.kind, e.slot, &e.new)
            .map_err(|err| format!("{} (slot {}): {err}", e.kind.name(), e.slot))?;
        drop(tree);
        self.refresh_lines(&e.old, &e.new);
        Ok(())
    }

    /// The lines of the cards counted again (one pass over the cards), when
    /// an edit from `before` to `after` can have moved them.
    fn refresh_lines(&self, before: &str, after: &str) {
        let newlines = |t: &str| t.bytes().filter(|&b| b == b'\n').count();
        let leading = |t: &str| t.lines().take_while(|l| maybe_comment(l)).count();
        if newlines(before) == newlines(after) && leading(before) == leading(after) {
            return;
        }
        let lines = CardLines::of(&self.tree());
        *self.lines.write().unwrap_or_else(|e| e.into_inner()) = lines;
    }

    /// The input as it is now.
    pub fn to_source(&self) -> String {
        self.tree().to_source()
    }

    /// Write the input as it is now to `w`, card by card.
    pub fn write_source(&self, w: &mut impl io::Write) -> io::Result<()> {
        self.tree().write_source(w)
    }
}

/// The slot and text of the card of `kind` and number `id`.
fn find(tree: &migjorn::Model, kind: CardKind, id: i64) -> Option<(u32, String)> {
    match kind {
        CardKind::Cell => tree.cell(id).map(|c| (c.slot(), c.text().to_string())),
        CardKind::Surface => tree.surface(id).map(|c| (c.slot(), c.text().to_string())),
        CardKind::Material => tree.material(id).map(|c| (c.slot(), c.text().to_string())),
        CardKind::Transform => tree.transform(id).map(|c| (c.slot(), c.text().to_string())),
        CardKind::Data => None,
    }
}

/// Whether a line may be one migjorn counts before a card's first token (a
/// comment or blank line). Errs towards yes: it only decides whether the
/// lines are counted again.
fn maybe_comment(l: &str) -> bool {
    let t = l.trim_start();
    t.is_empty() || t.starts_with(['c', 'C', '$'])
}

/// Write the card at `slot` as `text`, which must read as a card of `kind`
/// with a number; else the card is left as it was.
fn set_text(
    tree: &mut migjorn::Model,
    kind: CardKind,
    slot: u32,
    text: &str,
) -> std::result::Result<(), String> {
    let write = |tree: &mut migjorn::Model, text: &str| match kind {
        CardKind::Cell => tree.cell_mut_at(slot).map(|mut c| c.set_text(text)),
        CardKind::Surface => tree.surface_mut_at(slot).map(|mut c| c.set_text(text)),
        CardKind::Material => tree.material_mut_at(slot).map(|mut c| c.set_text(text)),
        CardKind::Transform => tree.transform_mut_at(slot).map(|mut c| c.set_text(text)),
        CardKind::Data => tree.data_card_mut_at(slot).map(|mut c| c.set_text(text)),
    };
    let numbered = |tree: &migjorn::Model| match kind {
        CardKind::Cell => tree.cell_at(slot).and_then(|c| c.id()),
        CardKind::Surface => tree.surface_at(slot).and_then(|c| c.id()),
        CardKind::Material => tree.material_at(slot).and_then(|c| c.id()),
        CardKind::Transform => tree.transform_at(slot).and_then(|c| c.id()),
        CardKind::Data => tree.data_card_at(slot).map(|_| 0),
    };
    let before = tree
        .cst()
        .card(slot)
        .map(|c| c.text().to_string())
        .ok_or("not in the input")?;
    match write(tree, text) {
        Some(Ok(())) => {}
        Some(Err(e)) => return Err(e.to_string()),
        None => return Err("not in the input".into()),
    }
    if numbered(tree).is_none() {
        // migjorn took it (as some card); not as one of this kind.
        if let Some(Err(e)) = write(tree, &before) {
            return Err(format!("could not put the card back: {e}"));
        }
        return Err(format!("does not read as a {} card", kind.name()));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn surface_coefficients_written_in_place() {
        let src = Source::from_str(
            "title
1 0 -10 imp:n=1

10 5 so   2.5  $ the ball
11 pz 0

",
        );
        let (kind, c) = src.surface_coeffs(10).unwrap();
        assert_eq!(kind.to_ascii_lowercase(), "so");
        assert_eq!(c, vec![2.5]);
        let text = src.surface_with_coeffs(10, &[3.25]).unwrap();
        assert!(text.contains("3.25"), "{text}");
        assert!(text.contains("$ the ball"), "{text}");
        assert!(text.trim_start().starts_with("10 5 so"), "{text}");
        let e = src.edit(CardKind::Surface, 10, &text).unwrap();
        assert_eq!(src.surface_coeffs(10).unwrap().1, vec![3.25]);
        assert!(e.old.contains("2.5"));
        assert!(src.surface_with_coeffs(99, &[1.0]).is_err());
    }

    #[test]
    fn cards_by_number_with_their_lines() {
        let s = Source::from_str(
            "title
c a comment
1 1 -7.8 -1 imp:n=1
     $ more
2 0 1
     -9 imp:n=1
3 0 9 imp:n=0

1 so 5
9 so 100

m1 26000 1
",
        );
        let c2 = s.cell(2).unwrap();
        assert_eq!(c2.line, 5);
        assert!(c2.text.starts_with("2 0 1") && c2.text.contains("-9 imp:n=1"));
        assert_eq!(s.cell(1).unwrap().line, 3);
        assert_eq!(s.surface(9).unwrap().line, 10);
        assert!(s.material(1).unwrap().text.starts_with("m1 26000"));
        assert!(s.cell(4).is_none());
    }

    const INPUT: &str = "title
c a comment
1 1 -7.8 -1 imp:n=1 $ the ball
2 0 1 -9 imp:n=1
3 0 9 imp:n=0

1 so 5
9 so 100

m1 26000 1
";

    #[test]
    fn an_edited_card_changes_that_card_alone() {
        let (new, old) = edit_card(INPUT, CardKind::Surface, 1, "1 so 6").unwrap();
        assert_eq!(old, "1 so 5\n");
        assert_eq!(new, INPUT.replace("1 so 5\n", "1 so 6\n"));
        let (new, _) =
            edit_card(INPUT, CardKind::Cell, 1, "1 1 -7.9 -1 imp:n=1 $ heavier\n").unwrap();
        // The comment above a card is part of its text (as the inspector shows it).
        assert_eq!(
            new,
            INPUT.replace(
                "c a comment\n1 1 -7.8 -1 imp:n=1 $ the ball",
                "1 1 -7.9 -1 imp:n=1 $ heavier"
            )
        );
        let (new, _) = edit_card(INPUT, CardKind::Material, 1, "m1 26056 1").unwrap();
        assert!(new.contains("m1 26056 1\n") && !new.contains("26000"));
        let with_tr = INPUT.replace("m1 26000 1\n", "m1 26000 1\n*tr3 1 0 0\n");
        let (new, old) =
            edit_card(&with_tr, CardKind::Transform, 3, "*tr3 1 2 0  10 80 90").unwrap();
        assert_eq!(old, "*tr3 1 0 0\n");
        assert_eq!(
            new,
            with_tr.replace("*tr3 1 0 0\n", "*tr3 1 2 0  10 80 90\n")
        );
        assert!(edit_card(&with_tr, CardKind::Transform, 3, "1 so 5").is_err());
        // Renumbered: found under its new number.
        let (new, _) = edit_card(INPUT, CardKind::Surface, 9, "19 so 100").unwrap();
        assert!(Source::from_str(&new).surface(19).is_some());
    }

    #[test]
    fn edits_that_cannot_be_made_change_nothing() {
        assert!(edit_card(INPUT, CardKind::Cell, 7, "7 0 -1").is_err());
        assert!(edit_card(INPUT, CardKind::Surface, 1, "1 so").is_err());
        assert!(edit_card(INPUT, CardKind::Surface, 1, "this is not a surface").is_err());
        let long = format!("1 so 5 {}", " ".repeat(130));
        let e = edit_card(INPUT, CardKind::Surface, 1, &long).unwrap_err();
        assert!(e.contains("128"), "{e}");
    }

    #[test]
    fn the_transforms_a_card_names() {
        let s = Source::from_str(
            "t
1 0 -1 trcl=4 imp:n=1
2 0 1 -9 fill=1 (5) imp:n=1
3 0 9 trcl=(1 0 0) imp:n=0

1 2 so 5
9 so 100

tr2 0 0 1
tr4 1 0 0
*tr5 0 1 0
",
        );
        assert_eq!(s.transforms_of(CardKind::Surface, 1), vec![2]);
        assert!(s.transforms_of(CardKind::Surface, 9).is_empty());
        assert_eq!(s.transforms_of(CardKind::Cell, 1), vec![4]);
        assert_eq!(s.transforms_of(CardKind::Cell, 2), vec![5]);
        assert!(s.transforms_of(CardKind::Cell, 3).is_empty());
        assert_eq!(s.transform(5).unwrap().line, 11);
        assert!(s.transform(5).unwrap().text.starts_with("*tr5"));
    }

    /// Edits in place, one after another, are what a parse of the text
    /// gives (cards and lines), and each takes back to the text before.
    #[test]
    fn edits_in_place_and_taken_back() {
        let s = Source::from_str(INPUT);
        let a = s
            .edit(CardKind::Cell, 2, "2 0 1\n     -9 imp:n=1\n")
            .unwrap();
        let b = s.edit(CardKind::Surface, 9, "c bigger\n19 so 200").unwrap();
        let c = s.edit(CardKind::Cell, 1, "1 1 -7.9 -1 imp:n=1").unwrap();
        let text = s.to_source();
        // Edits that cannot be made leave the text as it is.
        assert!(s
            .edit(CardKind::Surface, 1, "this is not a surface")
            .is_err());
        assert!(s.edit(CardKind::Cell, 3, "c only a comment").is_err());
        assert_eq!(s.to_source(), text);
        let fresh = Source::from_str(&text);
        for (k, id) in [
            (CardKind::Cell, 1),
            (CardKind::Cell, 2),
            (CardKind::Cell, 3),
            (CardKind::Surface, 1),
            (CardKind::Surface, 19),
            (CardKind::Material, 1),
        ] {
            assert_eq!(s.card(k, id), fresh.card(k, id), "{k:?} {id}");
        }
        assert!(s.surface(9).is_none());
        let mut out = Vec::new();
        s.write_source(&mut out).unwrap();
        assert_eq!(out, text.as_bytes());
        for e in [&c, &b, &a] {
            s.restore(e).unwrap();
        }
        let back = s.to_source();
        for e in [&a, &b, &c] {
            s.redo(e).unwrap();
        }
        assert_eq!(s.to_source(), text);
        for e in [&c, &b, &a] {
            s.restore(e).unwrap();
        }
        assert_eq!(back, INPUT);
        assert_eq!(s.to_source(), INPUT);
        assert_eq!(s.surface(9).unwrap().line, 8);
        assert_eq!(s.cell(3).unwrap().line, 5);
    }
}
