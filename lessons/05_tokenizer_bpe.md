# Lesson 05 — BPE tokenizer from scratch

## Objective

Build a **byte-level Byte-Pair-Encoding** tokenizer (GPT-2 style): pre-tokenize
raw text, learn merges greedily until the vocab hits a target, then
encode/decode any text with exact roundtrip and persist the merges to disk.

By the end: you can turn a 5 MB FineWeb slice into a stream of integer token
ids, the input the language models from lesson 06 onward consume.

## Why byte-level BPE

You could feed bytes directly (vocab 256, no training) — but every word is many
tokens, so the model wastes capacity learning that `t`,`h`,`e` is "the". You
could feed whole words (vocab = every word in the corpus) — but the vocab is
huge and unseen words are un-tokenizable. BPE is the sweet spot: start at the
256 bytes, then **merge the most frequent adjacent pair** into a new token,
repeatedly. Common subwords (`the`, `ing`, `tion`) become single tokens; rare
or unseen sequences fall back to bytes. The vocab is bounded, and **any text
encodes** because byte coverage is total. This is what GPT-2/3 actually use.

The other property that makes this work for *us*: **decode is byte
concatenation**, so roundtrip is *exactly* the input bytes — no special
handling of spaces, punctuation, or unicode. That's why the checkpoint tests
can assert `decode(encode(text)) == text` on arbitrary unicode.

## The model

### Pre-tokenization (the GPT-2 regex, hand-rolled)

BPE merges only ever combine pairs *within a pre-token*, never across the
boundary. The split, roughly:
```
's | 't | 're | 've | 'm | 'll | 'd   (contractions)
| ?\p{L}+                            (optional leading space + letters)
| ?\p{N}+                            (optional leading space + numbers)
| ?[^\s\p{L}\p{N}]+                   (optional leading space + other/punct)
| \s+                                (whitespace run, with a lookahead subtlety)
```
The leading-`?`-space rule is the clever bit: a single space before a word
attaches *to that word's token* (`" world"` not `"world"` after a space). So
the tokenizer can distinguish word-initial from word-medial occurrences of the
same letters — exactly how GPT-2 keeps `" the"` and `"the"` as different tokens.

Zig's std has no `\p{L}` unicode tables. Approximate: ASCII letters/digits/whitespace
exact; codepoints `≥ 0xC0` treated as letters (covers accented Latin + most
scripts); the rest "other." Roundtrip stays exact because the *partition* of
bytes is what matters, not perfect category fidelity. Decode concatenates each
token's raw bytes → the original bytes come back regardless.

### Training (minbpe-style, recomputing stats each merge)

1. Pre-tokenize the training slice into chunks. Each chunk becomes a list of
   token ids, initialized to its raw bytes (`0..255`).
2. Count every adjacent pair across all chunks. Find the most frequent.
3. Append a new token id `= 256 + k` whose bytes are `bytes[a] ++ bytes[b]`;
   record the merge `{a, b}` at rank `k`.
4. Replace every `(a, b)` occurrence with the new id, in every chunk.
5. Repeat until `vocab == target` or the best pair occurs < 2 times.

### Encoding (apply merges by priority)

To encode a chunk: start from its bytes. Repeatedly find the **lowest-rank
(highest-priority) merge present** in the current sequence and apply it
everywhere; stop when no learned merge remains. Rank order matters — later
merges are built from earlier ones, so you must apply them in the order they
were learned (earliest first), not "most frequent remaining pair."

### Decoding

`decode(ids) = concat(token_bytes[id] for id in ids)`. That's it. No
detokenization rules, no special tokens to strip. Because each token is a
slice of the original bytes, concatenating reconstructs them exactly.

## Worked example (tiny)

Corpus: `"ab ab ab"` (after pre-tokenize: `["ab", " ab", " ab"]` — the latter
two carry the leading space; for this example treat all three as byte-lists
`[a,b]`, `[0x20,a,b]`, `[0x20,a,b]`).

Pair counts (merge round 1):
```
(a,b): 3      <- most frequent (appears in all three chunks)
(0x20,a): 2
(b,0x20): 2
```
Merge `(a,b)` → token 256, bytes `"ab"`. Now chunks are `[256]`, `[0x20,256]`,
`[0x20,256]`.

Pair counts (round 2):
```
(0x20, 256): 2   <- most frequent
```
Merge `(0x20,256)` → token 257, bytes `" ab"`. Chunks: `[256]`, `[257]`, `[257]`.

Now `encode("ab")` → look for lowest-rank merge in `[a,b]` → `(a,b)` rank 0 →
`[256]` → no further merges → **single token 256**. `decode([256])` → `"ab"`.
Roundtrip exact, and `"ab"` collapsed from 2 bytes to 1 token. That's the
compression. On real text, a 512-vocab BPE compresses English to ~⅓-½ bytes/token.

> **Determinism:** when two pairs tie in count, you must pick the same one
> every run. The reference iterates the hash map and takes the strict `>`
> winner — so among ties, the *first one encountered in iteration* wins. Hash
> iteration order in Zig's `AutoHashMap` is deterministic for a given set of
> inserted keys, so training is reproducible, but the tie-break is implicit.
> Make it explicit in your head: "highest count, ties broken by first-seen."
> The test `"BPE: training is deterministic"` pins this.

## Tasks

1. **Pre-tokenizer.** Write `preTokenize(text) [][]const u8` returning slices
   *into* the input (no copies). Handle the contraction list, the three
   `?class+` rules, and whitespace runs. Force-advance one byte if no rule
   matches, so you never infinite-loop. Test: `"Hello, world! 123"` splits to
   `|Hello|,| world|!| 123` (leading space attaches to `world` and `123`).

2. **Base vocab + training.** `BPE.train(text, target_vocab)`: build the 256
   single-byte tokens, pre-tokenize, then merge-loop. Store `merges` (ordered
   list of `{a,b}`) and `token_bytes` (id → bytes). Build a `pair_rank` map for
   encode. Stop at `target_vocab` or when the best pair count `< 2`.

3. **Encode.** `encodeChunk` applies the lowest-rank present merge until none;
   `encode(text)` pre-tokenizes and concatenates chunk encodings.

4. **Decode.** `decode(ids)` concatenates `token_bytes[id]`. Assert every id is
   in range.

5. **Persistence.** `save(path)` / `load(path)`: write `num_merges`, then each
   merge's `(a,b)` as little-endian `u32`, then the merged tokens' raw bytes
   (the base 256 are implied, not re-saved). Round-trip: load into a fresh `BPE`
   and `encode` identically.

6. **The tests:** roundtrip on ASCII; roundtrip on unicode+punctuation; the
   `"ab"`-corpus learns `(a,b)` as merge #0 and encodes `"ab"` to one token;
   training is deterministic; save/load preserves merges + roundtrip.

## Checkpoint: the tests that must pass (all in `tokenizer.zig`)

- `test "preTokenize: splits words, keeps leading space, handles punctuation"`
- `test "BPE: roundtrip on plain ASCII"` — `decode(encode(text)) == text`
- `test "BPE: roundtrip on unicode + punctuation"` — exact on `Café — don't!`
- `test "BPE: learns the obvious merge on a repetitive corpus"` —
  `"ab"**200` → first merge is `('a','b')`; `encode("ab")` is one token, id 256.
- `test "BPE: training is deterministic"` — same corpus → identical merge list.
- `test "BPE: save/load preserves merges and roundtrip"` — loaded tokenizer
  encodes identically.

Run: `zig test src/tokenizer.zig`. The bar: all six green.

## Common pitfalls

- **No base 256.** If you forget the single-byte tokens, the first merge has
  nowhere to land and encode can't fall back to bytes for unseen pairs. Always
  seed `token_bytes[0..256]` = the 256 single bytes.
- **Merging across pre-token boundaries.** Pairs must be counted *within* a
  chunk only. If you flatten the whole corpus into one token list, you'll
  merge across spaces and destroy the leading-space semantics. Keep chunks
  separate through training.
- **Non-deterministic tie-breaking.** If your "best pair" scan isn't
  reproducible, two runs on the same corpus produce different vocabs → your
  checkpoint tokenizer doesn't match your model's embedding table. Make the
  tie-break explicit and pinned by a test.
- **Applying merges in frequency order at encode time.** Wrong. At encode you
  apply by **learned rank** (earliest first), because later merges' tokens are
  composed of earlier ones. Encoding by "current most frequent" gives different
  (wrong) tokenization.
- **Zig 0.16 file IO for save/load.** `std.fs.cwd()` is gone. Use
  `std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = bytes })` and
  `readFileAlloc(io, path, alloc, .limited(1<<28))` with
  `const io = std.Options.debug_io;`. Write `u32`s with
  `std.mem.writeInt(u32, &buf, v, .little)` and read with
  `std.mem.readInt(u32, data[off..][0..4], .little)`.
- **Stray/truncated UTF-8.** Your codepoint reader must never crash on invalid
  UTF-8 — decode the stray byte as itself. Total byte coverage is what makes
  roundtrip exact on arbitrary input.
- **`std.ArrayList(u8)` is unmanaged in 0.16** — `.empty`, and `append` /
  `appendSlice` / `deinit` take the allocator as the first arg.

## What good looks like

- All six tokenizer tests green; `decode(encode(any text)) == text` holds on
  ASCII, unicode, and punctuation.
- `"ab"**200` collapses to a single token; the first merge is exactly
  `('a','b')` → id 256.
- Save/load round-trips the merge table and reproduces encodings.
- You can explain *why* decode is just byte concatenation (the tokens partition
  the input bytes), and *why* encode uses rank order not frequency order.

Lesson 06 puts this tokenizer in front of the FineWeb corpus and streams
next-token-prediction batches.