//! Lossless JPEG (ITU-T T.81 process 14, "LJ92") encoder for the DNG raw
//! plane. DNG `Compression = 7` means every strip/tile is a complete
//! SOI..EOI lossless-JPEG stream; this is the only compression the DNG
//! spec sanctions for 16-bit *integer* raw data, and the one every RAW
//! engine decodes (Adobe, LibRaw/RawTherapee, Capture One, Apple —
//! Apple's engine rejects the whole file when the raw IFD uses Deflate,
//! which the spec reserves for floating-point/32-bit data; that rejection
//! broke Finder/Quick Look previews of `-compress` DNGs on macOS).
//!
//! Encoding choices (all within what the common decoders support):
//! - predictor 1 (Ra, the sample to the left), point transform 0;
//! - one scan, components interleaved (matches our interleaved raster);
//! - a single optimal Huffman table shared by all components, built per
//!   strip with the Annex-K algorithm (same as libjpeg / the DNG SDK);
//! - no restart markers.

/// Encode one strip as a standalone lossless-JPEG stream.
///
/// `samples` is the interleaved raster, `width * height * components`
/// long. `width`/`height` must fit the SOF3 16-bit fields and
/// `components` the 8-bit field — callers pass strip dimensions, which
/// are far below the limits.
pub(crate) fn encode(samples: &[u16], width: usize, height: usize, components: usize) -> Vec<u8> {
    assert!(width > 0 && width <= u16::MAX as usize);
    assert!(height > 0 && height <= u16::MAX as usize);
    assert!(components > 0 && components <= 4);
    assert_eq!(samples.len(), width * height * components);

    // Pass 1: per-sample prediction differences (mod 2^16) + the
    // category histogram the Huffman table is built from.
    let mut diffs: Vec<u16> = Vec::with_capacity(samples.len());
    let mut freq = [0u32; 17];
    let row_samples = width * components;
    for row in 0..height {
        for col in 0..width {
            let off = row * row_samples + col * components;
            for c in 0..components {
                let pred = if col > 0 {
                    samples[off + c - components] // Ra
                } else if row > 0 {
                    samples[off + c - row_samples] // Rb
                } else {
                    1 << 15 // 2^(P - Pt - 1)
                };
                let diff = samples[off + c].wrapping_sub(pred);
                freq[category(diff) as usize] += 1;
                diffs.push(diff);
            }
        }
    }

    let table = HuffTable::optimal(&freq);

    let mut out = Vec::with_capacity(samples.len()); // ~50% typical ratio
    out.extend_from_slice(&[0xFF, 0xD8]); // SOI
    write_dht(&mut out, &table);
    write_sof3(&mut out, width as u16, height as u16, components as u8);
    write_sos(&mut out, components as u8);

    // Pass 2: entropy-coded data.
    let mut bw = BitWriter::new(out);
    for &diff in &diffs {
        let ssss = category(diff);
        let (code, len) = table.codes[ssss as usize];
        bw.put(code as u32, len);
        if ssss > 0 && ssss < 16 {
            // Low SSSS bits of diff for d > 0, of diff-1 for d < 0 (the
            // standard DC-difference convention). SSSS 0 and 16 carry no
            // extra bits.
            let d = diff as i16 as i32;
            let bits = if d > 0 { d } else { d - 1 };
            bw.put((bits as u32) & ((1 << ssss) - 1), ssss as u32);
        }
    }
    let mut out = bw.finish();
    out.extend_from_slice(&[0xFF, 0xD9]); // EOI
    out
}

/// JPEG difference category: the SSSS value for a mod-2^16 difference.
/// 0 → 0, ±1 → 1, … ±32767 → 15, and the special 32768 (≡ −32768) → 16.
fn category(diff: u16) -> u8 {
    if diff == 0 {
        return 0;
    }
    if diff == 0x8000 {
        return 16;
    }
    let mag = (diff as i16 as i32).unsigned_abs();
    (32 - mag.leading_zeros()) as u8
}

/// One canonical Huffman table: DHT fields plus the per-symbol codes.
struct HuffTable {
    /// `bits[l]` = number of codes of length `l + 1` (DHT's L1..L16).
    bits: [u8; 16],
    /// Symbols in DHT order (by code length, then value).
    huffval: Vec<u8>,
    /// `codes[symbol]` = (code, length). Length 0 = symbol unused.
    codes: [(u16, u32); 17],
}

impl HuffTable {
    /// Build the optimal length-limited table for a category histogram.
    /// This is the Annex K.2 algorithm as implemented by libjpeg's
    /// `jpeg_gen_optimal_table`: a pseudo-symbol with frequency 1 both
    /// reserves the all-ones codeword (required by JPEG so a run of 1-fill
    /// bits can't decode as a real symbol) and guarantees every real
    /// symbol gets a code even when only one category occurs.
    fn optimal(freq_in: &[u32; 17]) -> Self {
        const RESERVED: usize = 17;
        let mut freq = [0u64; 18];
        for (f, &fi) in freq.iter_mut().zip(freq_in.iter()) {
            *f = fi as u64;
        }
        freq[RESERVED] = 1;

        let mut codesize = [0usize; 18];
        let mut others: [isize; 18] = [-1; 18];

        // Repeatedly merge the two least-frequent trees. Ties pick the
        // higher symbol so the reserved pseudo-symbol sinks to the
        // deepest, all-ones leaf.
        loop {
            let mut c1: isize = -1;
            let mut v = u64::MAX;
            for (i, &f) in freq.iter().enumerate() {
                if f > 0 && f <= v {
                    v = f;
                    c1 = i as isize;
                }
            }
            let mut c2: isize = -1;
            v = u64::MAX;
            for (i, &f) in freq.iter().enumerate() {
                if f > 0 && f <= v && i as isize != c1 {
                    v = f;
                    c2 = i as isize;
                }
            }
            if c2 < 0 {
                break;
            }
            let (c1, c2) = (c1 as usize, c2 as usize);
            freq[c1] += freq[c2];
            freq[c2] = 0;
            let mut p = c1;
            codesize[p] += 1;
            while others[p] >= 0 {
                p = others[p] as usize;
                codesize[p] += 1;
            }
            others[p] = c2 as isize;
            let mut p = c2;
            codesize[p] += 1;
            while others[p] >= 0 {
                p = others[p] as usize;
                codesize[p] += 1;
            }
        }

        // Count codes per length. 18 symbols bound the depth at 17, but
        // run the standard Figure K.3 length-limiting anyway.
        let mut bits32 = [0i32; 33];
        for (i, &cs) in codesize.iter().enumerate() {
            if cs > 0 {
                debug_assert!(cs <= 32, "codesize overflow for symbol {i}");
                bits32[cs] += 1;
            }
        }
        for i in (17..=32).rev() {
            while bits32[i] > 0 {
                let mut j = i - 2;
                while bits32[j] == 0 {
                    j -= 1;
                }
                bits32[i] -= 2;
                bits32[i - 1] += 1;
                bits32[j + 1] += 2;
                bits32[j] -= 1;
            }
        }
        // Drop the reserved pseudo-symbol: it owns the longest (all-ones)
        // code, so removing one count at the deepest level excises it.
        for i in (1..=16).rev() {
            if bits32[i] > 0 {
                bits32[i] -= 1;
                break;
            }
        }

        let mut bits = [0u8; 16];
        for l in 1..=16 {
            bits[l - 1] = bits32[l] as u8;
        }
        // Symbols in canonical DHT order. Lengths beyond 16 were folded
        // into shorter ones above, so clamp each real symbol's length to
        // its post-adjustment group by re-deriving from the sorted order.
        let mut by_len: Vec<(usize, usize)> = codesize[..17]
            .iter()
            .enumerate()
            .filter(|(_, &cs)| cs > 0)
            .map(|(sym, &cs)| (cs, sym))
            .collect();
        by_len.sort_unstable();
        let huffval: Vec<u8> = by_len.iter().map(|&(_, sym)| sym as u8).collect();

        // Assign canonical codes from BITS + HUFFVAL (Annex C).
        let mut codes = [(0u16, 0u32); 17];
        let mut code: u32 = 0;
        let mut k = 0usize;
        for (l, &count) in bits.iter().enumerate() {
            for _ in 0..count {
                codes[huffval[k] as usize] = (code as u16, l as u32 + 1);
                code += 1;
                k += 1;
            }
            code <<= 1;
        }
        debug_assert_eq!(k, huffval.len());

        HuffTable {
            bits,
            huffval,
            codes,
        }
    }
}

fn write_dht(out: &mut Vec<u8>, t: &HuffTable) {
    let len = 2 + 1 + 16 + t.huffval.len();
    out.extend_from_slice(&[0xFF, 0xC4]); // DHT
    out.extend_from_slice(&(len as u16).to_be_bytes());
    out.push(0x00); // Tc=0 (DC/lossless), Th=0
    out.extend_from_slice(&t.bits);
    out.extend_from_slice(&t.huffval);
}

fn write_sof3(out: &mut Vec<u8>, width: u16, height: u16, components: u8) {
    let len = 8 + 3 * components as usize;
    out.extend_from_slice(&[0xFF, 0xC3]); // SOF3 (lossless, Huffman)
    out.extend_from_slice(&(len as u16).to_be_bytes());
    out.push(16); // sample precision
    out.extend_from_slice(&height.to_be_bytes());
    out.extend_from_slice(&width.to_be_bytes());
    out.push(components);
    for c in 0..components {
        out.push(c + 1); // component id
        out.push(0x11); // H=V=1
        out.push(0); // Tq (unused in lossless)
    }
}

fn write_sos(out: &mut Vec<u8>, components: u8) {
    let len = 6 + 2 * components as usize;
    out.extend_from_slice(&[0xFF, 0xDA]); // SOS
    out.extend_from_slice(&(len as u16).to_be_bytes());
    out.push(components);
    for c in 0..components {
        out.push(c + 1); // component selector
        out.push(0x00); // Td=0, Ta=0
    }
    out.push(1); // Ss: predictor 1 (Ra)
    out.push(0); // Se
    out.push(0); // Ah=0, Al=0 (no point transform)
}

/// MSB-first bit writer with JPEG 0xFF byte stuffing.
struct BitWriter {
    out: Vec<u8>,
    acc: u32,
    nbits: u32,
}

impl BitWriter {
    fn new(out: Vec<u8>) -> Self {
        BitWriter {
            out,
            acc: 0,
            nbits: 0,
        }
    }

    /// Append the low `n` bits of `bits` (n <= 16).
    fn put(&mut self, bits: u32, n: u32) {
        debug_assert!(n <= 16);
        self.acc = (self.acc << n) | (bits & ((1u32 << n) - 1));
        self.nbits += n;
        while self.nbits >= 8 {
            let byte = ((self.acc >> (self.nbits - 8)) & 0xFF) as u8;
            self.out.push(byte);
            if byte == 0xFF {
                self.out.push(0x00);
            }
            self.nbits -= 8;
        }
    }

    /// Pad the final byte with 1-bits (decoders treat the fill as a
    /// non-code thanks to the reserved all-ones codeword) and return the
    /// underlying buffer.
    fn finish(mut self) -> Vec<u8> {
        if self.nbits > 0 {
            let pad = 8 - self.nbits;
            self.put((1 << pad) - 1, pad);
        }
        self.out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal lossless-JPEG decoder for round-trip tests. Parses the
    /// markers the encoder emits (single DHT, SOF3, single interleaved
    /// scan, predictor 1, no restarts) and reconstructs the raster.
    fn decode(stream: &[u8]) -> (Vec<u16>, usize, usize, usize) {
        assert_eq!(&stream[..2], &[0xFF, 0xD8], "missing SOI");
        let mut pos = 2;
        let mut bits = [0u8; 16];
        let mut huffval: Vec<u8> = Vec::new();
        let (mut width, mut height, mut ncomp) = (0usize, 0usize, 0usize);
        let predictor;
        loop {
            assert_eq!(stream[pos], 0xFF, "expected marker at {pos}");
            let marker = stream[pos + 1];
            let seg_len = u16::from_be_bytes([stream[pos + 2], stream[pos + 3]]) as usize;
            let seg = &stream[pos + 4..pos + 2 + seg_len];
            match marker {
                0xC4 => {
                    assert_eq!(seg[0], 0x00, "table class/id");
                    bits.copy_from_slice(&seg[1..17]);
                    huffval = seg[17..].to_vec();
                }
                0xC3 => {
                    assert_eq!(seg[0], 16, "precision");
                    height = u16::from_be_bytes([seg[1], seg[2]]) as usize;
                    width = u16::from_be_bytes([seg[3], seg[4]]) as usize;
                    ncomp = seg[5] as usize;
                }
                0xDA => {
                    let ns = seg[0] as usize;
                    assert_eq!(ns, ncomp);
                    predictor = seg[1 + 2 * ns];
                    pos += 2 + seg_len;
                    break;
                }
                m => panic!("unexpected marker FF{m:02X}"),
            }
            pos += 2 + seg_len;
        }
        assert_eq!(predictor, 1);

        // Rebuild (length, symbol) pairs in canonical order.
        let mut code_to_sym = std::collections::HashMap::new();
        let mut code: u32 = 0;
        let mut k = 0;
        for (l, &count) in bits.iter().enumerate() {
            for _ in 0..count {
                code_to_sym.insert((l as u32 + 1, code), huffval[k]);
                code += 1;
                k += 1;
            }
            code <<= 1;
        }

        // Bit reader with FF00 unstuffing.
        let mut bytes: Vec<u8> = Vec::new();
        let mut i = pos;
        while i < stream.len() {
            if stream[i] == 0xFF {
                if stream[i + 1] == 0x00 {
                    bytes.push(0xFF);
                    i += 2;
                    continue;
                }
                assert_eq!(stream[i + 1], 0xD9, "expected EOI");
                break;
            }
            bytes.push(stream[i]);
            i += 1;
        }
        let mut bitpos = 0usize;
        let read_bit = |bp: &mut usize| -> u32 {
            let b = (bytes[*bp / 8] >> (7 - *bp % 8)) & 1;
            *bp += 1;
            b as u32
        };

        let mut out = vec![0u16; width * height * ncomp];
        let row_samples = width * ncomp;
        for row in 0..height {
            for col in 0..width {
                for c in 0..ncomp {
                    // Huffman-decode the category.
                    let (mut len, mut code) = (0u32, 0u32);
                    let ssss = loop {
                        code = (code << 1) | read_bit(&mut bitpos);
                        len += 1;
                        assert!(len <= 16, "bad code");
                        if let Some(&sym) = code_to_sym.get(&(len, code)) {
                            break sym;
                        }
                    };
                    // Extra bits → signed difference.
                    let diff: i32 = match ssss {
                        0 => 0,
                        16 => 32768,
                        _ => {
                            let mut v: i32 = 0;
                            for _ in 0..ssss {
                                v = (v << 1) | read_bit(&mut bitpos) as i32;
                            }
                            if v < (1 << (ssss - 1)) {
                                v - (1 << ssss) + 1
                            } else {
                                v
                            }
                        }
                    };
                    let off = row * row_samples + col * ncomp + c;
                    let pred = if col > 0 {
                        out[off - ncomp]
                    } else if row > 0 {
                        out[off - row_samples]
                    } else {
                        1 << 15
                    };
                    out[off] = pred.wrapping_add(diff as u16);
                }
            }
        }
        (out, width, height, ncomp)
    }

    fn round_trip(samples: &[u16], w: usize, h: usize, c: usize) {
        let stream = encode(samples, w, h, c);
        let (decoded, dw, dh, dc) = decode(&stream);
        assert_eq!((dw, dh, dc), (w, h, c));
        assert_eq!(decoded, samples, "{w}x{h}x{c} round trip");
    }

    #[test]
    fn category_boundaries() {
        assert_eq!(category(0), 0);
        assert_eq!(category(1), 1);
        assert_eq!(category(0xFFFF), 1); // -1
        assert_eq!(category(2), 2);
        assert_eq!(category(0xFFFE), 2); // -2
        assert_eq!(category(32767), 15);
        assert_eq!(category(0x8001), 15); // -32767
        assert_eq!(category(0x8000), 16); // ±32768 special case
    }

    #[test]
    fn bit_writer_stuffs_ff() {
        let mut bw = BitWriter::new(Vec::new());
        bw.put(0xFF, 8);
        bw.put(0xAB, 8);
        assert_eq!(bw.finish(), vec![0xFF, 0x00, 0xAB]);
    }

    #[test]
    fn round_trips_gradient_three_component() {
        let (w, h, c) = (31, 17, 3);
        let mut data = Vec::with_capacity(w * h * c);
        for r in 0..h {
            for col in 0..w {
                for ch in 0..c {
                    data.push((r * 1000 + col * 13 + ch * 7) as u16);
                }
            }
        }
        round_trip(&data, w, h, c);
    }

    #[test]
    fn round_trips_pseudorandom_full_range() {
        // Deterministic LCG noise spanning the full u16 range exercises
        // every category, including the no-extra-bits 16.
        let (w, h, c) = (64, 16, 3);
        let mut state = 0x1234_5678u32;
        let data: Vec<u16> = (0..w * h * c)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                (state >> 16) as u16
            })
            .collect();
        round_trip(&data, w, h, c);
    }

    #[test]
    fn round_trips_extreme_alternation() {
        // 0 ↔ 65535 alternation forces ±32767/±32768-class diffs.
        let (w, h, c) = (16, 4, 3);
        let data: Vec<u16> = (0..w * h * c)
            .map(|i| if i % 2 == 0 { 0 } else { 65535 })
            .collect();
        round_trip(&data, w, h, c);
    }

    #[test]
    fn round_trips_constant_image() {
        // Single-category histogram: the reserved-symbol trick must still
        // yield a decodable 1-entry table.
        round_trip(&vec![512u16; 24 * 3 * 3], 24, 3, 3);
    }

    #[test]
    fn round_trips_single_component_and_single_pixel() {
        round_trip(&[40000u16], 1, 1, 1);
        let data: Vec<u16> = (0..63u16).map(|i| i * 997).collect();
        round_trip(&data, 21, 3, 1);
    }

    #[test]
    fn stream_is_framed_correctly() {
        let stream = encode(&[1, 2, 3, 4, 5, 6], 2, 1, 3);
        assert_eq!(&stream[..2], &[0xFF, 0xD8]);
        assert_eq!(&stream[stream.len() - 2..], &[0xFF, 0xD9]);
        // SOF3 marker present.
        assert!(stream.windows(2).any(|w| w == [0xFF, 0xC3]));
    }
}
