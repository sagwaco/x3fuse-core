//! Tests `Reader::from_bytes` parity against `Reader::open` on a corpus
//! file. The buffer-based path is what makes the wasm32 build runnable
//! in environments without a host filesystem (browser, JNI, …); the
//! same code path runs on host via libc's `fmemopen`, so this test
//! gives us host coverage of every byte the wasm version will execute.

use std::path::PathBuf;

fn corpus_dir() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("X3F_TEST_FILES") {
        let p = PathBuf::from(p);
        return p.is_dir().then_some(p);
    }
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest.parent()?.parent()?.to_path_buf();
    let candidate = workspace_root.join("x3f_test_files");
    candidate.is_dir().then_some(candidate)
}

/// First X3F file (any extension `.x3f` / `.X3F`) under the corpus, or
/// `None` if the corpus isn't on disk. Tests that depend on this should
/// `?`-fall back into a skip-with-notice rather than failing.
fn pick_corpus_file() -> Option<PathBuf> {
    let dir = corpus_dir()?;
    let mut entries: Vec<PathBuf> = std::fs::read_dir(&dir)
        .ok()?
        .filter_map(|r| r.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| e.eq_ignore_ascii_case("x3f"))
        })
        .collect();
    entries.sort();
    entries.into_iter().next()
}

#[test]
fn from_bytes_matches_open_header_version() {
    let Some(path) = pick_corpus_file() else {
        eprintln!(
            "  (skip: no .X3F corpus file found; set X3F_TEST_FILES to a dir containing one)"
        );
        return;
    };

    // Reference: header_version via the file-path constructor.
    let opened = x3f_core::Reader::open(&path).expect("Reader::open should succeed");
    let v_open = opened.header_version();
    drop(opened);

    // Subject: header_version via the buffer constructor.
    let bytes = std::fs::read(&path).expect("read corpus file");
    let from_buf = x3f_core::Reader::from_bytes(&bytes).expect("Reader::from_bytes should succeed");
    let v_buf = from_buf.header_version();

    assert_eq!(
        v_buf,
        v_open,
        "from_bytes and open disagree on header.version for {}",
        path.display()
    );
}

#[test]
fn from_bytes_metadata_dump_succeeds() {
    let Some(path) = pick_corpus_file() else {
        eprintln!("  (skip: no .X3F corpus file found)");
        return;
    };

    let bytes = std::fs::read(&path).expect("read corpus file");
    let mut r = x3f_core::Reader::from_bytes(&bytes).expect("Reader::from_bytes");

    // Force the loaders to read further into the file via the FILE*
    // cursor — exercises fseek / fread / ftell / fgetc through the
    // buffer-backed code path.
    r.load_camf().expect("load_camf");
    r.load_property_list().expect("load_property_list");

    let tmp = std::env::temp_dir().join(format!("x3f-core-from-bytes-{}.txt", std::process::id()));
    r.dump_meta(&tmp).expect("dump_meta");
    let s = std::fs::read_to_string(&tmp).expect("read meta dump");
    assert!(
        s.contains("BEGIN: file header meta data"),
        "metadata dump missing expected header marker: {} bytes, prefix `{:.80}`",
        s.len(),
        s
    );
}

#[test]
fn from_bytes_rejects_garbage() {
    // Anything that isn't `FOVb` at offset 0 is not a valid X3F.
    let mut garbage = vec![0xAAu8; 1024];
    garbage[0..4].copy_from_slice(b"NOPE");
    let r = x3f_core::Reader::from_bytes(&garbage);
    assert!(r.is_err(), "expected parse error for garbage input");
}

#[test]
fn from_bytes_rejects_empty() {
    let r = x3f_core::Reader::from_bytes(&[]);
    assert!(r.is_err(), "expected error for empty input");
}
