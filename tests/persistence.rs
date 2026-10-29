#[test]
fn portable_atomic_replacement_preserves_unicode_and_leaves_no_temps() {
    let root = tempfile::tempdir().unwrap();
    let directory = root.path().join("ଓଡ଼ିଆ space");
    std::fs::create_dir(&directory).unwrap();
    let path = directory.join("run.json");
    for text in ["first", "ଓଡ଼ିଆ", "final"] {
        rag_eval::persistence::atomic_write(&path, text.as_bytes()).unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), text);
    }
    assert_eq!(std::fs::read_dir(&directory).unwrap().count(), 1);
    rag_eval::persistence::sync_file(&path).unwrap();
    rag_eval::persistence::sync_directory(&directory).unwrap();
}

#[test]
fn failed_atomic_replacement_cleans_up_only_its_temporary_file() {
    let root = tempfile::tempdir().unwrap();
    let target = root.path().join("directory");
    std::fs::create_dir(&target).unwrap();
    assert!(rag_eval::persistence::atomic_write(&target, b"data").is_err());
    assert!(target.is_dir());
    assert_eq!(std::fs::read_dir(root.path()).unwrap().count(), 1);
}
