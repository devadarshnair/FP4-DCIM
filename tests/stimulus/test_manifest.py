from cimfp4.stimulus.manifest import verify_manifest, write_manifest


def test_manifest_roundtrip_and_tamper_detection(tmp_path):
    files = []
    for i in range(3):
        f = tmp_path / "sub" / f"f{i}.bin"
        f.parent.mkdir(exist_ok=True)
        f.write_bytes(bytes([i]) * (100 + i))
        files.append(f)
    write_manifest(tmp_path, {"model": "x@rev"}, files)
    assert verify_manifest(tmp_path) == []
    files[1].write_bytes(b"tampered")
    files[2].unlink()
    problems = verify_manifest(tmp_path)
    assert any("mismatch" in p and "f1" in p for p in problems)
    assert any("missing" in p and "f2" in p for p in problems)
