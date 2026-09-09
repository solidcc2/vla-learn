import pytest

from cloud.prepare import main


def test_prepare_only_builds_local_code_directory(tmp_path):
    output = tmp_path / "release"
    main(["--code-only", "--output-dir", str(output)])
    assert {p.name for p in output.iterdir()} == {"code"}
    code = output / "code"
    assert (code / "train.py").is_file()
    assert (code / "operators/soft_medoid_pool.py").is_file()
    assert (code / "SHA256SUMS").is_file()
    assert not (code / "cloud/prepare.py").exists()
    with pytest.raises(SystemExit):
        main(["--code-only", "--output-dir", str(output)])
