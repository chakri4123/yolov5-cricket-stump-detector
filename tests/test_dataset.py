import yaml

from stump_detector.dataset import filter_label_lines, prepare_dataset, stump_class_ids


def test_finds_stump_class_in_both_spellings():
    assert stump_class_ids(["ball", "Bat", "Stamps"]) == {2}
    assert stump_class_ids({0: "stumps", 1: "bat"}) == {0}
    assert stump_class_ids(["ball", "Bat"]) == set()


def test_filter_keeps_stumps_and_renumbers_them():
    lines = ["0 0.1 0.1 0.1 0.1", "2 0.5 0.5 0.2 0.3", "1 0.9 0.9 0.1 0.1"]
    assert filter_label_lines(lines, {2}) == ["0 0.5 0.5 0.2 0.3"]


def test_prepare_dataset_keeps_empty_frames(tmp_path):
    source = tmp_path / "raw"
    (source / "train" / "images").mkdir(parents=True)
    (source / "train" / "labels").mkdir()
    (source / "valid" / "images").mkdir(parents=True)
    (source / "valid" / "labels").mkdir()
    (source / "data.yaml").write_text(yaml.safe_dump({"names": ["ball", "Bat", "Stamps"]}), encoding="utf-8")

    (source / "train" / "images" / "a.jpg").write_bytes(b"x")
    (source / "train" / "labels" / "a.txt").write_text("2 0.5 0.5 0.1 0.2\n0 0.1 0.1 0.1 0.1", encoding="utf-8")
    (source / "train" / "images" / "b.jpg").write_bytes(b"x")
    (source / "train" / "labels" / "b.txt").write_text("0 0.1 0.1 0.1 0.1", encoding="utf-8")
    (source / "valid" / "images" / "c.jpg").write_bytes(b"x")

    output = tmp_path / "stumps"
    stats = prepare_dataset(source, output)

    assert stats["train"] == {"images": 2, "with_stumps": 1, "boxes": 1}
    assert stats["val"] == {"images": 1, "with_stumps": 0, "boxes": 0}
    assert (output / "labels" / "train" / "a.txt").read_text(encoding="utf-8") == "0 0.5 0.5 0.1 0.2"
    assert (output / "labels" / "train" / "b.txt").read_text(encoding="utf-8") == ""

    data = yaml.safe_load((output / "data.yaml").read_text(encoding="utf-8"))
    assert data["names"] == {0: "stumps"}
    assert "test" not in data


def test_prepare_dataset_fails_without_stump_class(tmp_path):
    source = tmp_path / "raw"
    source.mkdir()
    (source / "data.yaml").write_text(yaml.safe_dump({"names": ["ball"]}), encoding="utf-8")
    try:
        prepare_dataset(source, tmp_path / "out")
    except ValueError as error:
        assert "no stump class" in str(error)
    else:
        raise AssertionError("expected ValueError")
