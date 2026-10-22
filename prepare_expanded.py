"""Audit all Garbage265 and the original 20k, deduplicate, then split by group."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import random
import zipfile
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageDraw
from scipy.fft import dctn

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
OUT = DATA / "expanded"
OUT.mkdir(exist_ok=True)

def extract(archive_path, destination):
    marker = destination / (".extracted-" + archive_path.name)
    if marker.exists():
        return
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            if not (destination / member.filename).resolve().is_relative_to(destination.resolve()):
                raise ValueError("Unsafe archive path")
        archive.extractall(destination)
    marker.touch()

def map_label(name):
    major, item = name.split("-", 1)
    if item in {"电池板", "血压计"}:
        return "ewaste"
    if item in {"订书机", "锅盖", "过滤网", "钥匙"}:
        return "mixed_items"
    if item in {"茶壶碎片", "破碎陶瓷", "陶瓷"}:
        return "ceramics"
    if item in {"电池", "纽扣电池", "蓄电池", "充电宝"}:
        return "battery"
    if item in {"药膏", "药片", "药品包装", "药瓶"}:
        return "medicines"
    if item == "灯":
        return "light_bulbs"
    if major == "有害垃圾" or item in {"燃气瓶", "灭火器", "打火机"}:
        return "household_problem_waste"
    if major == "厨余垃圾":
        return "mixed_items" if item in {"秸秆杯", "秸秆碗"} else "food_organics"
    if "玻璃" in item or item in {"酒瓶", "鱼缸"}:
        return "glass"
    if item in {"不锈钢制品", "磁铁", "刀", "钉子", "订书机", "锅", "锅盖", "过滤网", "金属罐", "金属制品", "笼子", "铝制用品", "奶粉罐", "镊子", "铅球", "铁丝球", "钥匙", "图钉", "U型回形针"}:
        return "metal"
    if item in {"保鲜膜内芯", "纸箱"}:
        return "cardboard"
    if item in {"档案袋", "地铁票", "登机牌", "吊牌", "购物纸袋", "卡片", "快递纸袋", "日历", "书", "信封", "纸牌", "纸制品", "便利贴", "彩票", "车票", "电影票"}:
        return "paper"
    if item in {"果冻杯", "泡沫板", "食用油桶", "塑料碗盆", "塑料制品", "洗发水瓶", "瑜伽球", "塑料袋"}:
        return "plastic"
    if item in {"布制品", "耳套", "裤子", "帽子", "裙子", "毯子", "袜子", "衣服", "枕头", "毛巾", "厨房抹布", "眼镜布"}:
        return "textile_trash"
    if item in {"木雕", "木棍", "木桶", "木制切菜板", "木质锅铲", "木质梳子", "串串竹签", "牙签", "竹筷"}:
        return "wood"
    if item in {"插头电线", "插线板", "充电头", "充电线", "充电牙刷", "吹风机", "打印机", "电磁炉", "电动卷发棒", "电动剃须刀", "电饭煲", "电风扇", "电话", "电路板", "显示器", "电视机", "电熨斗", "电子秤", "豆浆机", "耳机", "话筒", "计算器", "键盘", "空气加湿器", "空气净化器", "路由器", "闹铃", "燃气灶", "扫地机器人", "收音机", "手表", "手电筒", "手机", "台灯", "太阳能热水器", "体重秤", "网卡", "遥控器", "音响", "电蚊香"}:
        return "ewaste"
    if major == "其他垃圾":
        return "trash"
    return "mixed_items"

def audit(row):
    path = Path(row["path"])
    try:
        content = path.read_bytes()
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            image.load()
            if min(image.size) < 64:
                return {**row, "status": "too_small"}
            pixels = np.asarray(image.resize((128, 128)))
            gray = image.convert("L").resize((32, 32))
            variants = []
            for transform in (None, Image.Transpose.ROTATE_90, Image.Transpose.ROTATE_180, Image.Transpose.ROTATE_270):
                rotated = gray if transform is None else gray.transpose(transform)
                for candidate in (rotated, ImageOps.mirror(rotated)):
                    values = dctn(np.asarray(candidate, dtype=float), norm="ortho")[:8, :8].ravel()[1:]
                    variants.append(int.from_bytes(np.packbits(values > np.median(values)).tobytes(), "big"))
        return {**row, "sha256": hashlib.sha256(content).hexdigest(), "pixel_hash": hashlib.sha256(pixels.tobytes()).hexdigest(), "phash": str(min(variants)), "status": "valid"}
    except Exception as error:
        return {**row, "status": "invalid", "error": str(error)}

def main():
    for name in ("train", "val"):
        extract(DATA / "garbage265_download" / (name + ".zip"), DATA / "garbage265")
    metadata = DATA / "garbage265_metadata"
    names = (metadata / "classname.txt").read_text().splitlines()
    mapping = {index: {"source_label": name, "label": map_label(name)} for index, name in enumerate(names)}
    (OUT / "category_mapping.json").write_text(json.dumps(mapping, ensure_ascii=False, indent=2))
    rows = []
    old = pd.read_csv(BASE / "saved_results/split_manifest.csv")
    for row in old.itertuples():
        path = DATA / Path(row.path).relative_to("/content")
        assert path.is_file(), path
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row.sha256
        rows.append({"dataset": row.dataset, "label": row.label, "fine_label": row.label, "path": str(path), "family": ""})
    for filename in ("train.csv", "val.csv"):
        frame = pd.read_csv(metadata / filename)
        for relative, index in frame.itertuples(index=False, name=None):
            path = DATA / "garbage265" / relative
            assert path.is_file(), path
            label = mapping[int(index)]
            rows.append({"dataset": "garbage265", "label": label["label"], "fine_label": label["source_label"], "path": str(path), "family": "garbage265:" + path.stem.split("_")[0]})
    print("Raw candidates:", len(rows), flush=True)
    cache = OUT / "raw_audit.csv"
    if cache.exists():
        audited = pd.read_csv(cache, dtype={"phash": "string"}).to_dict("records")
        assert len(audited) == len(rows)
    else:
        audited = []
        with ThreadPoolExecutor(max_workers=10) as pool:
            for result in pool.map(audit, rows):
                audited.append(result)
                if len(audited) % 10000 == 0:
                    print("Audited:", len(audited), flush=True)
        pd.DataFrame(audited).to_csv(cache, index=False)
    frame = pd.DataFrame(audited)
    current_labels = {row["path"]: row["label"] for row in rows}
    frame["label"] = frame.path.map(current_labels)
    rejected = {
        "RealWaste/realwaste-main/RealWaste/Cardboard/Cardboard_414.jpg": "Crushed beverage can, not cardboard",
        "RealWaste/realwaste-main/RealWaste/Cardboard/Cardboard_364.jpg": "Crushed beverage can, not cardboard",
        "RealWaste/realwaste-main/RealWaste/Cardboard/Cardboard_249.jpg": "Foil wrapper, not cardboard",
        "garbage265/train/146/c873832ee9ee_50.jpg": "Street scene, not a visible plastic item",
    }
    (OUT / "manual_rejections.json").write_text(json.dumps(rejected, indent=2))
    frame.loc[frame.path.isin([str(DATA / path) for path in rejected]), "status"] = "manual_reject"
    valid = frame[frame.status == "valid"].reset_index(drop=True)
    parent = list(range(len(valid)))
    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index
    def merge(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[b] = a
    for key in ("sha256", "pixel_hash", "family"):
        for value, indexes in valid.groupby(key).groups.items():
            if key == "family" and (not value or pd.isna(value)):
                continue
            indexes = list(indexes)
            for index in indexes[1:]:
                merge(indexes[0], index)
    buckets = {}
    hashes = [int(value) for value in valid.phash]
    for index, value in enumerate(hashes):
        candidates = set()
        for band in range(3):
            key = (band, (value >> (band * 22)) & ((1 << 22) - 1))
            candidates.update(buckets.get(key, [])[-300:])
        for other in candidates:
            if (value ^ hashes[other]).bit_count() <= 2:
                merge(index, other)
        for band in range(3):
            key = (band, (value >> (band * 22)) & ((1 << 22) - 1))
            buckets.setdefault(key, []).append(index)
    valid["group_id"] = [find(i) for i in range(len(valid))]
    conflicts = valid.groupby("group_id").label.nunique()
    conflict_ids = set(conflicts[conflicts > 1].index)
    valid[valid.group_id.isin(conflict_ids)].to_csv(OUT / "conflicting_labels_excluded.csv", index=False)
    clean = valid[~valid.group_id.isin(conflict_ids)].drop_duplicates("sha256").drop_duplicates("pixel_hash").copy()
    rng = random.Random(42)
    assignments = {}
    for label, group in clean.groupby("label", sort=True):
        ids = sorted(group.group_id.unique())
        rng.shuffle(ids)
        a, b = round(len(ids) * 0.8), round(len(ids) * 0.9)
        for split, subset in (("train", ids[:a]), ("validation", ids[a:b]), ("test", ids[b:])):
            assignments.update({identifier: split for identifier in subset})
    clean["split"] = clean.group_id.map(assignments)
    assert clean.sha256.is_unique and clean.pixel_hash.is_unique
    assert clean.groupby("group_id").split.nunique().max() == 1
    clean.to_csv(OUT / "split_manifest.csv", index=False)
    labels = sorted(clean.label.unique())
    (OUT / "class_names.json").write_text(json.dumps(labels, indent=2))
    counts = pd.crosstab(clean.label, clean.split)
    counts.to_csv(OUT / "class_counts.csv")
    report = {"raw_candidate_images": len(rows), "garbage265_metadata_images": len(rows) - 20000, "original_images": 20000, "valid_decoded": len(valid), "retained_images": len(clean), "split_counts": clean.groupby("split").size().to_dict(), "class_counts": clean.groupby("label").size().to_dict(), "conflicting_groups_removed": len(conflict_ids), "duplicate_group_cross_split": False, "note": "Source taxonomies are retained as fine_label. Chinese disposal categories are not used as Sydney disposal rules. Near-duplicate filtering is heuristic, not a proof of independence."}
    (OUT / "dataset_verification.json").write_text(json.dumps(report, indent=2))
    canvas = Image.new("RGB", (960, 150 * len(labels)), "white")
    draw = ImageDraw.Draw(canvas)
    for row_index, label in enumerate(labels):
        examples = clean[clean.label == label].sample(n=min(8, int((clean.label == label).sum())), random_state=42)
        for col, row in enumerate(examples.itertuples()):
            with Image.open(row.path) as source:
                canvas.paste(ImageOps.contain(source.convert("RGB"), (115, 115)), (col * 120, row_index * 150 + 25))
        draw.text((4, row_index * 150 + 4), label, fill="black")
    canvas.save(OUT / "class_review.jpg")
    print(json.dumps(report, indent=2), flush=True)

if __name__ == "__main__":
    main()
