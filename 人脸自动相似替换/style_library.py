"""本地摄影风格库：把不统一的 Markdown 压缩成有限、去身份的画面约束。"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re


MAX_STYLE_CONSTRAINTS = 8


class NotAStyleDocument(ValueError):
    """目录中的已知说明文档，不属于可选摄影风格。"""

SECTION_LABELS = (
    "无字底图提示词",
    "更适合直接出图的版本是",
    "通用正向提示词可以这样写",
    "通用正向提示词",
    "正向提示词",
    "提示词可以写成",
)

TEXT_PATTERNS = re.compile(
    r"文字|标题|字母|字体|排版|书法|标语|logo|水印|masthead|headline|cover lines|typography|serif|sans serif",
    re.IGNORECASE,
)
HAIR_FACE_PATTERNS = re.compile(
    r"发型|短发|长发|卷发|直发|寸头|背头|湿发|湿感|黑发|金发|棕发|刘海|头发|眼镜|墨镜|眼睛|眼型|眼距|眉毛|眉形|鼻子|鼻部|鼻梁|鼻翼|嘴巴|嘴唇|嘴型|唇形|人中|下巴|下颌|颧骨|左右脸|不对称|脸型|脸部|面部|五官|皮肤|hair|hairstyle|wavy hair|glasses|sunglasses|facial|face shape|skin texture",
    re.IGNORECASE,
)
EXPRESSION_PATTERNS = re.compile(
    r"表情|眼神|微笑|笑容|忧郁|冷静|冷淡|疏离|强势|疲惫|锐利|克制|自信|expression|gaze|smile|confident|calm",
    re.IGNORECASE,
)
IDENTITY_DIRECTIVE_PATTERNS = re.compile(
    r"身份|本人|参考人物|目标图|人物还原度|真实脸|换脸|替换到|锁脸|identity|same person",
    re.IGNORECASE,
)
IDENTITY_WORDS = re.compile(
    r"(?:年轻|成年|中年|老年)?\s*(?:亚洲|东方|中国)?\s*(?:男性|女性|男人|女人|男士|女士|男孩|女孩|男子|女子)",
    re.IGNORECASE,
)
EMBEDDED_IMAGE = re.compile(r"!\[\[[^\]]+\]\]")
DOCUMENT_NOISE = re.compile(
    r"^(?:实拍|拍摄时|后期|复现|核心公式|风格命名|这类风格|负面提示词|不要|避免|说明|建议)",
    re.IGNORECASE,
)

BUCKET_PATTERNS = {
    "composition": re.compile(
        r"肖像|构图|近景|中景|远景|半身|全身|七分身|特写|景别|机位|低角度|高角度|俯拍|仰拍|镜头|居中|留白|比例|画幅|透视",
        re.IGNORECASE,
    ),
    "pose": re.compile(
        r"站姿|坐姿|蹲|站立|坐在|侧身|回头|低头|抬头|仰头|伸向|插兜|姿态|动作|身体|手臂|手掌|看向|直视|奔跑|骑马",
        re.IGNORECASE,
    ),
    "clothing": re.compile(
        r"穿|西装|衬衫|外套|大衣|毛衣|长裤|短裤|裙|领带|领结|皮鞋|长靴|鞋|服装|夹克|风衣|上衣|内搭|高领|背心|披肩",
        re.IGNORECASE,
    ),
    "scene": re.compile(
        r"背景|摄影棚|影棚|街|城市|山|草地|室内|户外|海边|河岸|咖啡馆|汽车|跑车|墙面|天空|树林|走廊|地面|道具|椅|花田|建筑|公路|湖岸",
        re.IGNORECASE,
    ),
    "lighting": re.compile(
        r"灯光|布光|阳光|光线|硬光|柔光|侧光|逆光|顶光|轮廓光|硬闪|阴影|明暗|曝光|光斑|光束",
        re.IGNORECASE,
    ),
    "color": re.compile(
        r"色调|配色|黑白|灰阶|低饱和|高饱和|单色|冷调|暖调|暖色|冷色|蓝色|红色|橙色|绿色|米色|棕褐|黑金|蓝橙",
        re.IGNORECASE,
    ),
    "texture": re.compile(
        r"摄影|胶片|颗粒|质感|半调|纸张|旧纸|复古|电影|时尚|杂志|广告|海报|极简|超现实|梦幻|纪实|editorial|analog|cinematic",
        re.IGNORECASE,
    ),
}

INCOMPATIBLE_PATTERNS = (
    (re.compile(r"Q版|二次元|动漫|漫画|插画|卡通", re.IGNORECASE), "非真人摄影表现"),
    (re.compile(r"只露出[^，。]*双腿|不露[^，。]*脸|背影", re.IGNORECASE), "脸部不可观测"),
)
LIMITED_PATTERNS = (
    (re.compile(r"小比例|远景|全身|七分身", re.IGNORECASE), "脸部在画面中可能偏小"),
    (re.compile(r"侧脸|低头|仰头|俯拍|仰拍|低机位", re.IGNORECASE), "姿态会降低正脸可观测性"),
    (re.compile(r"群像|双人|多重人像|多人", re.IGNORECASE), "多人构图可能分散身份注意力"),
    (re.compile(r"倒影|镜子|遮住|遮挡|墨镜|眼镜|前景虚化", re.IGNORECASE), "脸部存在遮挡或间接成像"),
)


@dataclass(frozen=True)
class StylePreset:
    preset_id: str
    title: str
    safe_title: str
    source_file: str
    constraints: tuple[str, ...]
    compact_prompt: str
    compatibility: str
    compatibility_label: str
    compatibility_reasons: tuple[str, ...]

    def to_metadata(self) -> dict:
        return {
            "preset_id": self.preset_id,
            "title": self.safe_title,
            "constraints": list(self.constraints),
            "compact_prompt": self.compact_prompt,
            "compatibility": self.compatibility,
            "compatibility_label": self.compatibility_label,
            "compatibility_reasons": list(self.compatibility_reasons),
        }


def default_style_library(root: Path) -> Path:
    return root.parent / "ai人像摄影技能包" / "ai人像摄影提示词"


def _clean_title(path: Path) -> str:
    title = re.sub(r"^\d{3}_", "", path.stem).strip(" ，,。. _")
    return title or path.stem


def _clean_markdown(text: str) -> str:
    text = EMBEDDED_IMAGE.sub("", text)
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    text = re.sub(r"[\t\r]+", " ", text)
    return text.strip()


def _extract_candidate(text: str, fallback: str) -> str:
    cleaned = _clean_markdown(text)
    for label in SECTION_LABELS:
        match = re.search(
            rf"{re.escape(label)}\s*[：:]\s*(.+?)(?=\n\s*\n|\n(?:负面提示词|实拍|拍摄|后期|复现)|\Z)",
            cleaned,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if match:
            return " ".join(match.group(1).split())

    for paragraph in re.split(r"\n\s*\n", cleaned):
        paragraph = " ".join(paragraph.split()).strip()
        if paragraph and not DOCUMENT_NOISE.search(paragraph):
            return paragraph
    return fallback


def _sanitize_segment(segment: str) -> str | None:
    segment = re.sub(r"^[\-—–•*\d.、\s]+", "", segment).strip()
    if not segment or DOCUMENT_NOISE.search(segment) or TEXT_PATTERNS.search(segment):
        return None
    if (
        HAIR_FACE_PATTERNS.search(segment)
        or EXPRESSION_PATTERNS.search(segment)
        or IDENTITY_DIRECTIVE_PATTERNS.search(segment)
    ):
        return None
    segment = IDENTITY_WORDS.sub("人物", segment)
    segment = re.sub(r"\b(?:young|adult|asian|male|female|man|woman)\b", "", segment, flags=re.IGNORECASE)
    segment = re.sub(r"\s{2,}", " ", segment).strip(" ，,。. ")
    if len(segment) < 2:
        return None
    return segment


def _classify_bucket(segment: str) -> str | None:
    for name in ("composition", "pose", "clothing", "scene", "lighting", "color", "texture"):
        if BUCKET_PATTERNS[name].search(segment):
            return name
    return None


def compact_constraints(candidate: str, fallback: str) -> tuple[str, ...]:
    raw_segments = re.split(r"[，,；;。\n]+", candidate)
    safe_segments: list[tuple[str | None, str]] = []
    seen = set()
    for raw in raw_segments:
        segment = _sanitize_segment(raw)
        if not segment or segment.casefold() in seen:
            continue
        seen.add(segment.casefold())
        safe_segments.append((_classify_bucket(segment), segment))

    if not safe_segments and candidate != fallback:
        return compact_constraints(fallback, fallback)

    chosen: list[str] = []
    per_bucket = {name: 0 for name in BUCKET_PATTERNS}
    bucket_limits = {name: (2 if name in {"clothing", "scene"} else 1) for name in BUCKET_PATTERNS}
    for bucket in ("texture", "composition", "pose", "clothing", "scene", "lighting", "color"):
        for item_bucket, segment in safe_segments:
            if item_bucket != bucket or per_bucket[bucket] >= bucket_limits[bucket]:
                continue
            chosen.append(segment)
            per_bucket[bucket] += 1
            if len(chosen) >= MAX_STYLE_CONSTRAINTS:
                return tuple(chosen)

    for _bucket, segment in safe_segments:
        if segment not in chosen:
            chosen.append(segment)
        if len(chosen) >= MAX_STYLE_CONSTRAINTS:
            break
    return tuple(chosen)


def _compatibility(title: str, candidate: str) -> tuple[str, str, tuple[str, ...]]:
    haystack = f"{title}，{candidate}"
    incompatible = tuple(reason for pattern, reason in INCOMPATIBLE_PATTERNS if pattern.search(haystack))
    if incompatible:
        return "incompatible", "非锁脸优先", incompatible
    limited = tuple(reason for pattern, reason in LIMITED_PATTERNS if pattern.search(haystack))
    if limited:
        return "limited", "锁脸受限", limited
    return "preferred", "锁脸优先", ()


def load_style_preset(path: Path) -> StylePreset:
    path = path.expanduser().resolve()
    if not path.is_file() or path.suffix.lower() != ".md":
        raise ValueError(f"不是有效的摄影风格 Markdown：{path}")
    title = _clean_title(path)
    if "人物还原度提示词" in title:
        raise NotAStyleDocument("身份建模文档不是摄影风格")
    raw_text = path.read_text(encoding="utf-8")
    candidate = _extract_candidate(raw_text, title)
    constraints = compact_constraints(candidate, title)
    if not constraints:
        raise ValueError(f"摄影风格没有可用的非身份画面要素：{path.name}")
    compatibility, label, reasons = _compatibility(title, candidate)
    return StylePreset(
        preset_id=hashlib.sha256(path.name.encode("utf-8")).hexdigest()[:12],
        title=title,
        safe_title=constraints[0],
        source_file=path.name,
        constraints=constraints,
        compact_prompt="，".join(constraints),
        compatibility=compatibility,
        compatibility_label=label,
        compatibility_reasons=reasons,
    )


def load_style_library(path: Path) -> list[StylePreset]:
    path = path.expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"系统摄影风格目录不存在：{path}")
    presets = []
    errors = []
    for source in sorted(path.glob("*.md"), key=lambda item: item.name.casefold()):
        try:
            presets.append(load_style_preset(source))
        except NotAStyleDocument:
            continue
        except (OSError, UnicodeError, ValueError) as exc:
            errors.append(f"{source.name}: {exc}")
    if errors:
        raise ValueError("系统摄影风格读取失败：" + "；".join(errors[:5]))
    if not presets:
        raise ValueError("系统摄影风格目录中没有可用风格。")
    return presets


def resolve_library_preset(value: str | Path, library_root: Path) -> StylePreset:
    library_root = library_root.expanduser().resolve()
    requested = Path(value).expanduser()
    path = requested.resolve() if requested.is_absolute() else (library_root / requested).resolve()
    try:
        path.relative_to(library_root)
    except ValueError as exc:
        raise ValueError("系统摄影风格必须来自内置提示词目录。") from exc
    return load_style_preset(path)


def merge_style_prompt(base_prompt: str, preset: StylePreset) -> str:
    base_prompt = base_prompt.strip()
    base_prompt = base_prompt.replace(
        "replacement 模式中的图片2只提供动作、服饰、构图、镜头、滤镜、布光和氛围；portrait 模式没有图片2。",
        "系统摄影风格只提供非身份画面约束。",
    )
    paragraphs = [
        paragraph
        for paragraph in re.split(r"\n\s*\n", base_prompt)
        if not (paragraph.strip().startswith("身份硬约束：") and "图片2" in paragraph)
    ]
    base_prompt = "\n\n".join(paragraphs)
    style_block = (
        "【系统摄影风格】\n"
        f"只执行这些画面约束：{preset.compact_prompt}。\n"
        "图1人物仍是唯一身份来源；不得采用样例人物的性别、年龄、脸型、五官、发型或眼镜。"
        "只生成无字底图，no text, no logo, no watermark。"
    )
    marker = "\n质量硬约束"
    if marker in base_prompt:
        head, tail = base_prompt.split(marker, 1)
        return f"{head.rstrip()}\n\n{style_block}\n\n质量硬约束{tail}".strip()
    return f"{base_prompt}\n\n{style_block}".strip()
