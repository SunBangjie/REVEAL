#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CATALOG_PATH = Path(__file__).resolve().with_name("scene_families.json")
UNGATED_DEFAULT_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
KNOWN_GATED_STABILITY_MODEL_IDS = {
    "stabilityai/stable-diffusion-3.5-large",
    "stabilityai/stable-diffusion-3.5-medium",
    "stabilityai/stable-diffusion-3-medium-diffusers",
}

NEGATIVE_PROMPT = (
    "blurry, low quality, distorted, watermark, collage, cartoon, anime, painting, illustration, cgi, 3d render, "
    "synthetic, stylized, overprocessed, over-saturated, glossy plastic skin, waxy skin, doll-like face, uncanny, "
    "fake lighting, unrealistic reflections, unrealistic anatomy, airbrushed textures, excessive depth of field blur, "
    "cropped primary object, duplicate primary object, oversized secondary object, oversized ternary object, "
    "hidden ternary object, cluttered foreground, overlapping subjects, occluded objects, inconsistent background, "
    "graphic violence, blood, gore, explicit nudity, weapon focus, riot fire, smoke clouds"
)

SAFETY_BLOCKLIST = (
    "blood",
    "gore",
    "corpse",
    "weapon",
    "gun",
    "knife",
    "explosion",
    "dismembered",
)


@dataclass(frozen=True)
class RoleOption:
    key: str
    label: str
    semantic_type: str
    attributes: tuple[str, ...]
    description: str
    allowed_backgrounds: tuple[str, ...]
    scene_tags: tuple[str, ...]

    @classmethod
    def from_payload(cls, payload):
        return cls(
            key=payload["key"],
            label=payload["label"],
            semantic_type=payload["semantic_type"],
            attributes=tuple(payload.get("attributes", [])),
            description=payload["description"],
            allowed_backgrounds=tuple(payload.get("allowed_backgrounds", [])),
            scene_tags=tuple(payload.get("scene_tags", [])),
        )


@dataclass(frozen=True)
class BackgroundOption:
    key: str
    label: str
    semantic_type: str
    attributes: tuple[str, ...]
    description: str
    scene_tags: tuple[str, ...]

    @classmethod
    def from_payload(cls, payload):
        return cls(
            key=payload["key"],
            label=payload["label"],
            semantic_type=payload["semantic_type"],
            attributes=tuple(payload.get("attributes", [])),
            description=payload["description"],
            scene_tags=tuple(payload.get("scene_tags", [])),
        )


@dataclass(frozen=True)
class SceneFamily:
    key: str
    label: str
    description: str
    primary: tuple[RoleOption, ...]
    secondary: tuple[RoleOption, ...]
    ternary: tuple[RoleOption, ...]
    backgrounds: tuple[BackgroundOption, ...]


@dataclass(frozen=True)
class ScenarioSpec:
    family: SceneFamily
    primary_object: RoleOption
    secondary_object: RoleOption
    ternary_object: RoleOption
    background_scene: BackgroundOption
    coherence_tags: tuple[str, ...]


def ensure_safe_text(label: str, text: str):
    lowered = text.lower()
    matches = [term for term in SAFETY_BLOCKLIST if term in lowered]
    if matches:
        joined = ", ".join(sorted(matches))
        raise ValueError(f"Unsafe term(s) {joined} found in {label}.")


def load_scene_families(catalog_path: Path):
    with open(catalog_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    families = {}
    seen_keys = set()
    for family_key, family_payload in payload.items():
        family = SceneFamily(
            key=family_key,
            label=family_payload["label"],
            description=family_payload["description"],
            primary=tuple(RoleOption.from_payload(item) for item in family_payload["primary"]),
            secondary=tuple(RoleOption.from_payload(item) for item in family_payload["secondary"]),
            ternary=tuple(RoleOption.from_payload(item) for item in family_payload["ternary"]),
            backgrounds=tuple(BackgroundOption.from_payload(item) for item in family_payload["background"]),
        )
        families[family_key] = family

        ensure_safe_text(f"family {family_key}", family.description)
        for role_name, options in (
            ("primary", family.primary),
            ("secondary", family.secondary),
            ("ternary", family.ternary),
        ):
            if not options:
                raise ValueError(f"{family_key} is missing {role_name} options.")
            for option in options:
                if option.key in seen_keys:
                    raise ValueError(f"Duplicate catalog key detected: {option.key}")
                seen_keys.add(option.key)
                ensure_safe_text(f"{family_key}.{role_name}.{option.key}", option.description)
                ensure_safe_text(
                    f"{family_key}.{role_name}.{option.key}.attributes",
                    " ".join(option.attributes),
                )
        if not family.backgrounds:
            raise ValueError(f"{family_key} is missing background options.")
        for background in family.backgrounds:
            if background.key in seen_keys:
                raise ValueError(f"Duplicate catalog key detected: {background.key}")
            seen_keys.add(background.key)
            ensure_safe_text(f"{family_key}.background.{background.key}", background.description)
            ensure_safe_text(
                f"{family_key}.background.{background.key}.attributes",
                " ".join(background.attributes),
            )

    return families


def print_catalog(families):
    for family_key in sorted(families):
        family = families[family_key]
        print(f"{family_key}: {family.label}")
        print(f"  description: {family.description}")
        print(f"  primary: {', '.join(option.key for option in family.primary)}")
        print(f"  secondary: {', '.join(option.key for option in family.secondary)}")
        print(f"  ternary: {', '.join(option.key for option in family.ternary)}")
        print(f"  background: {', '.join(option.key for option in family.backgrounds)}")


def stable_choice(parts, options):
    key = "|".join(parts).encode("utf-8")
    digest = hashlib.sha1(key).digest()
    return options[digest[0] % len(options)]


def role_allowed_in_background(option: RoleOption, background_key: str):
    return not option.allowed_backgrounds or background_key in option.allowed_backgrounds


def shared_scene_tags(*items):
    tag_sets = [set(item.scene_tags) for item in items if item.scene_tags]
    if not tag_sets:
        return tuple()
    shared = set.intersection(*tag_sets)
    return tuple(sorted(shared))


def build_structured_prompt(spec: ScenarioSpec):
    key_parts = [
        spec.family.key,
        spec.primary_object.key,
        spec.secondary_object.key,
        spec.ternary_object.key,
        spec.background_scene.key,
    ]
    secondary_position = stable_choice(
        key_parts + ["secondary"],
        [
            "to the left side of the primary object",
            "to the right side of the primary object",
            "slightly behind the primary object on the left",
            "slightly behind the primary object on the right",
        ],
    )
    ternary_position = stable_choice(
        key_parts + ["ternary"],
        [
            "near the secondary object and fully unobstructed",
            "adjacent to the secondary object on a nearby surface",
            "next to the secondary object at eye-catching but small scale",
            "close to the secondary object with clear visibility and no occlusion",
        ],
    )
    camera_treatment = stable_choice(
        key_parts + ["camera"],
        [
            "eye-level documentary photography",
            "editorial reportage photography",
            "naturalistic professional photography",
            "clean cinematic location photography",
        ],
    )
    background_detail = stable_choice(
        key_parts + ["background"],
        [
            "richly detailed and spatially coherent",
            "clear, realistic, and visually consistent",
            "fully legible with strong environmental context",
            "detailed across the full depth of the frame",
        ],
    )

    prompt = (
        "Ultra-photorealistic natural real-world photograph with exactly four compositional elements and a strict size hierarchy. "
        f"Primary object: {spec.primary_object.description}, large, foreground center, occupying the largest region of the frame, "
        "and serving as the main focal point. "
        f"Secondary object: {spec.secondary_object.description}, clearly visible but significantly smaller than the primary object, "
        f"positioned {secondary_position}. "
        f"Ternary object: {spec.ternary_object.description}, small but still identifiable, positioned {ternary_position}. "
        f"Background scene: {spec.background_scene.description}, in the background, {background_detail}, and consistent with the "
        f"{spec.family.description}. "
        "Maintain decreasing size and salience from primary to secondary to ternary, with clean separation between objects. "
        "The ternary object must remain recognizable and not be occluded. "
        "Exactly one primary object, one secondary object, one ternary object, and one coherent background scene. "
        f"{camera_treatment}, true-to-life colors, physically plausible lighting, natural materials, realistic skin and fabric detail, "
        "documentary-grade realism, authentic perspective, professional composition, no stylization."
    )
    ensure_safe_text("prompt", prompt)
    return prompt


def scenario_to_record(spec: ScenarioSpec):
    return {
        "set_name": "set2",
        "task_attribute": "primary_object",
        "object_class": "",
        "weather": "",
        "time_of_day": "",
        "child_presence": "",
        "scene_family": spec.family.key,
        "scene_family_label": spec.family.label,
        "primary_object": spec.primary_object.key,
        "primary_label": spec.primary_object.label,
        "primary_semantic_type": spec.primary_object.semantic_type,
        "primary_attributes": "|".join(spec.primary_object.attributes),
        "secondary_object": spec.secondary_object.key,
        "secondary_label": spec.secondary_object.label,
        "secondary_semantic_type": spec.secondary_object.semantic_type,
        "secondary_attributes": "|".join(spec.secondary_object.attributes),
        "ternary_object": spec.ternary_object.key,
        "ternary_label": spec.ternary_object.label,
        "ternary_semantic_type": spec.ternary_object.semantic_type,
        "ternary_attributes": "|".join(spec.ternary_object.attributes),
        "background_scene": spec.background_scene.key,
        "background_label": spec.background_scene.label,
        "background_semantic_type": spec.background_scene.semantic_type,
        "background_attributes": "|".join(spec.background_scene.attributes),
        "coherence_tags": "|".join(spec.coherence_tags),
        "keyword_list": "|".join(
            [
                spec.family.key,
                spec.primary_object.key,
                spec.secondary_object.key,
                spec.ternary_object.key,
                spec.background_scene.key,
            ]
        ),
        "prompt": build_structured_prompt(spec),
    }


def selected_family_keys(families, scene_family: str):
    if not scene_family or scene_family == "all":
        return sorted(families)
    if scene_family not in families:
        available = ", ".join(sorted(families))
        raise ValueError(f"Unknown scene family '{scene_family}'. Available families: {available}")
    return [scene_family]


def build_candidate_records(
    families,
    scene_family=None,
    primary_key=None,
    secondary_key=None,
    ternary_key=None,
    background_key=None,
):
    records = []
    for family_key in selected_family_keys(families, scene_family):
        family = families[family_key]
        primaries = [option for option in family.primary if primary_key is None or option.key == primary_key]
        secondaries = [option for option in family.secondary if secondary_key is None or option.key == secondary_key]
        ternaries = [option for option in family.ternary if ternary_key is None or option.key == ternary_key]
        backgrounds = [
            option for option in family.backgrounds if background_key is None or option.key == background_key
        ]

        for background in backgrounds:
            valid_primaries = [
                option for option in primaries if role_allowed_in_background(option, background.key)
            ]
            valid_secondaries = [
                option for option in secondaries if role_allowed_in_background(option, background.key)
            ]
            valid_ternaries = [
                option for option in ternaries if role_allowed_in_background(option, background.key)
            ]

            for primary in valid_primaries:
                for secondary in valid_secondaries:
                    for ternary in valid_ternaries:
                        coherence_tags = shared_scene_tags(primary, secondary, ternary, background)
                        if not coherence_tags:
                            continue
                        spec = ScenarioSpec(
                            family=family,
                            primary_object=primary,
                            secondary_object=secondary,
                            ternary_object=ternary,
                            background_scene=background,
                            coherence_tags=coherence_tags,
                        )
                        records.append(scenario_to_record(spec))
    return records


def score_subset(subset, fields):
    score = 0.0
    for field in fields:
        vals = [row[field] for row in subset]
        uniq = sorted(set(vals))
        counts = {value: vals.count(value) for value in uniq}
        target = len(subset) / len(uniq)
        score += sum((counts[value] - target) ** 2 for value in uniq)
    return score


def balanced_subset(candidates, n, fields, seed=7, trials=5000):
    if n > len(candidates):
        raise ValueError(f"Requested subset size {n} exceeds available candidates {len(candidates)}.")
    if n == len(candidates):
        return sorted(
            candidates,
            key=lambda row: (
                row["scene_family"],
                row["background_scene"],
                row["primary_object"],
                row["secondary_object"],
                row["ternary_object"],
            ),
        )
    rng = random.Random(seed)
    best = None
    best_score = None
    idxs = list(range(len(candidates)))
    for _ in range(trials):
        picked = rng.sample(idxs, n)
        subset = [candidates[i] for i in picked]
        score = score_subset(subset, fields)
        if best_score is None or score < best_score:
            best = subset
            best_score = score
            if score == 0:
                break
    return sorted(
        best,
        key=lambda row: (
            row["scene_family"],
            row["background_scene"],
            row["primary_object"],
            row["secondary_object"],
            row["ternary_object"],
        ),
    )


def build_manifest(
    families,
    base_seed=42,
    unique_prompts=80,
    replicates=2,
    selection_seed=17,
    trials=12000,
    scene_family="all",
    scenario_mode="random",
    primary_key=None,
    secondary_key=None,
    ternary_key=None,
    background_key=None,
):
    candidate_rows = build_candidate_records(
        families=families,
        scene_family=scene_family,
        primary_key=primary_key,
        secondary_key=secondary_key,
        ternary_key=ternary_key,
        background_key=background_key,
    )
    if not candidate_rows:
        raise ValueError("No compatible scene quadruples matched the requested configuration.")

    if scenario_mode == "fixed":
        missing_flags = [
            name
            for name, value in (
                ("primary_key", primary_key),
                ("secondary_key", secondary_key),
                ("ternary_key", ternary_key),
                ("background_key", background_key),
            )
            if not value
        ]
        if missing_flags:
            joined = ", ".join(f"--{flag}" for flag in missing_flags)
            raise ValueError(f"Fixed mode requires explicit role keys: {joined}")
        if len(candidate_rows) != 1:
            raise ValueError(
                f"Fixed mode requires exactly one compatible quadruple, but found {len(candidate_rows)} matches."
            )
        chosen_rows = candidate_rows
    else:
        chosen_rows = balanced_subset(
            candidate_rows,
            n=unique_prompts,
            fields=[
                "scene_family",
                "primary_object",
                "secondary_object",
                "ternary_object",
                "background_scene",
            ],
            seed=selection_seed,
            trials=trials,
        )

    manifest = []
    image_idx = 0
    for row in chosen_rows:
        for rep in range(replicates):
            seed = base_seed + image_idx
            out_name = f"{row['set_name']}_{row['scene_family']}_{image_idx:03d}_seed{seed}.png"
            record = dict(row)
            record["replicate"] = rep
            record["seed"] = seed
            record["output_file"] = out_name
            manifest.append(record)
            image_idx += 1
    return manifest


def write_manifest_csv(manifest, csv_path):
    fieldnames = [
        "set_name",
        "task_attribute",
        "object_class",
        "weather",
        "time_of_day",
        "child_presence",
        "scene_family",
        "scene_family_label",
        "primary_object",
        "primary_label",
        "primary_semantic_type",
        "primary_attributes",
        "secondary_object",
        "secondary_label",
        "secondary_semantic_type",
        "secondary_attributes",
        "ternary_object",
        "ternary_label",
        "ternary_semantic_type",
        "ternary_attributes",
        "background_scene",
        "background_label",
        "background_semantic_type",
        "background_attributes",
        "coherence_tags",
        "keyword_list",
        "replicate",
        "seed",
        "output_file",
        "prompt",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest)


def resolve_runtime_device(device_arg: str):
    import torch

    try:
        requested = torch.device(device_arg)
    except (TypeError, RuntimeError, ValueError) as exc:
        raise ValueError(f"Invalid device '{device_arg}'.") from exc

    if requested.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"Requested device '{device_arg}' but CUDA is not available.")
        device_index = requested.index if requested.index is not None else 0
        device_count = torch.cuda.device_count()
        if device_index >= device_count:
            raise RuntimeError(
                f"Requested device '{device_arg}' but only {device_count} CUDA device(s) are available."
            )
        return f"cuda:{device_index}"
    if requested.type == "cpu":
        return "cpu"
    raise ValueError(f"Unsupported device '{device_arg}'. Use 'cpu' or 'cuda:N'.")


def detect_model_family(model_id: str):
    lowered = model_id.lower()
    if "stable-diffusion-3" in lowered:
        return "sd3"
    if "stable-diffusion-xl" in lowered or "sdxl" in lowered:
        return "sdxl"
    return "sdxl"


def resolve_model_family(model_id: str, model_family: str):
    if model_family == "auto":
        return detect_model_family(model_id)
    return model_family


def is_gated_repo_error(exc: Exception):
    text = str(exc).lower()
    markers = (
        "gated repo",
        "access to model",
        "is restricted",
        "must have access",
        "please log in",
        "authenticated",
    )
    return any(marker in text for marker in markers)


def load_pipeline(model_id: str, offload: bool, device: str, model_family: str, fallback_model_id: str | None = None):
    import torch

    runtime_device = resolve_runtime_device(device)
    use_cuda = runtime_device.startswith("cuda")
    model_family = resolve_model_family(model_id, model_family)

    if model_family == "sd3":
        from diffusers import StableDiffusion3Pipeline

        pipe_cls = StableDiffusion3Pipeline
        torch_dtype = torch.bfloat16 if use_cuda else torch.float32
        pipe_kwargs = {
            "torch_dtype": torch_dtype,
        }
    else:
        from diffusers import StableDiffusionXLPipeline

        pipe_cls = StableDiffusionXLPipeline
        torch_dtype = torch.float16 if use_cuda else torch.float32
        pipe_kwargs = {
            "torch_dtype": torch_dtype,
            "use_safetensors": True,
        }
        if use_cuda:
            pipe_kwargs["variant"] = "fp16"

    try:
        pipe = pipe_cls.from_pretrained(model_id, **pipe_kwargs)
    except Exception as exc:
        if fallback_model_id and fallback_model_id != model_id and is_gated_repo_error(exc):
            fallback_family = resolve_model_family(fallback_model_id, "auto")
            print(
                f"Model '{model_id}' requires Hugging Face access. Falling back to ungated model '{fallback_model_id}'."
            )
            return load_pipeline(
                model_id=fallback_model_id,
                offload=offload,
                device=device,
                model_family=fallback_family,
                fallback_model_id=None,
            )
        raise
    pipe.enable_attention_slicing()

    if offload and use_cuda:
        torch.cuda.set_device(int(runtime_device.split(":")[1]))
        pipe.enable_sequential_cpu_offload()
        generator_device = runtime_device
    elif use_cuda:
        pipe.to(runtime_device)
        generator_device = runtime_device
    else:
        pipe.to("cpu")
        generator_device = "cpu"

    return pipe, generator_device, model_family, model_id


def generate_images(
    manifest,
    output_dir,
    model_id,
    model_family,
    steps,
    guidance,
    height,
    width,
    offload,
    device,
    max_sequence_length,
    fallback_model_id,
):
    import torch

    pipe, generator_device, model_family, resolved_model_id = load_pipeline(
        model_id=model_id,
        offload=offload,
        device=device,
        model_family=model_family,
        fallback_model_id=fallback_model_id,
    )
    print(f"Using generation device: {generator_device}")
    print(f"Using model family: {model_family}")
    print(f"Using model id: {resolved_model_id}")

    for i, row in enumerate(manifest, start=1):
        generator = torch.Generator(device=generator_device).manual_seed(int(row["seed"]))
        pipe_kwargs = {
            "prompt": row["prompt"],
            "negative_prompt": NEGATIVE_PROMPT,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "height": height,
            "width": width,
            "generator": generator,
        }
        if model_family == "sd3":
            pipe_kwargs["max_sequence_length"] = max_sequence_length

        image = pipe(**pipe_kwargs).images[0]
        image.save(output_dir / row["output_file"])
        if i % 10 == 0 or i == len(manifest):
            print(f"Generated {i}/{len(manifest)} images")


def choose_default_model_id():
    return UNGATED_DEFAULT_MODEL_ID


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="./sdxl_eval_set2_four_roles")
    parser.add_argument("--catalog_path", type=str, default=str(DEFAULT_CATALOG_PATH))
    parser.add_argument("--scene_family", type=str, default="all", help="Scene family from the catalog, or 'all'.")
    parser.add_argument(
        "--scenario_mode",
        type=str,
        choices=["random", "fixed"],
        default="random",
        help="Use random balanced sampling or one explicit fixed quadruple.",
    )
    parser.add_argument("--primary_key", type=str, default=None, help="Primary object key to pin or filter.")
    parser.add_argument("--secondary_key", type=str, default=None, help="Secondary object key to pin or filter.")
    parser.add_argument("--ternary_key", type=str, default=None, help="Ternary object key to pin or filter.")
    parser.add_argument("--background_key", type=str, default=None, help="Background scene key to pin or filter.")
    parser.add_argument("--list_catalog", action="store_true", help="Print available families and role keys, then exit.")
    parser.add_argument(
        "--model_id",
        type=str,
        default=choose_default_model_id(),
        help="Model repo id or local path. The default is the latest official ungated Stability checkpoint I could verify for no-login local use.",
    )
    parser.add_argument(
        "--model_family",
        type=str,
        choices=["auto", "sd3", "sdxl"],
        default="auto",
        help="Pipeline family. Use 'sd3' when loading Stable Diffusion 3.x from a local path with a custom folder name.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:1",
        help="Runtime device for local generation, for example 'cuda:1' or 'cpu'.",
    )
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=7.5)
    parser.add_argument("--max_sequence_length", type=int, default=256)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument(
        "--unique_prompts",
        type=int,
        default=80,
        help="Number of unique random scene quadruples to keep when scenario_mode=random.",
    )
    parser.add_argument("--replicates", type=int, default=2, help="How many image seeds to render per unique prompt.")
    parser.add_argument(
        "--selection_seed",
        type=int,
        default=17,
        help="Seed used when selecting the balanced subset of random prompts.",
    )
    parser.add_argument(
        "--subset_trials",
        type=int,
        default=12000,
        help="Number of random subset trials when balancing prompt coverage.",
    )
    parser.add_argument("--manifest_only", action="store_true")
    parser.add_argument("--offload", action="store_true")
    args = parser.parse_args()

    fallback_model_id = None
    if args.model_id in KNOWN_GATED_STABILITY_MODEL_IDS and args.model_id != UNGATED_DEFAULT_MODEL_ID:
        fallback_model_id = UNGATED_DEFAULT_MODEL_ID

    catalog_path = Path(args.catalog_path).resolve()
    families = load_scene_families(catalog_path)

    if args.list_catalog:
        print_catalog(families)
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "prompt_manifest.csv"

    manifest = build_manifest(
        families=families,
        base_seed=args.base_seed,
        unique_prompts=args.unique_prompts,
        replicates=args.replicates,
        selection_seed=args.selection_seed,
        trials=args.subset_trials,
        scene_family=args.scene_family,
        scenario_mode=args.scenario_mode,
        primary_key=args.primary_key,
        secondary_key=args.secondary_key,
        ternary_key=args.ternary_key,
        background_key=args.background_key,
    )
    write_manifest_csv(manifest, csv_path)
    print(f"Wrote manifest to {csv_path} with {len(manifest)} rows")

    if args.manifest_only:
        return

    generate_images(
        manifest=manifest,
        output_dir=output_dir,
        model_id=args.model_id,
        model_family=args.model_family,
        steps=args.steps,
        guidance=args.guidance,
        height=args.height,
        width=args.width,
        offload=args.offload,
        device=args.device,
        max_sequence_length=args.max_sequence_length,
        fallback_model_id=fallback_model_id,
    )


if __name__ == "__main__":
    main()
