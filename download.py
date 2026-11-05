import argparse
import re
import shutil
import sys

from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Set

from huggingface_hub import try_to_load_from_cache
from huggingface_hub.constants import HF_HUB_CACHE

from tagger.interrogators import interrogators


class Model(NamedTuple):
    size_mb: int
    tags: int
    characters: Optional[int]  # None when the tag list has no categories
    year: int
    notes: str


# Sizes, tag counts and years are from the model repositories on Hugging Face.
MODELS: Dict[str, Model] = {
    'pixai-tagger-v0.9': Model(1272, 13461, 3720, 2025, 'Adds series tags, no ratings'),
    'camie-tagger-v2': Model(797, 70527, 26968, 2025, 'Also artist, copyright and year tags'),
    'camie-tagger': Model(861, 70527, 26968, 2025, 'Also artist, copyright and year tags'),
    'wd-eva02-large-tagger-v3': Model(1260, 10861, 2751, 2024, 'Large EVA02'),
    'wd-vit-large-tagger-v3': Model(1261, 10861, 2751, 2024, 'Large ViT'),
    'wd-convnext-v3': Model(395, 10861, 2751, 2024, 'Default model'),
    'wd-v1-4-vit-tagger.v3': Model(379, 10861, 2751, 2024, 'ViT'),
    'wd-v1-4-swinv2-tagger.v3': Model(467, 10861, 2751, 2024, 'SwinV2'),
    'wd-v1-4-moat-tagger.v2': Model(326, 9083, 2132, 2023, 'MOAT'),
    'wd14-vit.v2': Model(373, 9083, 2132, 2023, 'ViT'),
    'wd14-convnext.v2': Model(388, 9083, 2132, 2023, 'ConvNeXT'),
    'wd14-convnextv2.v1': Model(388, 9083, 2132, 2023, 'ConvNeXTV2'),
    'wd14-swinv2-v1': Model(455, 9083, 2132, 2023, 'SwinV2'),
    'wd14-vit.v1': Model(365, 6549, 0, 2022, 'ViT, no character tags'),
    'wd14-convnext.v1': Model(377, 6549, 0, 2022, 'ConvNeXT, no character tags'),
    'z3d-e621-convnext-toynya': Model(390, 9525, 541, 2023, 'e621 (furry) tags, no ratings'),
    'z3d-e621-convnext-silveroxides': Model(390, 9525, 541, 2024, 'Mirror of the toynya model'),
    'mld-caformer.dec-5-97527': Model(287, 12547, None, 2023, 'No ratings, may be slow on AMD GPUs'),
    'mld-tresnetd.6-30000': Model(236, 12547, None, 2023, 'No ratings, may be slow on AMD GPUs'),
}
# The same files under a second name.
ALIASES = {'wd-v1-4-convnext-tagger.v3': 'wd-convnext-v3'}

PROMPT = 'Models to download (e.g. "1 3 5-7", a model name, "all" or "missing", Enter to skip): '


def is_downloaded(name: str) -> bool:
    model = interrogators[name]
    return all(
        isinstance(try_to_load_from_cache(model.repo_id, filename), str)
        for filename in (model.model_path, model.tags_path)
    )


def format_size(size_mb: int) -> str:
    return f'{size_mb / 1000:.1f} GB' if size_mb >= 1000 else f'{size_mb} MB'


def print_models(names: List[str]) -> None:
    width = shutil.get_terminal_size().columns
    print(f'{"#":>3}  {"Model":<31} {"Size":>7} {"Tags":>6} {"Chars":>6} {"Year":>4}  {"Have":<4}  Notes')
    for number, name in enumerate(names, 1):
        model = MODELS[name]
        have = 'yes' if is_downloaded(name) else ''
        characters = '?' if model.characters is None else model.characters
        line = (f'{number:>3}  {name:<31} {format_size(model.size_mb):>7} {model.tags:>6} '
                f'{characters:>6} {model.year:>4}  {have:<4}')
        # Keep the notes readable on narrow terminals.
        if len(line) + 2 + len(model.notes) > width:
            print(line)
            print(f'{"":>5}{model.notes}')
        else:
            print(f'{line}  {model.notes}')
    print('Tags is the number of tags the model knows, Chars how many of them are characters.')


def parse_selection(text: str, names: List[str]) -> List[str]:
    """
    Models named by the user's input, in list order. Raises ValueError on input it can't read.
    """
    selected: Set[str] = set()
    for token in re.split(r'[,\s]+', text.strip()):
        if not token:
            continue
        token = ALIASES.get(token, token)
        span = re.fullmatch(r'(\d+)(?:-(\d+))?', token)
        if token == 'all':
            selected.update(names)
        elif token == 'missing':
            selected.update(name for name in names if not is_downloaded(name))
        elif token in names:
            selected.add(token)
        elif span:
            first = int(span[1])
            last = int(span[2] or first)
            if not 1 <= first <= last <= len(names):
                raise ValueError(f'"{token}" is not in 1-{len(names)}')
            selected.update(names[first - 1:last])
        else:
            raise ValueError(f'"{token}" is not a number or a model name')
    return [name for name in names if name in selected]


def choose_models(names: List[str]) -> List[str]:
    print_models(names)
    while True:
        try:
            text = input(PROMPT)
        except EOFError:
            print()
            return []
        try:
            selected = parse_selection(text, names)
        except ValueError as error:
            print(error)
            continue
        if not selected:
            return []

        missing = [name for name in selected if not is_downloaded(name)]
        if not missing:
            print('All of those are already downloaded.')
            return []

        total = sum(MODELS[name].size_mb for name in missing)
        cache = Path(HF_HUB_CACHE)
        cache.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(cache).free // 1_000_000
        print(f'Download {len(missing)}: {", ".join(missing)}')
        print(f'Needs {format_size(total)}, {format_size(free)} free in {cache}')
        if total > free:
            print('Not enough disk space, choose fewer models.')
            continue

        try:
            answer = input('Continue? [Y/n] ')
        except EOFError:
            print()
            return []
        if answer.strip().lower() in ('', 'y', 'yes'):
            return missing


def download_models(names: List[str]) -> int:
    failed = []
    already = 0
    for number, name in enumerate(names, 1):
        print(f'[{number}/{len(names)}] {name}', flush=True)
        if is_downloaded(name):
            already += 1
            continue
        try:
            interrogators[name].download()
        except Exception as error:
            print(f'failed: {name}: {error}', file=sys.stderr)
            failed.append(name)

    print(f'Downloaded {len(names) - len(failed) - already}, already had {already}, failed {len(failed)}')
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Download tagger models to the Hugging Face cache. '
                    'Without arguments, it asks which ones you want.')
    parser.add_argument(
        'models',
        nargs='*',
        metavar='model',
        help='Models to download without asking')
    parser.add_argument(
        '--list',
        action='store_true',
        help='List the models and exit')
    args = parser.parse_args()

    names = list(MODELS)
    for name in args.models:
        if name not in MODELS and name not in ALIASES:
            parser.error(f'{name} is not a model, see --list')
    try:
        if args.list:
            print_models(names)
            return 0
        if args.models:
            selected = list(dict.fromkeys(ALIASES.get(name, name) for name in args.models))
        else:
            selected = choose_models(names)
        return download_models(selected) if selected else 0
    except KeyboardInterrupt:
        print('\nInterrupted, finished downloads are kept')
        return 130


if __name__ == '__main__':
    sys.exit(main())
