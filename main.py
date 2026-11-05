import argparse
import json
import multiprocessing
import os
import signal
import sys
import time

from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

from PIL import Image, ImageFile, ImageOps
from send2trash import send2trash
from tqdm import tqdm

from tagger.interrogator import PROVIDERS, Interrogator, log
from tagger.interrogators import interrogators

# Allow images with broken headers to load
ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp'}
DEFAULT_MODEL = 'wd-convnext-v3'

# Set in each process by init_worker
interrogator: Optional[Interrogator] = None
options: Optional[argparse.Namespace] = None


def threshold_value(value: str) -> float:
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise argparse.ArgumentTypeError(f'{value} is not between 0.0 and 1.0')
    return number


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError(f'{value} is less than 1')
    return number


def non_negative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError(f'{value} is less than 0')
    return number


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Tag images with a WD14-style model. A directory gets a caption file '
                    'per image, a single file prints its tags.')

    parser.add_argument(
        'path',
        type=Path,
        help='Directory of images, or a single image')
    parser.add_argument(
        '--threshold',
        type=threshold_value,
        help='Prediction threshold [0.0 - 1.0] (default is 0.35, for pixai-tagger-v0.9 general 0.3, character 0.85)')
    parser.add_argument(
        '--general-threshold',
        type=threshold_value,
        help='Threshold for non-character tags, overrides --threshold')
    parser.add_argument(
        '--character-threshold',
        type=threshold_value,
        help='Threshold for character tags, overrides --threshold')
    parser.add_argument(
        '-l',
        '--limit',
        type=non_negative_int,
        default=100,
        help='Limit the amount of tags to write, custom tags are excluded, 0 for no limit (default is 100)')
    parser.add_argument(
        '--ext',
        default='.txt',
        choices=['.txt', '.caption'],
        help='Extension of the caption files (default is .txt)')
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output tags and ratings with their confidence as JSON, written to .json files for a directory')
    parser.add_argument(
        '-w',
        '--overwrite',
        action='store_true',
        help='Replace existing caption files, the old ones are moved to the trash')
    parser.add_argument(
        '-r',
        '--recursive',
        action='store_true',
        help='Also tag images in subdirectories')
    parser.add_argument(
        '-tp',
        '--prepend',
        type=str,
        help='Prepend custom tags, write in the style (with the quotes) "tag1, tag2, ..."')
    parser.add_argument(
        '-ta',
        '--append',
        type=str,
        help='Append custom tags, write in the style (with the quotes) "tag1, tag2, ..."')
    parser.add_argument(
        '-e',
        '--exclude',
        type=str,
        help='Remove these tags from the model output, write in the style (with the quotes) "tag1, tag2, ..."')
    parser.add_argument(
        '--rawtag',
        action='store_true',
        help='Use the raw output of the model')
    parser.add_argument(
        '--model',
        default=DEFAULT_MODEL,
        choices=list(interrogators.keys()),
        help=f'Model to use for prediction (default is {DEFAULT_MODEL})')
    device = parser.add_mutually_exclusive_group()
    device.add_argument(
        '--provider',
        default='auto',
        choices=['auto', *PROVIDERS],
        help='Execution provider. auto uses the best one available and falls back to the CPU, '
             'any other value fails if the provider can\'t start (default is auto)')
    device.add_argument(
        '--cpu',
        action='store_true',
        help='Use CPU only, same as --provider cpu')
    parser.add_argument(
        '--device-id',
        type=non_negative_int,
        default=0,
        help='Index of the GPU to use (default is 0)')
    parser.add_argument(
        '-t',
        '--threads',
        type=positive_int,
        default=1,
        help='Number of worker processes, each loads its own copy of the model (default is 1)')
    args = parser.parse_args()

    if not args.path.exists():
        parser.error(f'{args.path} does not exist')
    if args.path.is_file() and args.path.suffix.lower() not in IMAGE_EXTENSIONS:
        parser.error(f'{args.path} is not one of: {", ".join(sorted(IMAGE_EXTENSIONS))}')
    return args


def split_tags(tags: Optional[str]) -> List[str]:
    if tags is None:
        return []
    return [tag.strip() for tag in tags.split(',') if tag.strip()]


def excluded_tags(tags: Optional[str]) -> set:
    """
    Tags as the user wrote them, and in the danbooru form the model uses.
    """
    written = split_tags(tags)
    danbooru = [tag.replace(' ', '_').replace(r'\(', '(').replace(r'\)', ')') for tag in written]
    return {*written, *danbooru}


def init_worker(args: argparse.Namespace, workers: int) -> None:
    global interrogator, options

    # The main process handles Ctrl+C and lets the running images finish.
    if workers > 1:
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    options = args
    interrogator = interrogators[args.model]
    interrogator.configure(
        provider='cpu' if args.cpu else args.provider,
        device_id=args.device_id,
        # Without a limit, every worker starts a thread for every core.
        cpu_threads=max(1, (os.cpu_count() or 1) // workers) if workers > 1 else 0,
    )
    interrogator.load()


def interrogate_image(image_path: Path) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Ratings and selected tags of an image path
    """
    with Image.open(image_path) as image:
        ratings, tags = interrogator.interrogate(ImageOps.exif_transpose(image))

    tags = interrogator.select_tags(
        tags,
        threshold=options.threshold if options.general_threshold is None else options.general_threshold,
        character_threshold=options.threshold if options.character_threshold is None else options.character_threshold,
    )
    tags = Interrogator.postprocess_tags(
        tags,
        threshold=0.0,
        exclude_tags=excluded_tags(options.exclude),
        escape_tag=not options.rawtag,
        replace_underscore=not options.rawtag,
    )
    return ratings, tags


def merge_custom_tags(tags: Dict[str, float]) -> Dict[str, float]:
    """
    Add --prepend and --append to the limited model tags. A tag is only kept the first time.
    """
    model_tags = list(tags.items())[:options.limit or None]
    merged = {}
    seen = set()
    for tag, confidence in [
        *[(tag, 1.0) for tag in split_tags(options.prepend)],
        *model_tags,
        *[(tag, 1.0) for tag in split_tags(options.append)],
    ]:
        key = tag.replace('\\', '').replace('_', ' ')
        if key not in seen:
            seen.add(key)
            merged[tag] = confidence
    return merged


def render_output(image_path: Path) -> str:
    ratings, tags = interrogate_image(image_path)
    tags = merge_custom_tags(tags)
    if options.json:
        return json.dumps({'tags': tags, 'ratings': ratings}, ensure_ascii=False, indent=2)
    return ', '.join(tags)


def write_caption(caption_path: Path, text: str) -> None:
    temporary = caption_path.with_name(f'.{caption_path.name}.{os.getpid()}.tmp')
    try:
        with open(temporary, 'w', encoding='utf-8') as fp:
            fp.write(text)
        if caption_path.exists():
            try:
                send2trash(caption_path)
            except Exception as error:
                log(f'Could not move {caption_path} to the trash, replacing it: {error}')
        os.replace(temporary, caption_path)
    finally:
        temporary.unlink(missing_ok=True)


def tag_image(image_path: Path, caption_path: Path) -> Tuple[Path, Optional[str]]:
    """
    Write the caption of an image. An error is returned, so one bad file doesn't stop the run.
    """
    try:
        write_caption(caption_path, render_output(image_path))
    except Exception as error:
        return image_path, f'{type(error).__name__}: {error}'
    return image_path, None


def find_images(directory: Path, recursive: bool) -> List[Path]:
    paths = directory.rglob('*') if recursive else directory.iterdir()
    return sorted(p for p in paths if p.suffix.lower() in IMAGE_EXTENSIONS and p.is_file())


def plan_jobs(args: argparse.Namespace) -> Tuple[List[Tuple[Path, Path]], int]:
    """
    Pairs of (image, caption) to process, and how many images are skipped.
    """
    ext = '.json' if args.json else args.ext
    jobs = {}
    skipped = 0
    for image_path in find_images(args.path, args.recursive):
        caption_path = image_path.parent / f'{image_path.stem}{ext}'
        if caption_path in jobs:
            log(f'skip: {image_path} shares its caption file with {jobs[caption_path]}')
            skipped += 1
        elif caption_path.is_file() and not args.overwrite:
            skipped += 1
        else:
            jobs[caption_path] = image_path
    return [(image, caption) for caption, image in jobs.items()], skipped


def run_jobs(args: argparse.Namespace, jobs: List[Tuple[Path, Path]], workers: int) -> Iterator[Tuple[Path, Optional[str]]]:
    if workers == 1:
        init_worker(args, workers)
        for image_path, caption_path in jobs:
            yield tag_image(image_path, caption_path)
        return

    # Each worker takes the next image when it's free, so slow images don't hold the others back.
    # GPU runtimes don't survive a fork, so the workers have to be spawned.
    context = multiprocessing.get_context('spawn')
    pool = ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
        initializer=init_worker,
        initargs=(args, workers),
    )
    try:
        futures = [pool.submit(tag_image, image, caption) for image, caption in jobs]
        for future in as_completed(futures):
            yield future.result()
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


def tag_directory(args: argparse.Namespace) -> int:
    start_time = time.time()
    jobs, skipped = plan_jobs(args)
    if not jobs:
        print(f'Nothing to do, skipped {skipped} images with existing captions')
        return 0

    # Download once here, so the workers don't all start the same download.
    interrogators[args.model].download()

    workers = min(args.threads, len(jobs))
    failed = 0
    with tqdm(total=len(jobs), desc='Tagging', unit='img', file=sys.stderr, disable=None) as progress:
        for image_path, error in run_jobs(args, jobs, workers):
            if error is not None:
                failed += 1
                tqdm.write(f'failed: {image_path}: {error}', file=sys.stderr)
            progress.update()

    print(f'Tagged {len(jobs) - failed}, skipped {skipped}, failed {failed} '
          f'in {round(time.time() - start_time, 2)} s')
    return 1 if failed else 0


def tag_file(args: argparse.Namespace) -> int:
    init_worker(args, 1)
    print(render_output(args.path))
    return 0


def main() -> int:
    args = parse_args()
    try:
        return tag_file(args) if args.path.is_file() else tag_directory(args)
    except BrokenProcessPool:
        log('A worker process died. If it was loading the model, the reason is above.')
    except KeyboardInterrupt:
        log('Interrupted, captions written so far are kept')
        return 130
    except (RuntimeError, OSError) as error:
        log(f'Error: {error}')
    return 1


if __name__ == '__main__':
    sys.exit(main())
