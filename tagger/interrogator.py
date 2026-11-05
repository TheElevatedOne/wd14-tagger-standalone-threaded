import json
import os
import re
import sys

from typing import Dict, Iterable, List, Set, Tuple

import numpy as np
import pandas as pd

from huggingface_hub import hf_hub_download
from PIL import Image

import tagger.dbimutils as dbimutils

tag_escape_pattern = re.compile(r'([\\()])')

# Emoticon tags, where the underscore is part of the tag.
KAOMOJI = {
    '0_0', '(o)_(o)', '+_+', '+_-', '._.', '<o>_<o>', '<|>_<|>', '=_=', '>_<',
    '3_3', '6_9', '>_o', '@_@', '^_^', 'o_o', 'u_u', 'x_x', '|_|', '||_||',
}

# Values of --provider. The order is the order "auto" tries them in.
PROVIDERS = {
    'cuda': 'CUDAExecutionProvider',            # NVIDIA
    'migraphx': 'MIGraphXExecutionProvider',    # AMD on Linux, onnxruntime >= 1.23
    'rocm': 'ROCMExecutionProvider',            # AMD on Linux, onnxruntime < 1.23
    'directml': 'DmlExecutionProvider',         # AMD/NVIDIA/Intel on Windows
    'coreml': 'CoreMLExecutionProvider',        # Apple
    'openvino': 'OpenVINOExecutionProvider',    # Intel
    'cpu': 'CPUExecutionProvider',
}
CPU_PROVIDER = PROVIDERS['cpu']
DEVICE_ID_PROVIDERS = {
    PROVIDERS['cuda'], PROVIDERS['migraphx'], PROVIDERS['rocm'], PROVIDERS['directml'],
}
AMD_PROVIDERS = {PROVIDERS['migraphx'], PROVIDERS['rocm']}
AMD_HINT = (
    'Check that your user is in the "render" and "video" groups. ROCm has no official '
    'support for some Radeon cards; for RX 6000 cards other than the 6800/6900 series, '
    'try setting HSA_OVERRIDE_GFX_VERSION=10.3.0.'
)

# Tag categories in selected_tags.csv.
CATEGORY_CHARACTER = 4
CATEGORY_RATING = 9


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def use_migraphx_cache() -> None:
    """
    Keep compiled models between runs, compiling one takes minutes.
    """
    if 'ORT_MIGRAPHX_MODEL_CACHE_PATH' not in os.environ:
        cache_home = os.environ.get('XDG_CACHE_HOME') or os.path.expanduser('~/.cache')
        os.environ['ORT_MIGRAPHX_MODEL_CACHE_PATH'] = os.path.join(cache_home, 'wd14-tagger', 'migraphx')
    os.makedirs(os.environ['ORT_MIGRAPHX_MODEL_CACHE_PATH'], exist_ok=True)


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


class Interrogator:
    default_threshold = 0.35
    default_character_threshold = 0.35

    @staticmethod
    def postprocess_tags(
        tags: Dict[str, float],
        threshold=0.35,
        additional_tags: Iterable[str] = (),
        exclude_tags: Iterable[str] = (),
        sort_by_alphabetical_order=False,
        add_confident_as_weight=False,
        replace_underscore=False,
        replace_underscore_excludes: Iterable[str] = KAOMOJI,
        escape_tag=False
    ) -> Dict[str, float]:
        tags = dict(tags)
        for t in additional_tags:
            tags[t] = 1.0

        # those lines are totally not "pythonic" but looks better to me
        tags = {
            t: c

            # sort by tag name or confident
            for t, c in sorted(
                tags.items(),
                key=lambda i: i[0 if sort_by_alphabetical_order else 1],
                reverse=not sort_by_alphabetical_order
            )

            # filter tags
            if (
                c >= threshold
                and t not in exclude_tags
            )
        }

        new_tags = []
        for tag in list(tags):
            new_tag = tag

            if replace_underscore and tag not in replace_underscore_excludes:
                new_tag = new_tag.replace('_', ' ')

            if escape_tag:
                new_tag = tag_escape_pattern.sub(r'\\\1', new_tag)

            if add_confident_as_weight:
                new_tag = f'({new_tag}:{tags[tag]})'

            new_tags.append((new_tag, tags[tag]))
        tags = dict(new_tags)

        return tags

    def __init__(self, name: str, repo_id: str, model_path: str, tags_path: str) -> None:
        self.name = name
        self.repo_id = repo_id
        self.model_path = model_path
        self.tags_path = tags_path
        self.model = None
        self.character_tags: Set[str] = set()
        self.provider = 'auto'
        self.device_id = 0
        self.cpu_threads = 0

    def configure(self, provider='auto', device_id=0, cpu_threads=0) -> None:
        """
        Set how the model runs. Takes effect the next time the model is loaded.

        provider is "auto" or a key of PROVIDERS. cpu_threads limits the threads of
        one session, 0 lets onnxruntime decide.
        """
        self.provider = provider
        self.device_id = device_id
        self.cpu_threads = cpu_threads

    def use_cpu(self) -> None:
        self.provider = 'cpu'

    def download(self) -> Tuple[str, str]:
        log(f'Loading {self.name} model file from {self.repo_id}')
        model_path = hf_hub_download(repo_id=self.repo_id, filename=self.model_path)
        tags_path = hf_hub_download(repo_id=self.repo_id, filename=self.tags_path)
        return model_path, tags_path

    def _session_options(self, provider: str):
        from onnxruntime import ExecutionMode, SessionOptions

        options = SessionOptions()
        if self.cpu_threads:
            options.intra_op_num_threads = self.cpu_threads
        if provider == PROVIDERS['directml']:
            # DirectML can't run with these on.
            options.enable_mem_pattern = False
            options.execution_mode = ExecutionMode.ORT_SEQUENTIAL
        return options

    def _create_session(self, model_path: str, provider: str):
        from onnxruntime import InferenceSession

        if provider == PROVIDERS['migraphx']:
            use_migraphx_cache()

        providers = []
        if provider != CPU_PROVIDER:
            options = {'device_id': str(self.device_id)} if provider in DEVICE_ID_PROVIDERS else {}
            providers.append((provider, options))
        providers.append(CPU_PROVIDER)
        return InferenceSession(
            model_path,
            sess_options=self._session_options(provider),
            providers=providers,
        )

    def create_session(self, model_path: str):
        """
        Create the session on the requested provider.

        A provider named with --provider has to work or this raises. With "auto", a
        provider that can't start falls through to the next one, ending at the CPU.
        """
        from onnxruntime import get_available_providers

        available = get_available_providers()

        if self.provider != 'auto':
            provider = PROVIDERS[self.provider]
            if provider not in available:
                raise RuntimeError(
                    f'{provider} is not in this onnxruntime build (it has: '
                    f'{", ".join(available)}). Run setup.sh to install a matching build.')
            try:
                session = self._create_session(model_path, provider)
            except Exception as error:
                raise RuntimeError(f'{provider} failed: {error}\n{self._hint(provider)}') from error
            if session.get_providers()[0] != provider:
                raise RuntimeError(
                    f'{provider} could not start and onnxruntime fell back to '
                    f'{session.get_providers()[0]}.\n{self._hint(provider)}')
        else:
            session = None
            for provider in [p for p in PROVIDERS.values() if p in available]:
                try:
                    session = self._create_session(model_path, provider)
                except Exception as error:
                    if provider == CPU_PROVIDER:
                        raise
                    log(f'[device] {provider} failed: {error}\n{self._hint(provider)}')
                    continue
                if session.get_providers()[0] == provider:
                    break
                log(f'[device] {provider} could not start.\n{self._hint(provider)}')

        active = session.get_providers()[0]
        log(f'[device] {self.name}: running on {active}')
        if active == CPU_PROVIDER and self.provider == 'auto':
            log('[device] No GPU in use. Run setup.sh to install the onnxruntime build for your GPU.')
        return session

    @staticmethod
    def _hint(provider: str) -> str:
        return AMD_HINT if provider in AMD_PROVIDERS else f'See the onnxruntime documentation for {provider}.'

    def load(self) -> None:
        raise NotImplementedError()

    def unload(self) -> bool:
        if self.model is None:
            return False
        self.model = None
        log(f'Unloaded {self.name}')
        return True

    def select_tags(self, tags: Dict[str, float], threshold=None, character_threshold=None) -> Dict[str, float]:
        """
        Keep the tags that reach their threshold. None uses the default of the model.
        """
        general = self.default_threshold if threshold is None else threshold
        character = self.default_character_threshold if character_threshold is None else character_threshold
        return {
            tag: confidence
            for tag, confidence in tags.items()
            if confidence >= (character if tag in self.character_tags else general)
        }

    def interrogate(
        self,
        image: Image.Image
    ) -> Tuple[
        Dict[str, float],  # rating confidents
        Dict[str, float]  # tag confidents
    ]:
        raise NotImplementedError()


class WaifuDiffusionInterrogator(Interrogator):
    def __init__(
        self,
        name: str,
        repo_id: str,
        model_path='model.onnx',
        tags_path='selected_tags.csv',
    ) -> None:
        super().__init__(name, repo_id, model_path, tags_path)
        self.tag_names: List[str] = []
        self.rating_tags: Set[str] = set()

    def load(self) -> None:
        model_path, tags_path = self.download()
        self.model = self.create_session(model_path)
        log(f'Loaded {self.name} model from {model_path}')
        self.load_tags(pd.read_csv(tags_path))

    def load_tags(self, tags: pd.DataFrame) -> None:
        self.tag_names = tags['name'].tolist()
        self.rating_tags = set(tags.loc[tags['category'] == CATEGORY_RATING, 'name'])
        self.character_tags = set(tags.loc[tags['category'] == CATEGORY_CHARACTER, 'name'])

    def preprocess(self, image: Image.Image) -> np.ndarray:
        # code for converting the image and running the model is taken from the link below
        # thanks, SmilingWolf!
        # https://huggingface.co/spaces/SmilingWolf/wd-v1-4-tags/blob/main/app.py
        _, height, _, _ = self.model.get_inputs()[0].shape

        image = np.asarray(dbimutils.fill_transparent(image))

        # PIL RGB to OpenCV BGR
        image = image[:, :, ::-1]

        image = dbimutils.make_square(image, height)
        image = dbimutils.smart_resize(image, height)
        image = image.astype(np.float32)
        return np.expand_dims(image, 0)

    def output_name(self) -> str:
        return self.model.get_outputs()[0].name

    def interrogate(
        self,
        image: Image.Image
    ) -> Tuple[
        Dict[str, float],  # rating confidents
        Dict[str, float]  # tag confidents
    ]:
        if self.model is None:
            self.load()

        input_name = self.model.get_inputs()[0].name
        confidents = self.model.run([self.output_name()], {input_name: self.preprocess(image)})[0][0]

        ratings = {}
        tags = {}
        for name, confident in zip(self.tag_names, confidents.tolist()):
            (ratings if name in self.rating_tags else tags)[name] = confident
        return ratings, tags


class PixAITaggerInterrogator(WaifuDiffusionInterrogator):
    """
    Interrogator for pixai-labs/pixai-tagger-v0.9, using the ONNX export by deepghs.

    It has no rating classes, and detected characters also add their series tag.
    The thresholds come from thresholds.csv in the model repository.
    """
    default_threshold = 0.3
    default_character_threshold = 0.85

    def __init__(self, name: str, repo_id: str, model_path='model.onnx', tags_path='selected_tags.csv') -> None:
        super().__init__(name, repo_id, model_path, tags_path)
        self.character_series: Dict[str, List[str]] = {}

    def load_tags(self, tags: pd.DataFrame) -> None:
        super().load_tags(tags)
        characters = tags[tags['category'] == CATEGORY_CHARACTER]
        self.character_series = {
            name: json.loads(series)
            for name, series in zip(characters['name'], characters['ips'])
            if isinstance(series, str) and series != '[]'
        }

    def preprocess(self, image: Image.Image) -> np.ndarray:
        # The stages from preprocess.json in the model repository.
        _, _, height, width = self.model.get_inputs()[0].shape
        image = dbimutils.fill_transparent(image).resize((width, height), Image.Resampling.BILINEAR)
        x = np.asarray(image, dtype=np.float32) / 255.0
        x = (x - 0.5) / 0.5
        return np.expand_dims(x.transpose((2, 0, 1)), 0)

    def output_name(self) -> str:
        # The first output is not the tag scores.
        names = [output.name for output in self.model.get_outputs()]
        return 'prediction' if 'prediction' in names else names[0]

    def select_tags(self, tags: Dict[str, float], threshold=None, character_threshold=None) -> Dict[str, float]:
        selected = super().select_tags(tags, threshold, character_threshold)
        series_tags: Dict[str, float] = {}
        for tag, confidence in selected.items():
            for series in self.character_series.get(tag, []):
                series_tags[series] = max(confidence, series_tags.get(series, 0.0))
        return {**selected, **series_tags}


class MLDanbooruInterrogator(Interrogator):
    """ Interrogator for the MLDanbooru model. """
    def __init__(self, name: str, repo_id: str, model_path: str, tags_path='classes.json') -> None:
        super().__init__(name, repo_id, model_path, tags_path)
        self.tags: List[str] = []

    def load(self) -> None:
        model_path, tags_path = self.download()
        self.model = self.create_session(model_path)
        log(f'Loaded {self.name} model from {model_path}')

        with open(tags_path, 'r', encoding='utf-8') as filen:
            self.tags = json.load(filen)

    def interrogate(
        self,
        image: Image.Image
    ) -> Tuple[
        Dict[str, float],  # rating confidents
        Dict[str, float]  # tag confidents
    ]:
        if self.model is None:
            self.load()

        image = dbimutils.fill_transparent(image)
        image = dbimutils.resize(image, 448)  # TODO CUSTOMIZE

        x = np.asarray(image, dtype=np.float32) / 255
        # HWC -> 1CHW
        x = np.expand_dims(x.transpose((2, 0, 1)), 0)

        input_ = self.model.get_inputs()[0]
        output = self.model.get_outputs()[0]
        y, = self.model.run([output.name], {input_.name: x})

        return {}, dict(zip(self.tags, sigmoid(y).flatten().tolist()))


class CamieTaggerInterrogator(Interrogator):
    """
    Interrogator for the Camie Tagger models.
    https://huggingface.co/Camais03/camie-tagger/blob/main/onnx_inference.py
    https://huggingface.co/Camais03/camie-tagger-v2/blob/main/onnx_inference.py
    """
    IMAGE_SIZE = 512
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    IMAGENET_PAD_COLOR = (124, 116, 104)

    def __init__(self, name: str, repo_id: str, model_path: str, tags_path: str, imagenet_norm=False) -> None:
        super().__init__(name, repo_id, model_path, tags_path)
        # v2 is trained on normalized images with mean-coloured padding, v1 on neither.
        self.imagenet_norm = imagenet_norm
        self.idx_to_tag: Dict[int, str] = {}
        self.rating_tags: Set[str] = set()

    def load(self) -> None:
        model_path, tags_path = self.download()
        self.model = self.create_session(model_path)
        log(f'Loaded {self.name} model from {model_path}')

        with open(tags_path, 'r', encoding='utf-8') as filen:
            metadata = json.load(filen)

        # v2 nests the mapping, v1 has it at the top level
        mapping = metadata['dataset_info']['tag_mapping'] if 'dataset_info' in metadata else metadata
        categories = mapping['tag_to_category']
        self.idx_to_tag = {int(idx): tag for idx, tag in mapping['idx_to_tag'].items()}
        self.rating_tags = {tag for tag, category in categories.items() if category == 'rating'}
        self.character_tags = {tag for tag, category in categories.items() if category == 'character'}

    def preprocess(self, image: Image.Image) -> np.ndarray:
        image = image.convert('RGB')
        size = self.IMAGE_SIZE

        # Fit inside a square, keeping the aspect ratio
        scale = size / max(image.size)
        resized = image.resize(
            (max(1, int(image.width * scale)), max(1, int(image.height * scale))),
            Image.Resampling.LANCZOS,
        )

        pad_color = self.IMAGENET_PAD_COLOR if self.imagenet_norm else (0, 0, 0)
        canvas = Image.new('RGB', (size, size), pad_color)
        canvas.paste(resized, ((size - resized.width) // 2, (size - resized.height) // 2))

        x = np.asarray(canvas, dtype=np.float32) / 255.0
        if self.imagenet_norm:
            x = (x - self.IMAGENET_MEAN) / self.IMAGENET_STD
        # HWC -> 1CHW
        return np.expand_dims(x.transpose((2, 0, 1)), 0)

    def interrogate(
        self,
        image: Image.Image
    ) -> Tuple[
        Dict[str, float],  # rating confidents
        Dict[str, float]  # tag confidents
    ]:
        if self.model is None:
            self.load()

        input_ = self.model.get_inputs()[0]
        outputs = self.model.run(None, {input_.name: self.preprocess(image)})

        # outputs: initial predictions, then refined predictions if the model has them
        confidents = sigmoid(outputs[1] if len(outputs) > 1 else outputs[0])[0].tolist()

        ratings = {}
        tags = {}
        for idx, tag in self.idx_to_tag.items():
            (ratings if tag in self.rating_tags else tags)[tag] = confidents[idx]
        return ratings, tags
