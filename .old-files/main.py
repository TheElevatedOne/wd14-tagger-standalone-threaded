from tagger.interrogator import Interrogator, WaifuDiffusionInterrogator
from PIL import Image
from pathlib import Path
import argparse
import threading
import os
import os.path as op

from tagger.interrogators import interrogators


class Iterrogate():
    def __init__(self):
        self.args = self.parse()
        self.interrogator = interrogators[self.args.model]

        if self.args.cpu:
            self.interrogator.use_cpu()

        if self.args.dir is not None:
            self.directory_iter()
        if self.args.file is not None:
            self.file_iter()

    def parse(self):
        parser = argparse.ArgumentParser()

        group = parser.add_mutually_exclusive_group(required=True)
        group.add_argument('--dir', help='Predictions for all images in the directory')
        group.add_argument('--file', help='Predictions for one file')

        parser.add_argument(
            '--threshold',
            type=float,
            default=0.35,
            help='Prediction threshold (default is 0.35)')
        parser.add_argument(
            '--ext',
            default='.txt',
            help='Extension to add to caption file in case of dir option (default is .txt)')
        parser.add_argument(
            '--overwrite',
            action='store_true',
            help='Overwrite caption file if it exists')
        parser.add_argument(
            '--cpu',
            action='store_true',
            help='Use CPU only')
        parser.add_argument(
            '--rawtag',
            action='store_true',
            help='Use the raw output of the model')
        parser.add_argument(
            '--model',
            default='wd14-convnextv2.v1',
            choices=list(interrogators.keys()),
            help='modelname to use for prediction (default is wd14-convnextv2.v1)')
        return parser.parse_args()

    def image_interrogate(self, image_path: Path):
        """
        Predictions from a image path
        """
        im = Image.open(image_path)
        result = self.interrogator.interrogate(im)

        if self.args.rawtag:
            return Interrogator.postprocess_tags(result[1], threshold=self.args.threshold)
        return Interrogator.postprocess_tags(result[1], threshold=self.args.threshold, escape_tag=True, replace_underscore=True)

    def directory_iter(self):
        d = op.abspath(self.args.dir)
        q = os.listdir(d)
        for f in q:
            image_path = op.join(d,f)
            if not op.isfile(image_path) or f.split(".")[-1] not in ['png', 'jpg', 'jpeg', 'webp']:
                continue

            caption_path = op.join(d, f.replace(f'.{f.split(".")[-1]}', "")+self.args.ext)

            if op.isfile(caption_path) and not self.args.overwrite:
                # skip if file exists
                print('skip:', image_path)
                continue

            print(f'processing: {image_path} | {"0"*(len(str(len(q)))-len(str(q.index(f)))+1)+str(q.index(f))}/{len(q)}', end="\r")
            tags = self.image_interrogate(Path(image_path))

            tags_str = ', '.join(tags.keys())

            with open(caption_path, 'w') as fp:
                fp.write(tags_str)

    def file_iter(self):
        tags = self.image_interrogate(Path(self.args.file))
        print()
        tags_str = ', '.join(tags.keys())
        print(tags_str)


if __name__ == "__main__":
    Iterrogate()
