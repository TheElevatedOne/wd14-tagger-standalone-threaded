from tagger.interrogator import Interrogator, WaifuDiffusionInterrogator
from PIL import Image
from pathlib import Path
from send2trash import send2trash
import argparse
import multiprocessing
import os
import os.path as op
import time
import statistics

from tagger.interrogators import interrogators


class Iterrogate():
    def __init__(self):
        self.start_time = time.time()
        self.thread_time = []
        self.position = 0
        self.args = self.parse()
        self.interrogator = interrogators[self.args.model]

        if self.args.ext not in [".txt", ".caption"]:
            raise ValueError(f'"{self.args.ext}" is not a valid caption file extension')
        if self.args.threshold > 1.0 or self.args.threshold < 0.0:
            raise ValueError(f'{self.args.threshold} is not a valid value')
        if self.args.cpu:
            self.interrogator.use_cpu()
        if self.args.dir is not None:
            self.dir_thread_main()
            return

    def parse(self):
        parser = argparse.ArgumentParser()

        parser.add_argument(
            'dir',
            type=str,
            help="Directory of images you want to run tagger on")
        parser.add_argument(
            '--threshold',
            type=float,
            default=0.35,
            help='Prediction threshold (default is 0.35) [0.0 - 1.0]')
        parser.add_argument(
            '-l',
            '--limit',
            type=int,
            default=100,
            help='Limit the amount of tags to write in caption files (Custom tags are excluded)')
        parser.add_argument(
            '--ext',
            default='.txt',
            choices=[".txt", ".caption"],
            help='Extension to add to caption file in case of dir option (default is .txt)')
        parser.add_argument(
            '-w',
            '--overwrite',
            action='store_true',
            help='Overwrite caption file if it exists')
        parser.add_argument(
            '-tp',
            '--prepend',
            type=str,
            required=False,
            help='Prepend custom tags, write in the style (with the quotes) "tag1, tag2, ..."')
        parser.add_argument(
            '-ta',
            '--append',
            type=str,
            required=False,
            help='Append custom tags, write in the style (with the quotes) "tag1, tag2, ..."')
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
            default='wd-convnext-v3',
            choices=list(interrogators.keys()),
            help='Modelname to use for prediction (default is wd14-convnextv2.v1)')
        parser.add_argument(
            '-t',
            "--threads",
            default=1,
            required=False,
            help='Ppecify the number of threads you want to run it with (multithreading)')
        return parser.parse_args()

    def time_taken(self):
        if len(self.thread_time) > 0:
            final_time = round(statistics.fmean(self.thread_time) - self.start_time, 2)
            print(f"Time taken: {final_time} s")

    def image_interrogate(self, image_path: Path):
        """
        Predictions from a image path
        """
        im = Image.open(image_path)
        result = self.interrogator.interrogate(im)
        im.close()

        if self.args.rawtag:
            return Interrogator.postprocess_tags(result[1], threshold=self.args.threshold)
        return Interrogator.postprocess_tags(result[1], threshold=self.args.threshold, escape_tag=True, replace_underscore=True)

    def chunks(self, lst, n) -> list:
        h = len(lst)//n
        temp_chunk =  [lst[i:i+h] for i in range(0, len(lst), h)]
        if len(temp_chunk[-1]) == 1 and len(temp_chunk) > 1:
            temp_chunk[-2].append(temp_chunk[-1][0])
            temp_chunk.pop()
            return temp_chunk
        return temp_chunk

    def dir_thread_main(self):
        d = op.abspath(self.args.dir)
        if self.args.overwrite is not None:
            [send2trash(op.join(d, x)) for x in os.listdir(d) if (".txt" in x) or (".caption" in x)]
        q = self.chunks(os.listdir(d), int(self.args.threads))
        self.position = len(q)
        jobs = []

        for i, s in enumerate(q):
            j = multiprocessing.Process(target=self.directory_iter, args=(i, s))
            jobs.append(j)
        for j in jobs:
            j.start()

    def additional_tags(self, tags: str, typ: bool) -> str:
        if tags is None:
            return ""
        tags = tags.split(", ")
        if typ:
            return ", ".join(tags)+", "
        return ", "+(", ".join(tags))

    def directory_iter(self, job, chunk):
        d = op.abspath(self.args.dir) #
        for f in chunk:
            image_path = op.join(d,f)
            if not op.isfile(image_path) or f.split(".")[-1] not in ['png', 'jpg', 'jpeg', 'webp']:
                continue

            caption_path = op.join(d, f.replace(f'.{f.split(".")[-1]}', "")+self.args.ext)

            if op.isfile(caption_path):
                # skip if file exists
                print('skip:', image_path)
                continue

            print(f'processing: {image_path} | {"0"*(len(str(len(chunk)))-len(str(chunk.index(f)+1)))+str(chunk.index(f)+1)}/{len(chunk)} | Job: {job}')
            tags = self.image_interrogate(Path(image_path))

            tags_str = self.additional_tags(self.args.prepend, True)+(', '.join(list(tags.keys())[:self.args.limit]))+self.additional_tags(self.args.append, False)
            with open(caption_path, 'w') as fp:
                fp.write(tags_str)
            if f == chunk[-1]:
                time.sleep(job/4)
                self.thread_time.append(time.time())
                if self.position == (job+1):
                    self.time_taken()


if __name__ == "__main__":
    Iterrogate()

