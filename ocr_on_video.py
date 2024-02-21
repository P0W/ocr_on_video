"""Performs OCR on a video and annotates it with the results."""

## Author: Prashant Srivastava
## Dated: February 21st, 2024

import argparse
import datetime
import logging
import multiprocessing
import os
import pathlib
import subprocess
import sys
import traceback
from contextlib import contextmanager
from itertools import tee
from multiprocessing.pool import ThreadPool

import cv2 as cv
import numpy as np
import pytesseract
import scipy.fft
import tqdm
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from profanity_check import predict_prob

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

## pylint: disable=no-member


def full_stack():
    """Print out the full stack trace, including"""
    exc = sys.exc_info()[0]
    stack = traceback.extract_stack()[:-1]  # last one would be full_stack()
    if exc is not None:  # i.e. an exception is present
        del stack[-1]  # remove call of full_stack, the printed exception
        # will contain the caught exception caller instead
    trc = "Traceback (most recent call last):\n"
    stackstr = trc + "".join(traceback.format_list(stack))
    if exc is not None:
        ## pylint: disable=bad-str-strip-call
        stackstr += "  " + traceback.format_exc().lstrip(trc)
    return stackstr


def configure_logger(log_level, prefix_log_file: str = __name__ + "_log"):
    """Configures the logger for the application."""
    # create a directory logs if it does not exist
    pathlib.Path.mkdir(pathlib.Path("logs"), exist_ok=True)
    # Create a filename suffixed with current date DDMMYY format with
    # current date inside logs directory
    log_file = pathlib.Path("logs") / (
        f"{prefix_log_file}_{datetime.datetime.now().strftime('%Y%m%d')}.log"
    )
    # pylint: disable=line-too-long
    logging.basicConfig(
        format="%(asctime)s.%(msecs)d %(filename)s:%(lineno)d:%(funcName)s() %(levelname)s %(message)s",
        datefmt="%A,%d/%m/%Y|%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file),
        ],
        level=log_level,
    )

    return logging.getLogger(__name__)


class Frame:
    """Represents a frame in a video."""

    def __init__(self, frame_number, image, ts_second):
        self.frame_number = frame_number
        self.image = image
        self.ts_second = ts_second
        self.profanity_prob = 0.0
        self.threshold_image = self.post_process(self.image)

    def post_process(self, image):
        """Preprocesses the frame for OCR."""
        gray_frame = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        enhanced_frame = cv.equalizeHist(gray_frame)
        blurred_frame = cv.GaussianBlur(enhanced_frame, (5, 5), 0)
        _, threshold_frame = cv.threshold(
            blurred_frame, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU
        )
        return threshold_frame

    def __str__(self) -> str:
        return f"Frame {self.frame_number} at {self.ts_second} seconds"


## pylint: disable=too-many-instance-attributes,too-many-locals
class VideoOCR:
    """Performs OCR on a video and annotates it with the results."""

    ## pylint: disable=too-many-arguments
    def __init__(
        self,
        input_video,
        output_video,
        sample_rate=1,
        debug_dir="",
        annotate_only=False,
    ):
        self.filepath = input_video
        self.output_video = output_video
        self.sample_rate = sample_rate
        self.debug_dir = debug_dir

        ## create if not exists the debug directory
        if not os.path.exists(self.debug_dir):
            os.makedirs(self.debug_dir)
        self.frames = []
        self.analyze_frames = []
        self.pbar = self._NoOpProgressBar()
        self.annotate_only = annotate_only

        self.logger = logging.getLogger(__name__)

    class _NoOpProgressBar:
        """No-op progress bar for when tqdm is not available."""

        def update(self):
            """No-op"""
            pass  ## pylint: disable=unnecessary-pass

        def total(self, n):
            """No-op"""
            pass  ## pylint: disable=unnecessary-pass

    @contextmanager
    def _open_cv_video(self, filepath):
        cap = cv.VideoCapture(filepath)
        try:
            yield cap
        finally:
            cap.release()

    def phash(self, frame_image, hash_size=8, highfreq_factor=4):
        """Calculate the perceptual hash for the given frame."""
        image = cv.cvtColor(frame_image, cv.COLOR_BGR2GRAY)
        img_size = hash_size * highfreq_factor
        image = cv.resize(image, (img_size, img_size), interpolation=cv.INTER_LINEAR)
        dct = scipy.fft.dct(scipy.fft.dct(image, axis=0), axis=1)
        ## pylint: disable=invalid-sequence-index
        dctlowfreq = dct[:hash_size, :hash_size]
        med = np.median(dctlowfreq)
        diff = dctlowfreq > med
        return diff

    def _parallel_ocr(self, frames):
        with ThreadPool(multiprocessing.cpu_count()) as pool:
            return pool.map(self._ocr, frames, chunksize=multiprocessing.cpu_count())

    def _write_if_debug(self, frames, debug_dir):
        if not debug_dir:
            return
        for frame in frames:
            cv.imwrite(
                os.path.join(debug_dir, f"{frame.frame_number}.png"), frame.image
            )
            with open(
                os.path.join(debug_dir, f"{frame.frame_number}.txt"),
                "w",
                encoding="utf-8",
            ) as f:
                f.write(frame.text)

    def perform_video_ocr(self):
        """Performs OCR on the video and returns the frames with text."""
        frames = []
        with self._open_cv_video(self.filepath) as cap:
            frames = self._parallel_ocr(
                self._filter_redundant_frames(self._get_frames(cap, self.sample_rate))
            )
        frames.sort(key=lambda frame: frame.frame_number)
        non_empty_frames = []
        for frame in frames:
            if frame.text.strip():
                non_empty_frames.append(frame)
        self._write_if_debug(non_empty_frames, self.debug_dir)
        return non_empty_frames

    def _get_frames(self, video_capture, sample_rate):
        fps = int(video_capture.get(cv.CAP_PROP_FPS))
        self.pbar.total = (
            video_capture.get(cv.CAP_PROP_FRAME_COUNT) // (fps // sample_rate)
        ) - 1
        frame_number = 0
        while video_capture.isOpened():
            ret, frame = video_capture.read()
            if not ret:
                break
            frame_number += 1
            if frame_number % (fps // sample_rate) != 0:
                continue
            ##frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            yield Frame(frame_number, frame, frame_number // fps)

    def _pairwise(self, iterable):
        a, b = tee(iterable)
        next(b, None)
        return zip(a, b)

    def _are_similar_frame(self, f1, f2):
        ##return False
        diff = np.count_nonzero(self.phash(f1.image) != self.phash(f2.image))
        return diff <= 2

    def _get_time_stamp(self, seconds):
        rem_seconds = seconds
        hours = rem_seconds // 3600
        rem_seconds %= 3600
        mins = rem_seconds // 60
        rem_seconds %= 60
        return f"{int(hours):02}:{int(mins):02}:{int(rem_seconds):02}"

    def _display_frames(self, frames):
        terminal_width = os.get_terminal_size().columns
        self.logger.info("")
        for frame in frames:
            self.logger.info("-" * terminal_width)
            self.logger.info("Frame %04d", frame.frame_number)
            self.logger.info("Timestamp = %s", self._get_time_stamp(frame.ts_second))
            self.logger.info(frame.text)
            self.logger.info("Profanity : %.2f", frame.profanity_prob)
        self.logger.info("-" * terminal_width)

        ## write self.analyze_frames to a file
        with open(
            os.path.join(self.debug_dir, "analyze_frames.txt"), "w", encoding="utf-8"
        ) as f:
            for frame in self.analyze_frames:
                f.write(f"{frame[0]} {frame[1]}\n")

    def _filter_redundant_frames(self, frames):
        self.analyze_frames = []
        for f1, f2 in self._pairwise(frames):
            if not self._are_similar_frame(f1, f2):
                self.analyze_frames.append((f1.frame_number, -1))
                yield f1
            else:
                try:
                    last_analyze_frame = self.analyze_frames[-1][1]
                    if last_analyze_frame == f1.frame_number:
                        self.analyze_frames[-1] = (
                            self.analyze_frames[-1][0],
                            f2.frame_number,
                        )
                    else:
                        self.analyze_frames.append((f1.frame_number, f2.frame_number))
                except IndexError:
                    self.analyze_frames.append((f1.frame_number, f2.frame_number))
                self.pbar.update()

    def _ocr(self, frame):
        pil_image = Image.fromarray(frame.image)
        # text = pytesseract.image_to_string(pil_image)
        boxes = pytesseract.image_to_data(pil_image)  # , config=r"--psm 11 --oem 1")
        frame.text = ""
        for x, b in enumerate(boxes.splitlines()):
            if x != 0:
                b = b.split()
                if len(b) == 12:
                    (x, y, w, h) = map(int, b[6:10])
                    cv.rectangle(frame.image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv.putText(
                        frame.image,
                        b[11],
                        (x, y),
                        cv.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),  # Green color in BGR
                        2,
                    )
                    frame.text += b[11] + " "
        frame_prediction = predict_prob([frame.text])
        if frame_prediction[0] > 0.5:
            frame.profanity_prob = frame_prediction[0]
            cv.putText(
                frame.image,
                f"Profanity Prob:{frame.profanity_prob:.2f}",
                (10, 50),
                cv.FONT_ITALIC,
                1,
                (0, 0, 139),  # Dark red in BGR
                3,  # Increase thickness to make text appear bolder
            )
        self.pbar.update()
        return frame

    def annotate_video(
        self,
        original_video_path,
        output_video_path,
        output_folder="logs",
    ):
        """Annotates a video with analyzed frames."""
        cap = cv.VideoCapture(original_video_path)
        fourcc = cv.VideoWriter_fourcc(*"mp4v")
        frame_rate = cap.get(cv.CAP_PROP_FPS)
        frame_size = (
            int(cap.get(cv.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)),
        )
        frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        out = cv.VideoWriter(output_video_path, fourcc, frame_rate, frame_size)

        # Prepare lookup table for analyzed frames
        ## read from analyze_frames.txt
        analyze_frames = []
        with open(
            os.path.join(output_folder, "analyze_frames.txt"), "r", encoding="utf-8"
        ) as f:
            for line in f:
                start, end = line.strip().split()
                analyze_frames.append((int(start), int(end)))

        look_up = {
            frame_no[0]: (
                {"count": frame_no[1] - frame_no[0] + 1, "end": frame_no[1]}
                if frame_no[1] != -1
                else {"count": 1, "end": frame_no[0]}
            )
            for frame_no in analyze_frames
        }

        self.logger.info("Writing %s with %s frames.", output_video_path, frame_count)
        index = 1
        original_frames = 0
        progress_bar = tqdm.tqdm(range(1, frame_count + 1))
        for index in progress_bar:
            frame_written = False
            if index in look_up:
                file_name = f"{look_up[index]['end']}.png"
                count_frames = look_up[index]["count"]
                image = os.path.join(output_folder, file_name)
                if os.path.exists(image):
                    if original_frames > 0:
                        # self.logger.info(f"Original frames written: {original_frames}")
                        progress_bar.set_postfix(
                            {"Original frames written": original_frames}, refresh=True
                        )
                        original_frames = 0
                    frame = cv.imread(image)
                    progress_bar.set_postfix(
                        {
                            "Writing frame": look_up[index]["end"],
                            "Count": count_frames,
                            "Starting from frame": index,
                        },
                        refresh=True,
                    )

                    for _ in range(count_frames):
                        out.write(frame)
                    index += count_frames
                    frame_written = True
                else:
                    # self.logger.info(
                    #     f"Image {image} does not exist, for starting frame {index}. Skipping."
                    # )
                    progress_bar.set_postfix(
                        {"Image does not exist, skipping frame": index}, refresh=True
                    )

            if not frame_written:
                cap.set(cv.CAP_PROP_POS_FRAMES, index)
                _, frame = cap.read()
                out.write(frame)
                original_frames += 1
                index += 1

        cap.release()
        out.release()
        cv.destroyAllWindows()

    def run_ffmpeg_command_v2(
        self, original_video_path, output_video_path, output_with_audio_video
    ):
        """Runs the ffmpeg command to add audio to the annotated video."""
        ## Subprocess to ffmpeg for audio
        check = subprocess.run(
            [
                "ffmpeg",
                "-i",
                output_video_path,
                "-i",
                original_video_path,
                "-c:v",
                "h264",
                "-c:a",
                "aac",
                "-strict",
                "experimental",
                output_with_audio_video,
            ],
            check=True,
        )
        if check.returncode == 0:
            self.logger.info(
                "Successfully wrote annotated video with audio to %s",
                output_with_audio_video,
            )

    def create_canvas(self, frames, images_folder, row_size, col_size, output_size):
        """Create a collage of images from the specified directory."""
        images = []

        # Load all PNG images from the specified directory
        for frame in frames:
            if frame.profanity_prob > 0.5:
                filename = f"{images_folder}/{frame.frame_number}.png"
                img = Image.open(filename)
                images.append(
                    {
                        "img": img,
                        "frame_number": frame.frame_number,
                        "ts_second": frame.ts_second,
                    }
                )

        # Calculate the number of rows and columns
        num_rows = min(row_size, len(images))
        num_cols = min(col_size, (len(images) + row_size - 1) // row_size)

        # Resize images
        # resized_images = [{ "img": img.resize(output_size, Image.LANCZOS),
        #                   for item in images]
        resized_images = [
            {
                "img": item["img"].resize(output_size, Image.LANCZOS),
                "frame_number": item["frame_number"],
                "ts_second": item["ts_second"],
            }
            for item in images
        ]

        # Create a blank white canvas to arrange the images
        canvas_width = num_cols * output_size[0]
        canvas_height = num_rows * output_size[1]
        canvas = Image.new("RGB", (canvas_width, canvas_height), "white")

        # Paste the resized images onto the canvas
        for i, item in enumerate(resized_images):
            row = i // num_cols
            col = i % num_cols
            img = item["img"]
            draw = ImageDraw.Draw(img)
            font = ImageFont.load_default()  # Use default font, adjust as needed
            ## pylint: disable=line-too-long
            text = f"Frame {item['frame_number']}\nTimestamp = {self._get_time_stamp(item['ts_second'])}"
            draw.text(
                (10, 10), text, font=font, fill=(255, 255, 255)
            )  # Adjust position and color as needed
            canvas.paste(img, (col * output_size[0], row * output_size[1]))

        # Save the final image
        canvas.save(f"{images_folder}/output_collage.png")

    def run(self):
        """Runs the OCR on the video and annotates it."""
        if not self.annotate_only:
            with tqdm.tqdm() as progress_bar:
                self.pbar = progress_bar
                self.frames = self.perform_video_ocr()
            self._display_frames(self.frames)
            try:
                self.create_canvas(self.frames, self.debug_dir, 5, 5, (200, 200))
            except Exception as e:  ## pylint: disable=broad-exception-caught
                self.logger.error("Error creating canvas: %s", e)
            if not os.path.exists(self.debug_dir):
                os.makedirs(self.debug_dir)
        try:
            input_video = self.filepath
            output_video = self.output_video
            if output_video == "":
                output_video = os.path.join(self.debug_dir, "annotated_output.mp4")
            output_video_temp = os.path.join(
                self.debug_dir, "annotated_output_temp.mp4"
            )
            self.annotate_video(input_video, output_video_temp, self.debug_dir)
            self.logger.info(
                "Annotating %s with the OCR results and writing to %s",
                input_video,
                output_video,
            )
            self.run_ffmpeg_command_v2(input_video, output_video_temp, output_video)
            ## remove the temp file
            os.remove(output_video_temp)
        except Exception as e:  ## pylint: disable=broad-exception-caught
            ## display entire stack trace
            self.logger.error(full_stack())
            self.logger.error("Error annotating video: %s", e)


if __name__ == "__main__":
    configure_logger(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_video", help="Path to the input video file")
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=1,
        help="Number of frames to sample per second",
    )
    parser.add_argument(
        "--debug_dir",
        default="logs",
        help="If provided, writes frame and their respective texts here, for debugging",
    )
    ## annotate_only
    parser.add_argument(
        "--annotate_only",
        action="store_true",
        help="If provided, only annotates the video with OCR results",
    )
    ## output_video
    parser.add_argument(
        "--output_video",
        default="",
        help="If provided, writes the annotated video here",
    )
    args = parser.parse_args()

    logging.info("Running OCR on video %s", args.input_video)

    video_ocr = VideoOCR(
        args.input_video,
        args.output_video,
        args.sample_rate,
        args.debug_dir,
        args.annotate_only,
    )
    video_ocr.run()
