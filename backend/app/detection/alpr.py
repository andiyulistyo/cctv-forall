"""Automatic License-Plate Recognition (ANPR) for Indonesian plates.

This module is intentionally self-contained and optional: if EasyOCR (or the
model weights) cannot be loaded, :meth:`ALPR.available` is False and the worker
simply skips plate reading. OCR is done with EasyOCR (latin characters, which
covers Indonesian plates). An optional dedicated plate-detector (YOLO) can be
plugged in via ``plate_model``; otherwise a heuristic region of interest is
used before OCR.

Indonesian plate format (roughly): 1-2 area letters, 1-4 digits, 1-3 suffix
letters, e.g. "B 1234 XYZ" / "AD 12 AB".

Reading a motorcycle plate
--------------------------
A motorcycle plate is roughly half the width of a car plate, sits lower, and on
an overview camera arrives as a handful of pixels per character. Measured on
four sample captures from the junction camera, plain EasyOCR on the old
lower-45%-of-the-vehicle crop returned *nothing at all* on two of them and only
the expiry stamp on the other two. Three things were wrong, and all three are
fixed below:

* The ROI threw the plate away. On a crop that is already a plate, "the bottom
  45%" is the expiry line and not the number. The ROI is now a short list of
  candidate regions (detector box, whole crop, lower band) rather than one
  guess, and whichever actually reads wins.
* Nothing was upscaled. The old rule keyed off ``width < 200``, which a
  569 px-wide motorcycle crop passes while its plate is 60 px across -- so the
  upscale never fired on the frames that needed it. Size is now judged on the
  plate band, not the crop around it.
* The image went to OCR untouched. Contrast-limited equalisation and, above
  all, an edge-preserving denoise are what separate a read from a misread on a
  compressed 200x93 crop: bilateral filtering alone is the difference between
  "B4893UBY" and "BL893UBY" on sample 3.

Cost is bounded by escalation, which matters because the worker gives plate
reading a fixed budget and must never wait on it (see ``_PlateReader``). The
stages run cheapest-first and stop as soon as one of them produces a read the
recogniser is genuinely sure of, so a clean car plate still costs a single OCR
pass and only a plate that resisted the cheap stage pays for the rest.
"""
from __future__ import annotations

import re

from pathlib import Path

import cv2
import numpy as np

# Loose Indonesian plate pattern after removing spaces.
_PLATE_RE = re.compile(r"^[A-Z]{1,2}\d{1,4}[A-Z]{1,3}$")

# Plates carry no punctuation and no lower case. Telling EasyOCR so removes a
# whole class of misreads outright -- "B" as "8" survives, but "B" as ":" does
# not -- and costs nothing, because the decoder is simply given a smaller
# alphabet to search.
_ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

# Fewest digits a string must carry before it is allowed to be a plate. The
# format on its own is far too easy to satisfy by accident; see _candidates.
_MIN_PLATE_DIGITS = 2

# Confidence a plate needs when only one preprocessing variant produced it and
# nothing else agreed, and how much of that each further agreeing variant pays
# off. Never below the configured floor. See ALPR._result.
_LONE_READ_CONFIDENCE = 0.55
_AGREEMENT_CREDIT = 0.12

# The character pairs OCR actually confuses on a plate, kept deliberately
# short. Every entry here is a shape collision in the Indonesian plate font;
# guesses that merely "could" be wrong are left out, because a repair that
# invents a plate is worse than a read that fails. Note what is *absent*: L is
# not mapped to 1, because on sample 3 the character misread as "L" was a 4,
# and a map that has to choose between them would be wrong half the time.
_DIGIT_FIX = {"O": "0", "Q": "0", "D": "0", "I": "1", "Z": "2", "S": "5", "G": "6", "B": "8"}
_ALPHA_FIX = {"0": "O", "1": "I", "2": "Z", "5": "S", "6": "G", "8": "B"}


def normalize_plate(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def looks_like_plate(normalized: str) -> bool:
    if not (4 <= len(normalized) <= 9):
        return False
    return bool(_PLATE_RE.match(normalized))


def repair_plate(normalized: str) -> tuple[str, int] | None:
    """Coerce an OCR string into plate shape using only known confusions.

    Indonesian plates are positional -- letters, then digits, then letters --
    so the format itself says which characters a given position is allowed to
    hold, and "6" in the suffix block is a "G" that OCR got wrong. Every way of
    splitting the string into the three blocks is tried and the one needing the
    fewest substitutions wins; the cost is returned so the caller can prefer a
    read that needed no help over one that did.

    Returns ``(plate, substitutions)``, or None when no split works. A string
    that is already a valid plate comes back unchanged with a cost of 0.
    """
    n = len(normalized)
    if not (4 <= n <= 9):
        return None
    best: tuple[str, int] | None = None
    for area in (1, 2):
        for suffix in (1, 2, 3):
            digits = n - area - suffix
            if not (1 <= digits <= 4):
                continue
            out: list[str] = []
            cost = 0
            for i, ch in enumerate(normalized):
                if area <= i < area + digits:  # digit block
                    if ch.isdigit():
                        out.append(ch)
                    elif ch in _DIGIT_FIX:
                        out.append(_DIGIT_FIX[ch])
                        cost += 1
                    else:
                        break
                else:  # area or suffix block
                    if ch.isalpha():
                        out.append(ch)
                    elif ch in _ALPHA_FIX:
                        out.append(_ALPHA_FIX[ch])
                        cost += 1
                    else:
                        break
            else:
                if best is None or cost < best[1]:
                    best = ("".join(out), cost)
    return best


def _text_lines(results) -> list[tuple[str, float, tuple[int, int, int, int]]]:
    """Group EasyOCR boxes into text lines, read left to right.

    This is the single highest-yield step for motorcycles. Their plates put
    wide gaps between the area code, the number and the suffix, so EasyOCR
    routinely reports "B 6084 TXB" as two or three separate boxes -- none of
    which is a plate on its own, so the old per-box scoring threw all of them
    away. Joining *everything* instead is no good either: the expiry stamp
    below the number would be glued on the end ("B6084TXB0829") and fail the
    length check. A line is the unit that is actually a plate.

    Returns ``(text, confidence, bbox)`` per line, confidence averaged over the
    boxes in proportion to how many characters each contributed.
    """
    boxes = []
    for bbox, text, conf in results:
        pts = np.asarray(bbox, dtype=np.float32)
        x1, y1 = float(pts[:, 0].min()), float(pts[:, 1].min())
        x2, y2 = float(pts[:, 0].max()), float(pts[:, 1].max())
        norm = normalize_plate(text)
        if norm:
            boxes.append((norm, float(conf), x1, y1, x2, y2))
    if not boxes:
        return []

    boxes.sort(key=lambda b: (b[3] + b[5]) / 2.0)
    lines: list[list] = []
    for box in boxes:
        centre = (box[3] + box[5]) / 2.0
        height = box[5] - box[3]
        placed = False
        for line in lines:
            l_centre = sum((b[3] + b[5]) / 2.0 for b in line) / len(line)
            l_height = sum(b[5] - b[3] for b in line) / len(line)
            # Same line if the centres sit within half a character height of
            # each other -- tight enough to keep the expiry stamp separate,
            # loose enough to survive a plate photographed at a slight angle.
            if abs(centre - l_centre) <= 0.5 * max(height, l_height, 1.0):
                line.append(box)
                placed = True
                break
        if not placed:
            lines.append([box])

    out = []
    for line in lines:
        line.sort(key=lambda b: b[2])
        text = "".join(b[0] for b in line)
        chars = sum(len(b[0]) for b in line) or 1
        conf = sum(b[1] * len(b[0]) for b in line) / chars
        bbox = (
            int(min(b[2] for b in line)), int(min(b[3] for b in line)),
            int(max(b[4] for b in line)), int(max(b[5] for b in line)),
        )
        out.append((text, conf, bbox))
    return out


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def _upscale(gray: np.ndarray, min_height: int, min_width: int, cap: float = 6.0) -> np.ndarray:
    """Enlarge a crop until OCR has enough pixels per character to work with.

    Keyed off both dimensions, unlike the width-only rule this replaces: a
    plate band is wide and short, so width alone says nothing about whether its
    characters are tall enough to recognise. Cubic, and capped, because past a
    point interpolation is inventing detail rather than revealing it.
    """
    h, w = gray.shape[:2]
    if h <= 0 or w <= 0:
        return gray
    scale = min(max(min_height / h, min_width / w), cap)
    # A marginal enlargement is worse than none: interpolation softens every
    # edge it touches, and under 1.5x it buys too few pixels to pay that back.
    # Measured -- resizing sample 3 by the 1.2x its width asked for was enough
    # on its own to lose the read the untouched crop gives up.
    #
    # This floor used to swallow the common case rather than the marginal one.
    # Against the old 64/240 targets a 172x74 plate -- the median crop off
    # these cameras -- asked for 1.40x and was handed back untouched, and 57%
    # of reviewed crops fell in that gap; the settings now ask for enough that
    # a genuine plate clears the floor and only an already-large crop is left
    # alone. See settings.ocr_min_width for the measurement behind the targets.
    if scale < 1.5:
        return gray
    return cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def _equalize(gray: np.ndarray) -> np.ndarray:
    """Local contrast, so a plate half in shadow reads like one that is not."""
    return cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)


def _denoise(gray: np.ndarray) -> np.ndarray:
    """Edge-preserving denoise, then local contrast.

    Worth calling out: this is what actually recovers a compressed plate. A
    200x93 JPEG crop carries ringing around every stroke, and to the recogniser
    that ringing reads as extra strokes -- the "4" of B4893UBY loses its open
    top and comes back as an "L". Bilateral filtering flattens the noise while
    leaving the character edges where they are, and on that sample it is the
    only variant of the five tried that returns the right plate.
    """
    return _equalize(_smooth(gray))


def _smooth(gray: np.ndarray) -> np.ndarray:
    """The same denoise without the contrast stretch.

    Kept as a variant of its own rather than folded into :func:`_denoise`
    because equalisation is not always an improvement: on an evenly lit plate
    it amplifies the very JPEG ringing the filter just removed. Sample 3 reads
    correctly here and nowhere else.
    """
    return cv2.bilateralFilter(gray, 7, 60, 60)


def _sharpen(gray: np.ndarray) -> np.ndarray:
    """Unsharp mask over equalised grey; recovers slightly out-of-focus edges."""
    equalized = _equalize(gray)
    blurred = cv2.GaussianBlur(equalized, (0, 0), 3)
    return cv2.addWeighted(equalized, 1.7, blurred, -0.7, 0)


# Recogniser names EasyOCR ships and fetches itself. They take no directories
# and no local files, so they are never checked for on disk.
_STOCK_NETWORKS = frozenset({"standard", "english_g2", "latin_g2"})


def recogniser_kwargs(
    name: str, model_dir: str, network_dir: str
) -> tuple[dict, list[str]]:
    """EasyOCR arguments for a custom recogniser, and any files it is missing.

    Returns ``(kwargs, missing)``. The second half is the point of doing this
    here rather than letting ``easyocr.Reader`` raise: EasyOCR stops at the
    first absent file with a message about that one path, while a fine-tune
    arrives as three files and it is usually the ``.py`` that did not get
    copied. Reporting all of them at once turns three restarts into one.

    A missing file is deliberately *not* an error to the caller. Plate reading
    that refuses to start is a worse outcome than plate reading that runs the
    stock network for another day -- a mistyped name should cost accuracy, not
    ANPR -- so the caller logs ``missing`` and carries on with ``{}``. This is
    the same trade ``settings.plate_model_path`` makes, with the difference
    that here the fallback is announced rather than silent.
    """
    if not name or name == "standard":
        return {}, []
    if name in _STOCK_NETWORKS:
        # EasyOCR downloads these on demand; the directories would be ignored.
        return {"recog_network": name}, []

    models = Path(model_dir)
    nets = Path(network_dir or model_dir)
    required = [nets / f"{name}.yaml", nets / f"{name}.py", models / f"{name}.pth"]
    kwargs = {
        "recog_network": name,
        "model_storage_directory": str(models),
        "user_network_directory": str(nets),
    }
    return kwargs, [str(p) for p in required if not p.is_file()]


class ALPR:
    def __init__(
        self,
        languages: str = "en",
        device: str = "cpu",
        plate_model: str = "",
        plate_imgsz: int = 320,
        plate_conf: float = 0.25,
        min_confidence: float = 0.20,
        half: bool = False,
        ocr_min_height: int = 160,
        ocr_min_width: int = 600,
        good_enough: float = 0.75,
        max_passes: int = 8,
        recog_network: str = "",
        model_dir: str = "",
        network_dir: str = "",
    ):
        self._reader = None
        self._plate_detector = None
        self._available = False
        self.device = device or "cpu"
        self.plate_imgsz = plate_imgsz
        self.plate_conf = plate_conf
        self.min_confidence = min_confidence
        self.ocr_min_height = ocr_min_height
        self.ocr_min_width = ocr_min_width
        # Confidence at which the escalation stops early. Set it too low and a
        # marginal first-stage read blocks the stages that would have read the
        # plate properly; too high and every plate pays for every stage. It
        # matches the worker's _PLATE_GOOD_ENOUGH_CONF on purpose: a read that
        # stops the worker retrying this vehicle has nothing left to gain from
        # more variants of the one frame either.
        self.good_enough = good_enough
        # Ceiling on OCR passes for a single crop. The plate queue is eight
        # deep and drops what it cannot keep up with, so an unreadable crop
        # that works its way through every stage is not merely slow -- it costs
        # other vehicles their turn while producing nothing. Hitting the
        # ceiling ends the read with whatever has been collected so far.
        self.max_passes = max_passes
        # fp16 is a GPU-only win, exactly as for the vehicle detector.
        self.half = bool(half) and self.device != "cpu"
        # Which recogniser actually ended up loaded. Worth keeping rather than
        # assuming the configured one: every fallback below is a case where the
        # reader running is not the reader asked for, and an accuracy figure
        # means something different depending on which of them produced it.
        self.recog_network = "standard"
        try:
            import easyocr

            langs = [s.strip() for s in languages.split(",") if s.strip()] or ["en"]
            # EasyOCR accepts gpu=False, gpu=True or an explicit device string.
            # Passing the string keeps us in control (e.g. "mps" on Apple
            # Silicon) instead of relying on its own auto-detection.
            gpu_arg = False if self.device == "cpu" else self.device

            kwargs, missing = recogniser_kwargs(recog_network, model_dir, network_dir)
            if missing:
                print(
                    f"[ALPR] custom recogniser {recog_network!r} not loaded, missing: "
                    + ", ".join(missing)
                    + " -- reading with the stock network instead."
                )
                kwargs = {}
            self.recog_network = kwargs.get("recog_network", "standard")

            try:
                self._reader = easyocr.Reader(langs, gpu=gpu_arg, verbose=False, **kwargs)
            except Exception as exc:
                # Two different things fail here and they want different
                # answers: the accelerator (some EasyOCR builds choke on
                # non-CUDA devices) and the custom recogniser (a malformed
                # yaml, or a lang_list in it that does not cover OCR_LANGUAGES).
                # Drop the device first and keep the fine-tune -- it is the
                # whole reason anyone configured one, so it is given up only
                # once it is demonstrably the broken half.
                print(f"[ALPR] {self.device} unavailable ({exc}); falling back to CPU")
                self.device = "cpu"
                try:
                    self._reader = easyocr.Reader(langs, gpu=False, verbose=False, **kwargs)
                except Exception as exc2:
                    if not kwargs:
                        raise
                    print(
                        f"[ALPR] custom recogniser {self.recog_network!r} failed to "
                        f"load ({exc2}); reading with the stock network instead."
                    )
                    self.recog_network = "standard"
                    self._reader = easyocr.Reader(langs, gpu=False, verbose=False)
            self._available = True
            if self.recog_network != "standard":
                print(f"[ALPR] recogniser: {self.recog_network}")
        except Exception as exc:  # pragma: no cover - environment dependent
            print(f"[ALPR] disabled: could not init EasyOCR: {exc}")
            self._available = False

        if plate_model:
            try:
                from ultralytics import YOLO

                self._plate_detector = YOLO(plate_model)
                # Warm it on the target device: the first call on CUDA pays
                # context + autotune, and here that would land on a live frame.
                self._plate_detector.predict(
                    np.zeros((plate_imgsz, plate_imgsz, 3), dtype=np.uint8),
                    imgsz=self.plate_imgsz,
                    device=self.device,
                    half=self.half,
                    verbose=False,
                )
            except Exception as exc:
                print(f"[ALPR] plate detector disabled: {exc}")
                self._plate_detector = None

    @property
    def available(self) -> bool:
        return self._available

    def _detector_roi(self, vehicle_crop: np.ndarray) -> np.ndarray | None:
        """The plate box from the dedicated detector, if one is loaded."""
        if self._plate_detector is None:
            return None
        # Without an explicit device this silently ran on the ultralytics
        # default rather than alongside the vehicle detector.
        res = self._plate_detector.predict(
            vehicle_crop,
            imgsz=self.plate_imgsz,
            device=self.device,
            half=self.half,
            conf=self.plate_conf,
            verbose=False,
        )
        if res and res[0].boxes is not None and len(res[0].boxes) > 0:
            # Highest-confidence plate box.
            boxes = res[0].boxes
            idx = int(boxes.conf.cpu().numpy().argmax())
            x1, y1, x2, y2 = boxes.xyxy.cpu().numpy()[idx].astype(int)
            roi = vehicle_crop[max(0, y1):y2, max(0, x1):x2]
            if roi.size:
                return roi
        return None

    def locate_plate(self, vehicle_crop: np.ndarray) -> np.ndarray | None:
        """Where the plate is, without trying to read it.

        Finding a plate and reading one are different problems, and on a
        motorcycle they have different answers: the detector puts a tight box
        on the plate at 0.8 confidence in frames the recogniser then makes
        nothing of. Everything that happens to a read that fails -- the picture
        an operator reviews it from, the crop it contributes to a training
        set -- is better served by that box than by the whole vehicle it was
        found in, so the caller can ask for it on its own.

        Returns None when no detector is loaded or none was found; the caller
        keeps whatever it already had.
        """
        if vehicle_crop is None or vehicle_crop.size == 0:
            return None
        try:
            return self._detector_roi(vehicle_crop)
        except Exception:
            return None

    def _plate_rois(self, vehicle_crop: np.ndarray) -> list[np.ndarray]:
        """Regions that might hold the plate, best guess first.

        A list rather than the single guess this used to make. The old rule --
        always the bottom 45% of the crop -- is right for a car photographed
        whole and catastrophic for anything else: handed a crop that is already
        a plate it returns the expiry stamp, and on the two tight samples that
        is exactly the failure, OCR dutifully reading "0820" off an image whose
        number was sitting in the half that had been discarded. Offering the
        whole crop as well costs one extra OCR pass at most, since the ladder
        stops as soon as something reads.
        """
        detected = self._detector_roi(vehicle_crop)
        if detected is not None:
            return [detected]
        rois = [vehicle_crop]
        h = vehicle_crop.shape[0]
        band = vehicle_crop[int(h * 0.45):, :]
        # Only worth a separate pass when it is actually a different picture:
        # on a crop that is already a plate the band is a slice of the same
        # characters, and the whole crop has already been tried.
        if band.size and h >= 80:
            rois.append(band)
        return rois

    def _ocr(self, image: np.ndarray) -> list:
        """One EasyOCR pass, tuned for plates.

        ``mag_ratio`` doubles the detector's working resolution, which is what
        lets it find text at all on a small plate; the two thresholds are
        lowered from their defaults because plate characters are isolated
        strokes on a plain field rather than words in a paragraph, and the
        stock settings drop them.
        """
        try:
            return self._reader.readtext(
                image,
                allowlist=_ALLOWLIST,
                mag_ratio=2.0,
                text_threshold=0.5,
                low_text=0.3,
            )
        except Exception:
            return []

    def _candidates(self, results, lines) -> list[tuple[str, float, int]]:
        """Every plate-shaped string one OCR pass supports.

        Returns ``(plate, confidence, repair_cost)`` per candidate. Lines come
        first, then the individual boxes they were built from -- a line that
        merged one box too many still leaves the good box behind, and a line
        that merged exactly right is the only place a split plate exists.
        """
        candidates = list(lines)
        for bbox, text, conf in results:
            norm = normalize_plate(text)
            if norm:
                pts = np.asarray(bbox, dtype=np.float32)
                candidates.append((norm, float(conf), (
                    int(pts[:, 0].min()), int(pts[:, 1].min()),
                    int(pts[:, 0].max()), int(pts[:, 1].max()),
                )))

        out = []
        for norm, conf, _bbox in candidates:
            repaired = repair_plate(norm)
            if repaired is None:
                continue
            plate, cost = repaired
            if sum(c.isdigit() for c in plate) < _MIN_PLATE_DIGITS:
                # A plate with one digit is a plate this did not read. The
                # format alone is far too easy to satisfy by accident: on
                # sample 1, whose plate is illegible, fragments of the tail
                # light came back as "G6JJO" -- one area letter, one digit,
                # three suffix letters, a perfectly well-formed plate and
                # entirely imaginary. Requiring two digits costs nothing real
                # (a road-going Indonesian plate carries three or four) and
                # removes the whole class of accident.
                continue
            out.append((plate, conf, cost))
        return out

    def read_plate(self, vehicle_crop: np.ndarray) -> tuple[str, float, np.ndarray] | None:
        """Try to read a plate from a vehicle crop.

        Returns ``(plate_text, confidence, plate_image)`` or None.

        The work escalates. Each candidate region is shown to OCR as equalised
        grey first, which is where an ordinary car plate is read and done with;
        then, only if that produced nothing convincing, as the two denoises and
        the sharpen, which is where a compressed motorcycle plate is won.
        Failing all of those, the widest run of text found so far is cropped
        out and zoomed hard, for the case the plate is a small part of a large
        vehicle crop and was never big enough to read where it sat.

        Nothing here trusts a single reading. Every pass votes, and the winner
        is decided in :meth:`_result` -- see there for why agreement matters
        more than any one pass's confidence. The ladder stops early on a read
        above ``good_enough``, and stops regardless after ``max_passes``.
        """
        if not self._available or vehicle_crop is None or vehicle_crop.size == 0:
            return None

        # plate -> [best confidence, votes, cheapest repair, roi it came from]
        votes: dict[str, list] = {}
        seen: list[tuple] = []  # (bbox, image, roi) from the passes so far

        budget = self.max_passes

        def consider(image: np.ndarray, roi: np.ndarray) -> bool:
            """Run one pass and record what it saw; True to stop the ladder."""
            nonlocal budget
            if budget <= 0:
                return True
            budget -= 1
            results = self._ocr(image)
            if not results:
                return False
            lines = _text_lines(results)
            for _text, _conf, bbox in lines:
                seen.append((bbox, image, roi))
            # One pass is one opinion. The candidate list deliberately holds a
            # joined line *and* the boxes it was joined from, so the same
            # string can turn up twice in a single pass -- counting both would
            # let one reading corroborate itself and walk straight through the
            # agreement rule below.
            pass_best: dict[str, tuple[float, int]] = {}
            for plate, conf, cost in self._candidates(results, lines):
                prior = pass_best.get(plate)
                if prior is None:
                    pass_best[plate] = (conf, cost)
                else:
                    pass_best[plate] = (max(conf, prior[0]), min(cost, prior[1]))

            strong = False
            for plate, (conf, cost) in pass_best.items():
                entry = votes.get(plate)
                if entry is None:
                    votes[plate] = [conf, 1, cost, roi]
                else:
                    entry[1] += 1
                    entry[2] = min(entry[2], cost)
                    if conf > entry[0]:
                        entry[0], entry[3] = conf, roi
                if conf >= self.good_enough:
                    strong = True
            return strong

        for roi in self._plate_rois(vehicle_crop):
            if roi is None or roi.size == 0:
                continue
            gray = _upscale(_to_gray(roi), self.ocr_min_height, self.ocr_min_width)
            # Deliberately *not* first-past-the-post. An early version stopped
            # at the first plate-shaped read above the floor, and on sample 3
            # that was "L893UBY" from the cheap stage -- well-formed, confident
            # and wrong, returned without ever running the denoise that reads
            # the plate correctly. Only a read the recogniser is genuinely sure
            # of ends the search; anything less collects a second opinion.
            if consider(_equalize(gray), roi):
                break
            if consider(_denoise(gray), roi):
                break
            if consider(_smooth(gray), roi):
                break
            if consider(_sharpen(gray), roi):
                break

        # The plate may be in there but too small to have been read where it
        # sat. Zoom the widest run of text found so far and look again; on a
        # full motorcycle crop that run *is* the plate. Only worth doing while
        # nothing convincing has turned up.
        if not any(v[0] >= self.good_enough for v in votes.values()):
            widest = sorted(seen, key=lambda s: s[0][2] - s[0][0], reverse=True)[:2]
            for bbox, image, _roi in widest:
                x1, y1, x2, y2 = bbox
                pad_x, pad_y = int((x2 - x1) * 0.08), int((y2 - y1) * 0.35)
                h, w = image.shape[:2]
                band = image[max(0, y1 - pad_y):min(h, y2 + pad_y),
                             max(0, x1 - pad_x):min(w, x2 + pad_x)]
                if band.size == 0 or band.shape[0] < 8:
                    continue
                zoomed = _upscale(_to_gray(band), 128, 480)
                if consider(_equalize(zoomed), band) or consider(_denoise(zoomed), band):
                    break

        return self._result(votes)

    def _result(self, votes: dict) -> tuple[str, float, np.ndarray] | None:
        """Pick a winner from the collected votes and apply the accept floor.

        Agreement is evidence. Each preprocessing variant distorts the crop
        differently, so two of them landing on the same string is a far
        stronger signal than either one's own confidence -- while a misread
        tends to be a different misread every time. Confidence still leads;
        agreement and a repair-free reading break the ties.
        """
        if not votes:
            return None
        plate, (conf, count, cost, roi) = max(
            votes.items(),
            key=lambda kv: kv[1][0] + 0.12 * (kv[1][1] - 1) - 0.08 * kv[1][2],
        )
        # Require a plausible plate to cut down on noise.
        if not looks_like_plate(plate):
            return None
        # ...and require the OCR to have actually been sure of it. The pattern
        # check alone is weak: a blurry plate at distance still yields a string
        # that matches it, just not the right one. Returning None here leaves
        # the track unresolved so a closer frame gets another go.
        if conf < self.min_confidence:
            return None
        # Agreement buys confidence, but never all of it. Running several
        # preprocessings makes a flat floor easier to clear by luck than it
        # was: on sample 1, whose plate is genuinely illegible, each variant
        # invented a *different* well-formed plate at 0.21-0.26, and on sample
        # 4 two of them agreed on a wrong one at 0.23 -- both would have been
        # stored as fact. So an uncorroborated read has to stand up on its own,
        # each additional variant that reproduces it takes something off that
        # bar, and enough of them bring it back down to the configured floor
        # and no lower. A plate several independent distortions of the crop all
        # read the same way is the one case a weak score is still worth having.
        if conf < max(self.min_confidence,
                      _LONE_READ_CONFIDENCE - _AGREEMENT_CREDIT * (count - 1)):
            return None
        return plate, conf, roi
