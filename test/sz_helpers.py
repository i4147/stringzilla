import os
import io
import contextlib
import tempfile
import zipfile
import urllib.request
import xml.etree.ElementTree as ET
from random import choice, randint, seed
from string import ascii_lowercase
from typing import Dict, List, Optional
try:
    import numpy as np
    numpy_available = True
except:
    numpy_available = False
try:
    import pyarrow as pa
    pyarrow_available = True
except:
    pyarrow_available = False
def _write_cache_atomically(cache_path: str, payload: bytes) -> None:
    directory = os.path.dirname(cache_path) or "."
    handle, staging_path = tempfile.mkstemp(dir=directory, suffix=".partial")
    try:
        with os.fdopen(handle, "wb") as staging_file:
            staging_file.write(payload)
        os.replace(staging_path, cache_path)
    except BaseException:
        try:
            os.unlink(staging_path)
        except OSError:
            pass
        raise
class UnicodeDataDownloadError(Exception):
    pass
class UnicodeDataParseError(Exception):
    pass
def _parsed_or_raise(entries, cache_path: str, description: str):
    if not entries:
        raise UnicodeDataParseError(f"No {description} parsed from {cache_path}")
    return entries
UNICODE_VERSION = "17.0.0"
def get_unicode_xml_data(version: str = UNICODE_VERSION) -> ET.Element:
    cache_path = os.path.join(tempfile.gettempdir(), f"ucd-{version}.all.flat.xml")
    if not os.path.exists(cache_path):
        url = f"https://www.unicode.org/Public/{version}/ucdxml/ucd.all.flat.zip"
        print(f"Downloading Unicode {version} UCD XML from {url}...")
        try:
            with urllib.request.urlopen(url) as response:
                zip_data = response.read()
            zip_bytes = io.BytesIO(zip_data)
            with zipfile.ZipFile(zip_bytes) as zip_file:
                xml_filename = zip_file.namelist()[0]
                with zip_file.open(xml_filename) as xml_file:
                    xml_content = xml_file.read()
                    _write_cache_atomically(cache_path, xml_content)
            print(f"Cached to {cache_path}")
        except Exception as error:
            raise UnicodeDataDownloadError(f"Could not download UCD XML from {url}: {error}")
    else:
        print(f"Using cached Unicode {version} UCD XML: {cache_path}")
    tree = ET.parse(cache_path)
    return tree.getroot()
def get_all_codepoints(version: str = UNICODE_VERSION) -> List[int]:
    root = get_unicode_xml_data(version)
    namespace = {"ucd": "http://www.unicode.org/ns/2003/ucd/1.0"}
    codepoints = []
    for char in root.findall(".//ucd:char", namespace):
        codepoint_string = char.get("cp")
        if codepoint_string:
            codepoints.append(int(codepoint_string, 16))
    return sorted(codepoints)
def parse_uncased_folding_file(filepath: str) -> Dict[int, bytes]:
    folds = {}
    with open(filepath, "r", encoding="utf-8") as data_file:
        for line in data_file:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(";")
            if len(parts) < 3:
                continue
            status = parts[1].strip()
            if status not in ("C", "F"):
                continue
            try:
                codepoint = int(parts[0].strip(), 16)
                target_codepoints = [int(hex_part, 16) for hex_part in parts[2].split("#")[0].strip().split()]
                folded_string = "".join(chr(codepoint) for codepoint in target_codepoints)
                folds[codepoint] = folded_string.encode("utf-8")
            except (ValueError, IndexError):
                continue
    return folds
def _download_uncased_folding_file(version: str) -> str:
    cache_path = os.path.join(tempfile.gettempdir(), f"CaseFolding-{version}.txt")
    if not os.path.exists(cache_path):
        url = f"https://www.unicode.org/Public/{version}/ucd/CaseFolding.txt"
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                _write_cache_atomically(cache_path, response.read())
        except Exception as error:
            raise UnicodeDataDownloadError(f"Could not download CaseFolding.txt from {url}: {error}")
    return cache_path
def get_uncased_folding_rules(version: str = UNICODE_VERSION) -> Dict[int, bytes]:
    cache_path = _download_uncased_folding_file(version)
    return parse_uncased_folding_file(cache_path)
def get_uncased_folding_rules_as_codepoints(
    version: str = UNICODE_VERSION,
) -> Dict[int, List[int]]:
    cache_path = _download_uncased_folding_file(version)
    folds = {}
    with open(cache_path, "r", encoding="utf-8") as data_file:
        for line in data_file:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(";")
            if len(parts) < 3:
                continue
            status = parts[1].strip()
            if status not in ("C", "F"):
                continue
            try:
                source_codepoint = int(parts[0].strip(), 16)
                target_codepoints = [int(hex_part, 16) for hex_part in parts[2].split("#")[0].strip().split()]
                folds[source_codepoint] = target_codepoints
            except (ValueError, IndexError):
                continue
    return folds
def get_normalization_props(version: str = UNICODE_VERSION) -> Dict[int, Dict[str, str]]:
    cache_path = os.path.join(tempfile.gettempdir(), f"DerivedNormalizationProps-{version}.txt")
    if not os.path.exists(cache_path):
        url = f"https://www.unicode.org/Public/{version}/ucd/DerivedNormalizationProps.txt"
        print(f"Downloading Unicode {version} DerivedNormalizationProps.txt from {url}...")
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                _write_cache_atomically(cache_path, response.read())
            print(f"Cached to {cache_path}")
        except Exception as error:
            raise UnicodeDataDownloadError(f"Could not download DerivedNormalizationProps.txt from {url}: {error}")
    else:
        print(f"Using cached Unicode {version} DerivedNormalizationProps.txt: {cache_path}")
    properties: Dict[int, Dict[str, str]] = {}
    with open(cache_path, "r", encoding="utf-8") as data_file:
        for line in data_file:
            line = line.split("#")[0].strip()
            if not line:
                continue
            parts = line.split(";")
            if len(parts) < 2:
                continue
            try:
                codepoint_range = parts[0].strip()
                property_field = parts[1].strip()
                if len(parts) >= 3:
                    property_name = property_field
                    property_default = parts[2].strip()
                else:
                    property_name = property_field
                    property_default = "Y"
                if ".." in codepoint_range:
                    start, end = codepoint_range.split("..")
                    start_codepoint = int(start, 16)
                    end_codepoint = int(end, 16)
                else:
                    start_codepoint = end_codepoint = int(codepoint_range, 16)
                for codepoint in range(start_codepoint, end_codepoint + 1):
                    if codepoint not in properties:
                        properties[codepoint] = {}
                    properties[codepoint][property_name] = property_default
            except (ValueError, IndexError):
                continue
    return properties
def _download_word_break_property_file(version: str) -> str:
    cache_path = os.path.join(tempfile.gettempdir(), f"WordBreakProperty-{version}.txt")
    if not os.path.exists(cache_path):
        url = f"https://www.unicode.org/Public/{version}/ucd/auxiliary/WordBreakProperty.txt"
        print(f"Downloading Unicode {version} WordBreakProperty.txt from {url}...")
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                _write_cache_atomically(cache_path, response.read())
            print(f"Cached to {cache_path}")
        except Exception as error:
            raise UnicodeDataDownloadError(f"Could not download WordBreakProperty.txt from {url}: {error}")
    return cache_path
def get_word_break_properties(version: str = UNICODE_VERSION) -> Dict[int, str]:
    cache_path = _download_word_break_property_file(version)
    properties: Dict[int, str] = {}
    with open(cache_path, "r", encoding="utf-8") as data_file:
        for line in data_file:
            line = line.split("#")[0].strip()
            if not line:
                continue
            parts = line.split(";")
            if len(parts) < 2:
                continue
            try:
                codepoint_range = parts[0].strip()
                property_name = parts[1].strip()
                if ".." in codepoint_range:
                    start, end = codepoint_range.split("..")
                    start_codepoint = int(start, 16)
                    end_codepoint = int(end, 16)
                else:
                    start_codepoint = end_codepoint = int(codepoint_range, 16)
                for codepoint in range(start_codepoint, end_codepoint + 1):
                    properties[codepoint] = property_name
            except (ValueError, IndexError):
                continue
    return properties
def _download_word_break_test_file(version: str) -> str:
    cache_path = os.path.join(tempfile.gettempdir(), f"WordBreakTest-{version}.txt")
    if not os.path.exists(cache_path):
        url = f"https://www.unicode.org/Public/{version}/ucd/auxiliary/WordBreakTest.txt"
        print(f"Downloading Unicode {version} WordBreakTest.txt from {url}...")
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                _write_cache_atomically(cache_path, response.read())
            print(f"Cached to {cache_path}")
        except Exception as error:
            raise UnicodeDataDownloadError(f"Could not download WordBreakTest.txt from {url}: {error}")
    return cache_path
def get_word_break_test_cases(version: str = UNICODE_VERSION) -> List[tuple]:
    cache_path = _download_word_break_test_file(version)
    test_cases = []
    with open(cache_path, "r", encoding="utf-8") as data_file:
        for line in data_file:
            line = line.split("#")[0].strip()
            if not line:
                continue
            parts = line.replace("÷", " ÷ ").replace("×", " × ").split()
            codepoints = []
            boundaries = []
            byte_position = 0
            for part in parts:
                part = part.strip()
                if part == "÷":
                    boundaries.append(byte_position)
                elif part == "×":
                    pass
                elif part:
                    try:
                        codepoint = int(part, 16)
                        codepoints.append(codepoint)
                        byte_position += len(chr(codepoint).encode("utf-8"))
                    except ValueError:
                        continue
            if codepoints:
                text = "".join(chr(codepoint) for codepoint in codepoints)
                test_cases.append((text, boundaries))
    return test_cases
def baseline_word_boundaries(text: str, word_break_properties: Dict[int, str] = None) -> List[int]:
    if word_break_properties is None:
        word_break_properties = get_word_break_properties()
    def word_break_property_of(codepoint: int) -> str:
        return word_break_properties.get(codepoint, "Other")
    codepoints = []
    byte_offset = 0
    for char in text:
        codepoint = ord(char)
        codepoint_bytes = char.encode("utf-8")
        codepoints.append((codepoint, byte_offset, len(codepoint_bytes)))
        byte_offset += len(codepoint_bytes)
    if not codepoints:
        return [0]
    total_bytes = byte_offset
    boundaries = [0]
    def is_ignorable(property_value: str) -> bool:
        return property_value in ("Extend", "Format", "ZWJ")
    def effective_property_at(relative_index: int, direction: int = 1) -> str:
        while 0 <= relative_index < len(codepoints):
            property_value = word_break_property_of(codepoints[relative_index][0])
            if not is_ignorable(property_value):
                return property_value
            relative_index += direction
        return "Other"
    def is_aletter_or_hebrew_letter(property_value: str) -> bool:
        return property_value in ("ALetter", "Hebrew_Letter")
    def is_midnumlet_or_single_quote(property_value: str) -> bool:
        return property_value in ("MidNumLet", "Single_Quote")
    for index in range(1, len(codepoints)):
        boundary_byte_position = codepoints[index][1]
        previous_codepoint = codepoints[index - 1][0]
        current_codepoint = codepoints[index][0]
        previous_property = word_break_property_of(previous_codepoint)
        current_property = word_break_property_of(current_codepoint)
        if previous_property == "CR" and current_property == "LF":
            continue
        if previous_property in ("CR", "LF", "Newline"):
            boundaries.append(boundary_byte_position)
            continue
        if current_property in ("CR", "LF", "Newline"):
            boundaries.append(boundary_byte_position)
            continue
        if previous_property == "ZWJ":
            continue
        if previous_property == "WSegSpace" and current_property == "WSegSpace":
            continue
        effective_previous = previous_property
        if is_ignorable(previous_property):
            for lookahead_index in range(index - 2, -1, -1):
                scanned_property = word_break_property_of(codepoints[lookahead_index][0])
                if not is_ignorable(scanned_property):
                    effective_previous = scanned_property
                    break
            else:
                effective_previous = "Other"
        effective_current = current_property
        if is_aletter_or_hebrew_letter(effective_previous) and is_aletter_or_hebrew_letter(effective_current):
            continue
        if is_aletter_or_hebrew_letter(effective_previous) and current_property in (
            "MidLetter",
            "MidNumLet",
            "Single_Quote",
        ):
            for lookahead_index in range(index + 1, len(codepoints)):
                next_property = word_break_property_of(codepoints[lookahead_index][0])
                if is_ignorable(next_property):
                    continue
                if is_aletter_or_hebrew_letter(next_property):
                    break
                else:
                    boundaries.append(boundary_byte_position)
                    break
            else:
                boundaries.append(boundary_byte_position)
            continue
        if current_property in ("MidLetter", "MidNumLet", "Single_Quote"):
            pass
        if is_midnumlet_or_single_quote(effective_previous) or effective_previous == "MidLetter":
            if is_aletter_or_hebrew_letter(effective_current):
                for lookahead_index in range(index - 2, -1, -1):
                    scanned_property = word_break_property_of(codepoints[lookahead_index][0])
                    if is_ignorable(scanned_property):
                        continue
                    if is_aletter_or_hebrew_letter(scanned_property):
                        break
                    else:
                        boundaries.append(boundary_byte_position)
                        break
                else:
                    boundaries.append(boundary_byte_position)
                continue
        if effective_previous == "Numeric" and effective_current == "Numeric":
            continue
        if is_aletter_or_hebrew_letter(effective_previous) and effective_current == "Numeric":
            continue
        if effective_previous == "Numeric" and is_aletter_or_hebrew_letter(effective_current):
            continue
        if effective_previous == "Numeric" and current_property in ("MidNum", "MidNumLet", "Single_Quote"):
            for lookahead_index in range(index + 1, len(codepoints)):
                next_property = word_break_property_of(codepoints[lookahead_index][0])
                if is_ignorable(next_property):
                    continue
                if next_property == "Numeric":
                    break
                else:
                    boundaries.append(boundary_byte_position)
                    break
            else:
                boundaries.append(boundary_byte_position)
            continue
        if current_property in ("MidNum", "MidNumLet", "Single_Quote") and effective_current == "Numeric":
            pass
        if effective_previous == "Katakana" and effective_current == "Katakana":
            continue
        if effective_previous in ("ALetter", "Hebrew_Letter", "Numeric", "Katakana", "ExtendNumLet"):
            if effective_current == "ExtendNumLet":
                continue
        if effective_previous == "ExtendNumLet":
            if effective_current in ("ALetter", "Hebrew_Letter", "Numeric", "Katakana"):
                continue
        if effective_previous == "Regional_Indicator" and effective_current == "Regional_Indicator":
            regional_indicator_count = 0
            for lookahead_index in range(index - 1, -1, -1):
                scanned_property = word_break_property_of(codepoints[lookahead_index][0])
                if scanned_property == "Regional_Indicator":
                    regional_indicator_count += 1
                elif is_ignorable(scanned_property):
                    continue
                else:
                    break
            if regional_indicator_count % 2 == 1:
                continue
        boundaries.append(boundary_byte_position)
    boundaries.append(total_bytes)
    return sorted(set(boundaries))
def _download_break_property_file(filename: str, version: str) -> str:
    cache_path = os.path.join(tempfile.gettempdir(), f"{filename[:-4]}-{version}.txt")
    subdir = "ucd" if filename == "LineBreak.txt" else "ucd/auxiliary"
    if not os.path.exists(cache_path):
        url = f"https://www.unicode.org/Public/{version}/{subdir}/{filename}"
        print(f"Downloading Unicode {version} {filename} from {url}...")
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                _write_cache_atomically(cache_path, response.read())
            print(f"Cached to {cache_path}")
        except Exception as error:
            raise UnicodeDataDownloadError(f"Could not download {filename} from {url}: {error}")
    return cache_path
def _parse_break_property_file(cache_path: str) -> Dict[int, str]:
    properties: Dict[int, str] = {}
    with open(cache_path, "r", encoding="utf-8") as data_file:
        for line in data_file:
            line = line.split("#")[0].strip()
            if not line:
                continue
            parts = line.split(";")
            if len(parts) < 2:
                continue
            try:
                codepoint_range = parts[0].strip()
                property_name = parts[1].strip()
                if ".." in codepoint_range:
                    start, end = codepoint_range.split("..")
                    start_codepoint = int(start, 16)
                    end_codepoint = int(end, 16)
                else:
                    start_codepoint = end_codepoint = int(codepoint_range, 16)
                for codepoint in range(start_codepoint, end_codepoint + 1):
                    properties[codepoint] = property_name
            except (ValueError, IndexError):
                continue
    return _parsed_or_raise(properties, cache_path, "break properties")
def _download_break_test_file(filename: str, version: str) -> str:
    cache_path = os.path.join(tempfile.gettempdir(), f"{filename[:-4]}-{version}.txt")
    if not os.path.exists(cache_path):
        url = f"https://www.unicode.org/Public/{version}/ucd/auxiliary/{filename}"
        print(f"Downloading Unicode {version} {filename} from {url}...")
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                _write_cache_atomically(cache_path, response.read())
            print(f"Cached to {cache_path}")
        except Exception as error:
            raise UnicodeDataDownloadError(f"Could not download {filename} from {url}: {error}")
    return cache_path
def _parse_break_test_file(cache_path: str) -> List[tuple]:
    test_cases = []
    with open(cache_path, "r", encoding="utf-8") as data_file:
        for line in data_file:
            line = line.split("#")[0].strip()
            if not line:
                continue
            parts = line.replace("÷", " ÷ ").replace("×", " × ").split()
            codepoints = []
            boundaries = []
            byte_position = 0
            for part in parts:
                part = part.strip()
                if part == "÷":
                    boundaries.append(byte_position)
                elif part == "×":
                    pass
                elif part:
                    try:
                        codepoint = int(part, 16)
                        codepoints.append(codepoint)
                        byte_position += len(chr(codepoint).encode("utf-8"))
                    except ValueError:
                        continue
            if codepoints:
                text = "".join(chr(codepoint) for codepoint in codepoints)
                test_cases.append((text, boundaries))
    return _parsed_or_raise(test_cases, cache_path, "break-test cases")
def get_grapheme_break_properties(version: str = UNICODE_VERSION) -> Dict[int, str]:
    cache_path = _download_break_property_file("GraphemeBreakProperty.txt", version)
    return _parse_break_property_file(cache_path)
def get_indic_conjunct_break_properties(version: str = UNICODE_VERSION) -> Dict[int, str]:
    cache_path = os.path.join(tempfile.gettempdir(), f"DerivedCoreProperties-{version}.txt")
    if not os.path.exists(cache_path):
        url = f"https://www.unicode.org/Public/{version}/ucd/DerivedCoreProperties.txt"
        print(f"Downloading Unicode {version} DerivedCoreProperties.txt from {url}...")
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                _write_cache_atomically(cache_path, response.read())
            print(f"Cached to {cache_path}")
        except Exception as error:
            raise UnicodeDataDownloadError(f"Could not download DerivedCoreProperties.txt from {url}: {error}")
    indic_conjunct_breaks: Dict[int, str] = {}
    with open(cache_path, "r", encoding="utf-8") as data_file:
        for line in data_file:
            line = line.split("#")[0].strip()
            parts = [part.strip() for part in line.split(";")]
            if len(parts) < 3 or parts[1] != "InCB":
                continue
            codepoint_range = parts[0]
            if ".." in codepoint_range:
                start_codepoint, end_codepoint = (int(bound, 16) for bound in codepoint_range.split(".."))
            else:
                start_codepoint = end_codepoint = int(codepoint_range, 16)
            for codepoint in range(start_codepoint, end_codepoint + 1):
                indic_conjunct_breaks[codepoint] = parts[2]
    return _parsed_or_raise(indic_conjunct_breaks, cache_path, "Indic_Conjunct_Break entries")
def get_grapheme_break_test_cases(version: str = UNICODE_VERSION) -> List[tuple]:
    cache_path = _download_break_test_file("GraphemeBreakTest.txt", version)
    return _parse_break_test_file(cache_path)
def baseline_grapheme_boundaries(
    text: str,
    properties: Dict[int, str] = None,
    indic_conjunct_breaks: Dict[int, str] = None,
    extended_pictographic: set = None,
) -> List[int]:
    if properties is None:
        properties = get_grapheme_break_properties()
    if indic_conjunct_breaks is None:
        indic_conjunct_breaks = get_indic_conjunct_break_properties()
    if extended_pictographic is None:
        extended_pictographic = get_extended_pictographic()
    def property_of(codepoint: int) -> str:
        return properties.get(codepoint, "Other")
    def indic_conjunct_of(codepoint: int) -> str:
        return indic_conjunct_breaks.get(codepoint, "None")
    codepoints = []
    byte_offset = 0
    for char in text:
        codepoint = ord(char)
        codepoint_bytes = char.encode("utf-8")
        codepoints.append((codepoint, byte_offset, len(codepoint_bytes)))
        byte_offset += len(codepoint_bytes)
    if not codepoints:
        return [0]
    total_bytes = byte_offset
    boundaries = [0]
    def is_extended_pictographic(codepoint: int) -> bool:
        return codepoint in extended_pictographic
    for index in range(1, len(codepoints)):
        boundary_byte_position = codepoints[index][1]
        previous_codepoint = codepoints[index - 1][0]
        current_codepoint = codepoints[index][0]
        previous_property = property_of(previous_codepoint)
        current_property = property_of(current_codepoint)
        if previous_property == "CR" and current_property == "LF":
            continue
        if previous_property in ("Control", "CR", "LF"):
            boundaries.append(boundary_byte_position)
            continue
        if current_property in ("Control", "CR", "LF"):
            boundaries.append(boundary_byte_position)
            continue
        if previous_property == "L" and current_property in ("L", "V", "LV", "LVT"):
            continue
        if previous_property in ("LV", "V") and current_property in ("V", "T"):
            continue
        if previous_property in ("LVT", "T") and current_property == "T":
            continue
        if current_property in ("Extend", "ZWJ"):
            continue
        if current_property == "SpacingMark":
            continue
        if previous_property == "Prepend":
            continue
        if indic_conjunct_of(current_codepoint) == "Consonant":
            seen_linker = False
            scan_index = index - 1
            while scan_index >= 0 and indic_conjunct_of(codepoints[scan_index][0]) in ("Extend", "Linker"):
                if indic_conjunct_of(codepoints[scan_index][0]) == "Linker":
                    seen_linker = True
                scan_index -= 1
            if scan_index >= 0 and seen_linker and indic_conjunct_of(codepoints[scan_index][0]) == "Consonant":
                continue
        if previous_property == "ZWJ" and is_extended_pictographic(current_codepoint):
            lookahead_index = index - 1
            scan_index = lookahead_index - 1
            while scan_index >= 0 and property_of(codepoints[scan_index][0]) == "Extend":
                scan_index -= 1
            if scan_index >= 0 and is_extended_pictographic(codepoints[scan_index][0]):
                continue
        if previous_property == "Regional_Indicator" and current_property == "Regional_Indicator":
            regional_indicator_count = 0
            for scan_index in range(index - 1, -1, -1):
                if property_of(codepoints[scan_index][0]) == "Regional_Indicator":
                    regional_indicator_count += 1
                else:
                    break
            if regional_indicator_count % 2 == 1:
                continue
        boundaries.append(boundary_byte_position)
    boundaries.append(total_bytes)
    return sorted(set(boundaries))
def get_sentence_break_properties(version: str = UNICODE_VERSION) -> Dict[int, str]:
    cache_path = _download_break_property_file("SentenceBreakProperty.txt", version)
    return _parse_break_property_file(cache_path)
def get_sentence_break_test_cases(version: str = UNICODE_VERSION) -> List[tuple]:
    cache_path = _download_break_test_file("SentenceBreakTest.txt", version)
    return _parse_break_test_file(cache_path)
def baseline_sentence_boundaries(text: str, properties: Dict[int, str] = None) -> List[int]:
    if properties is None:
        properties = get_sentence_break_properties()
    def property_of(codepoint: int) -> str:
        return properties.get(codepoint, "Other")
    codepoints = []
    byte_offset = 0
    for char in text:
        codepoint = ord(char)
        codepoint_bytes = char.encode("utf-8")
        codepoints.append((codepoint, byte_offset, len(codepoint_bytes)))
        byte_offset += len(codepoint_bytes)
    if not codepoints:
        return [0]
    total_bytes = byte_offset
    property_list = [property_of(codepoint) for codepoint, _, _ in codepoints]
    def is_ignorable(property_value: str) -> bool:
        return property_value in ("Extend", "Format")
    def effective_before(relative_index: int) -> str:
        scan_index = relative_index
        while scan_index >= 0 and is_ignorable(property_list[scan_index]):
            scan_index -= 1
        return property_list[scan_index] if scan_index >= 0 else "Other"
    def is_paragraph_separator(property_value: str) -> bool:
        return property_value in ("Sep", "CR", "LF")
    boundaries = [0]
    codepoint_count = len(codepoints)
    for index in range(1, codepoint_count):
        boundary_byte_position = codepoints[index][1]
        previous_property = property_list[index - 1]
        current_property = property_list[index]
        if previous_property == "CR" and current_property == "LF":
            continue
        if is_paragraph_separator(previous_property):
            boundaries.append(boundary_byte_position)
            continue
        if current_property in ("Extend", "Format"):
            continue
        effective_previous = effective_before(index - 1)
        if effective_previous == "ATerm" and current_property == "Numeric":
            continue
        if effective_previous == "ATerm" and current_property == "Upper":
            scan_index = index - 2
            while scan_index >= 0 and is_ignorable(property_list[scan_index]):
                scan_index -= 1
            if scan_index >= 0 and property_list[scan_index] in ("Upper", "Lower"):
                continue
        scan_index = index - 1
        while scan_index >= 0 and is_ignorable(property_list[scan_index]):
            scan_index -= 1
        while scan_index >= 0 and (property_list[scan_index] == "Sp" or is_ignorable(property_list[scan_index])):
            scan_index -= 1
        while scan_index >= 0 and (property_list[scan_index] == "Close" or is_ignorable(property_list[scan_index])):
            scan_index -= 1
        terminator_property = property_list[scan_index] if scan_index >= 0 else "Other"
        if terminator_property in ("ATerm", "STerm"):
            if current_property in ("SContinue", "STerm", "ATerm"):
                continue
            if terminator_property == "ATerm":
                lookahead_index = index
                lower_ahead = False
                while lookahead_index < codepoint_count:
                    scanned_property = property_list[lookahead_index]
                    if scanned_property == "Lower":
                        lower_ahead = True
                        break
                    if scanned_property in ("OLetter", "Upper", "Sep", "CR", "LF", "STerm", "ATerm"):
                        break
                    lookahead_index += 1
                if lower_ahead:
                    continue
            if current_property in ("Close", "Sp", "Sep", "CR", "LF"):
                continue
            boundaries.append(boundary_byte_position)
            continue
        continue
    boundaries.append(total_bytes)
    return sorted(set(boundaries))
def get_line_break_properties(version: str = UNICODE_VERSION) -> Dict[int, str]:
    cache_path = _download_break_property_file("LineBreak.txt", version)
    return _parse_break_property_file(cache_path)
def get_line_break_test_cases(version: str = UNICODE_VERSION) -> List[tuple]:
    cache_path = _download_break_test_file("LineBreakTest.txt", version)
    return _parse_break_test_file(cache_path)
def baseline_line_boundaries(text: str, properties: Dict[int, str] = None) -> List[tuple]:
    if properties is None:
        properties = get_line_break_properties()
    def property_of(codepoint: int) -> str:
        return properties.get(codepoint, "AL")
    codepoints = []
    byte_offset = 0
    for char in text:
        codepoint = ord(char)
        codepoint_bytes = char.encode("utf-8")
        codepoints.append((codepoint, byte_offset, len(codepoint_bytes)))
        byte_offset += len(codepoint_bytes)
    if not codepoints:
        return [(0, False)]
    total_bytes = byte_offset
    boundaries = []
    seen = set()
    def add(offset: int, mandatory: bool):
        if offset not in seen:
            seen.add(offset)
            boundaries.append((offset, mandatory))
    add(0, False)
    codepoint_count = len(codepoints)
    for index in range(1, codepoint_count):
        boundary_byte_position = codepoints[index][1]
        previous_property = property_of(codepoints[index - 1][0])
        current_property = property_of(codepoints[index][0])
        if previous_property == "CR" and current_property == "LF":
            continue
        if previous_property in ("BK", "CR", "LF", "NL"):
            add(boundary_byte_position, True)
            continue
    add(total_bytes, False)
    return sorted(boundaries)
def _download_ucd_text(filename: str, subdir: str, version: str = UNICODE_VERSION) -> str:
    cache_path = os.path.join(tempfile.gettempdir(), f"{filename[:-4]}-{version}.txt")
    if not os.path.exists(cache_path):
        url = f"https://www.unicode.org/Public/{version}/{subdir}/{filename}"
        print(f"Downloading Unicode {version} {filename} from {url}...")
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                _write_cache_atomically(cache_path, response.read())
            print(f"Cached to {cache_path}")
        except Exception as error:
            raise UnicodeDataDownloadError(f"Could not download {filename} from {url}: {error}")
    return cache_path
def get_normalization_test_cases(version: str = UNICODE_VERSION) -> List[tuple]:
    cache_path = _download_ucd_text("NormalizationTest.txt", "ucd", version)
    def field_to_string(field: str) -> str:
        return "".join(chr(int(codepoint, 16)) for codepoint in field.split())
    cases = []
    with open(cache_path, "r", encoding="utf-8") as data_file:
        for line in data_file:
            line = line.split("#")[0].strip()
            if not line or line.startswith("@"):
                continue
            fields = [field.strip() for field in line.split(";")]
            if len(fields) < 5:
                continue
            try:
                cases.append(tuple(field_to_string(fields[index]) for index in range(5)))
            except ValueError:
                continue
    return cases
def get_emoji_properties(version: str = UNICODE_VERSION) -> Dict[int, set]:
    cache_path = _download_ucd_text("emoji-data.txt", "ucd/emoji", version)
    properties: Dict[int, set] = {}
    with open(cache_path, "r", encoding="utf-8") as data_file:
        for line in data_file:
            line = line.split("#")[0].strip()
            if not line:
                continue
            parts = line.split(";")
            if len(parts) < 2:
                continue
            try:
                codepoint_range = parts[0].strip()
                property_name = parts[1].strip()
                if ".." in codepoint_range:
                    start_codepoint, end_codepoint = (int(bound, 16) for bound in codepoint_range.split(".."))
                else:
                    start_codepoint = end_codepoint = int(codepoint_range, 16)
                for codepoint in range(start_codepoint, end_codepoint + 1):
                    properties.setdefault(codepoint, set()).add(property_name)
            except (ValueError, IndexError):
                continue
    return properties
def get_extended_pictographic(version: str = UNICODE_VERSION) -> set:
    return {codepoint for codepoint, names in get_emoji_properties(version).items() if "Extended_Pictographic" in names}
def get_combining_classes(version: str = UNICODE_VERSION) -> Dict[int, int]:
    root = get_unicode_xml_data(version)
    namespace = "{http://www.unicode.org/ns/2003/ucd/1.0}"
    combining_classes: Dict[int, int] = {}
    for char in root.iter(namespace + "char"):
        codepoint = char.get("cp")
        raw_combining_class = char.get("ccc")
        if codepoint is None or raw_combining_class is None:
            continue
        value = int(raw_combining_class)
        if value:
            combining_classes[int(codepoint, 16)] = value
    return combining_classes
def get_decomposition_mappings(version: str = UNICODE_VERSION) -> Dict[int, tuple]:
    root = get_unicode_xml_data(version)
    namespace = "{http://www.unicode.org/ns/2003/ucd/1.0}"
    mappings: Dict[int, tuple] = {}
    for char in root.iter(namespace + "char"):
        codepoint = char.get("cp")
        decomposition_mapping = char.get("dm")
        if codepoint is None or not decomposition_mapping or decomposition_mapping == "#":
            continue
        decomposition_type = char.get("dt")
        kind = "canonical" if decomposition_type in (None, "can") else "compatibility"
        targets = [int(part, 16) for part in decomposition_mapping.split()]
        mappings[int(codepoint, 16)] = (kind, targets)
    return mappings
def representatives_by_class(
    properties: Dict[int, str], count: int = 3, bmp_only: bool = False
) -> Dict[str, List[int]]:
    representatives: Dict[str, List[int]] = {}
    for codepoint in sorted(properties):
        if 0xD800 <= codepoint <= 0xDFFF:
            continue
        if bmp_only and codepoint > 0xFFFF:
            continue
        bucket = representatives.setdefault(properties[codepoint], [])
        if len(bucket) < count:
            bucket.append(codepoint)
    return representatives
_random_seed_for_run = int.from_bytes(os.urandom(4), "little")
SEED_VALUES = [
    42,
    0,
    1,
    314159,
    _random_seed_for_run,
]
_env_seed = os.environ.get("SZ_TESTS_SEED")
if _env_seed:
    try:
        _parsed_seed = int(_env_seed)
        SEED_VALUES = [_parsed_seed]
        print(f"SZ_TESTS_SEED={_parsed_seed} (from environment, overriding default seeds)")
    except ValueError:
        pass
def _read_iterations_multiplier() -> float:
    raw = os.environ.get("SZ_TESTS_MULTIPLIER")
    if raw:
        try:
            value = float(raw)
            if value > 0.0:
                return value
        except ValueError:
            pass
    return 1.0
ITERATIONS_MULTIPLIER = _read_iterations_multiplier()
def scale_iterations(baseline: int) -> int:
    return max(1, int(baseline * ITERATIONS_MULTIPLIER))
def seed_random_generators(seed_value: Optional[int] = None):
    if seed_value is None:
        return
    seed(seed_value)
    if numpy_available:
        try:
            np.random.seed(seed_value)
        except (ImportError, AttributeError, Exception):
            pass
def get_random_string(
    length: Optional[int] = None, variability: Optional[int] = None, alphabet: Optional[str] = None
) -> str:
    if length is None:
        length = randint(3, 300)
    if alphabet is None:
        alphabet = ascii_lowercase
    if variability is None:
        variability = len(alphabet)
    return "".join(choice(alphabet[:variability]) for _ in range(length))
def is_equal_strings(native_strings, big_strings):
    for native_slice, big_slice in zip(native_strings, big_strings):
        assert native_slice == big_slice, f"Mismatch between `{native_slice}` and `{str(big_slice)}`"
DEGENERATE_HAYSTACKS = ["", "a", "hello world"]
DEGENERATE_BOUNDS = [-16, -6, -5, -1, 0, 1, 2, 5, 6, 10, 11, 12, 16, 99]
UNCASED_DEGENERATE_BOUNDS = [0, 1, 3, 4, 5, 6, 7, 8, 11, 99]
def capability_sweep():
    import stringzilla as sz
    simd_backends = [capability for capability in sz.__capabilities__ if capability != "serial"]
    sweep = [("serial",)] + [("serial", backend) for backend in simd_backends]
    if len(simd_backends) > 1:
        sweep.append(("any",))
    return sweep
@contextlib.contextmanager
def forced_capabilities(*names):
    import stringzilla as sz
    try:
        sz.reset_capabilities(list(names))
        yield tuple(sz.__capabilities__)
    finally:
        sz.reset_capabilities(["any"])
def run_across_backends(operation) -> Dict[tuple, object]:
    results = {}
    for config in capability_sweep():
        with forced_capabilities(*config):
            results[config] = operation()
    return results
def assert_backends_agree(results, *, oracle=None, format_inputs=None):
    context = f" [{format_inputs()}]" if format_inputs is not None else ""
    baseline_config = next(iter(results))
    baseline = results[baseline_config]
    diverged = {config: value for config, value in results.items() if value != baseline}
    assert not diverged, f"backend divergence{context}: baseline {baseline_config}={baseline!r}; diverged={diverged!r}"
    if oracle is not None:
        assert baseline == oracle, f"kernels agree but disagree with the oracle{context}: {baseline!r} != {oracle!r}"
VECTOR_WIDTH_LENGTHS = (
    [0, 1, 2, 3]
    + [7, 8, 9]
    + [15, 16, 17]
    + [31, 32, 33, 47, 48]
    + [63, 64, 65, 95, 96]
    + [127, 128, 129, 255, 256, 257]
    + [1024, 4096]
)
def boundary_strings(alphabet: str = "ab") -> List[str]:
    repeated = alphabet * (max(VECTOR_WIDTH_LENGTHS) // len(alphabet) + 1)
    return [repeated[:length] for length in VECTOR_WIDTH_LENGTHS]
def unaligned_views(text, offsets=(0, 1, 3, 7, 15)):
    import stringzilla as sz
    parent = sz.Str(text)
    for offset in offsets:
        if offset <= len(parent):
            yield offset, parent[offset:]
def malformed_utf8_corpus() -> List[bytes]:
    return [
        b"\x80",
        b"\xc3",
        b"\xe0\xa0",
        b"\xf0\x9f\x98",
        b"\xc0\x80",
        b"\xed\xa0\x80",
        b"\xf5\x80\x80\x80",
        b"a\x00b",
        b"\xff\xfe",
        b"valid\xfftext",
    ]
def vector_width_bracketing_strings() -> List[str]:
    multibyte_codepoints = ["é", "中", "\U0001f600"]
    lane_widths = [16, 32, 64]
    strings = []
    for lane_width in lane_widths:
        for byte_offset in (-2, -1, 0, 1, 2):
            for codepoint in multibyte_codepoints:
                pad_length = max(0, lane_width + byte_offset - len(codepoint.encode("utf-8")))
                strings.append("a" * pad_length + codepoint + "tail")
    return strings
_BOUNDARY_TILED_STRINGS = boundary_strings()
_BOUNDARY_SAME_CHAR_STRINGS = boundary_strings(alphabet="a")
_BOUNDARY_STRINGS_BY_LENGTH = {
    length: (tiled, same_char)
    for length, tiled, same_char in zip(VECTOR_WIDTH_LENGTHS, _BOUNDARY_TILED_STRINGS, _BOUNDARY_SAME_CHAR_STRINGS)
}
def differential_bodies(length, seed_value):
    seed_random_generators(seed_value)
    tiled_body, same_char_body = _BOUNDARY_STRINGS_BY_LENGTH[length]
    return get_random_string(length=length), tiled_body, same_char_body
