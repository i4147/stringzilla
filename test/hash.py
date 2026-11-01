import hashlib
import hmac
from array import array
import pytest
import stringzilla as sz
from stringzilla import Str
try:
    import numpy as np
except:
    pass
from test.sz_helpers import (
    SEED_VALUES,
    seed_random_generators,
    get_random_string,
    run_across_backends,
    assert_backends_agree,
    forced_capabilities,
    capability_sweep,
    VECTOR_WIDTH_LENGTHS,
    differential_bodies,
)
@pytest.mark.parametrize("body", ["", "hello", "world", "abcdefg", "a" * 32])
@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_hash_basic_equivalence(body: str, seed_value: int):
    hash_seeded = sz.hash(body, seed=seed_value)
    hash_member = sz.Str(body).hash(seed=seed_value)
    assert hash_seeded == hash_member
@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_hasher_incremental_vs_one_shot(seed_value: int):
    data_full = b"hello world"
    data_prefix = b"hello "
    data_suffix = b"world"
    hasher = sz.Hasher(seed=seed_value)
    hasher.update(data_prefix)
    hasher.update(data_suffix)
    streamed_hash = hasher.digest()
    expected_hash = sz.hash(data_full, seed=seed_value)
    assert isinstance(streamed_hash, int)
    assert streamed_hash == expected_hash
@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_hasher_reset_and_hexdigest(seed_value: int):
    data = b"some test payload"
    hasher = sz.Hasher(seed=seed_value)
    hasher.update(data)
    streamed_hash = hasher.digest()
    streamed_hex = hasher.hexdigest()
    assert isinstance(streamed_hex, str) and len(streamed_hex) == 16 and streamed_hex == format(streamed_hash, "016x")
    hasher.reset()
    hasher.update(data)
    re_streamed_hash = hasher.digest()
    re_streamed_hex = hasher.hexdigest()
    assert streamed_hash == re_streamed_hash
    assert streamed_hex == re_streamed_hex
@pytest.mark.parametrize("body", ["", "x", "hello", "abcdefg", "a" * 17, "a" * 32, "a" * 64, "a" * 100])
def test_hash_multiseed_equivalence(body: str):
    from array import array
    seeds_list = [0, 1, 42, 314159, 7, 8, 9, 10, 11]
    seeds = array("Q", seeds_list)
    expected = tuple(sz.hash(body, s) for s in seeds_list)
    assert sz.hash_multiseed(body, seeds) == expected
    assert sz.Str(body).hash_multiseed(seeds) == expected
    out = array("Q", [0] * len(seeds))
    assert sz.hash_multiseed(body, seeds, out=out) is None
    assert tuple(out) == expected
def test_hash_multiseed_errors():
    from array import array
    seeds = array("Q", [1, 2, 3])
    with pytest.raises(TypeError):
        sz.hash_multiseed("x", [1, 2, 3])
    with pytest.raises(TypeError):
        sz.hash_multiseed("x", array("I", [1, 2, 3]))
    with pytest.raises(ValueError):
        sz.hash_multiseed("x", seeds, out=array("Q", [0]))
@pytest.mark.parametrize("length", [0, 1, 3, 7, 15, 31, 63, 64, 65, 127, 128, 129, 255, 256, 1000, 4096, 10000])
@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_sha256(length: int, seed_value: int):
    seed_random_generators(seed_value)
    text = get_random_string(length=length)
    expected = hashlib.sha256(text.encode()).digest()
    assert sz.sha256(text) == expected
    assert Str(text).sha256() == expected
    assert sz.sha256(text.encode()) == expected
    h = sz.Sha256()
    h.update(text)
    assert h.digest() == expected
    assert h.hexdigest() == expected.hex()
    if length > 0:
        h = sz.Sha256()
        chunk_size = max(1, length // 3)
        for i in range(0, length, chunk_size):
            h.update(text[i : i + chunk_size])
        assert h.digest() == expected
        assert h.hexdigest() == expected.hex()
    h.reset().update(text)
    assert h.digest() == expected
    mid = length // 2
    h1 = sz.Sha256().update(text[:mid])
    h2 = h1.copy()
    h1.update(text[mid:])
    h2.update(text[mid:])
    assert h1.digest() == h2.digest() == expected
@pytest.mark.parametrize("lanes_count", [0, 1, 7, 8, 9, 15, 16, 17, 33])
@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_sha256_lanes(lanes_count: int, seed_value: int):
    seed_random_generators(seed_value)
    lengths = [(index * 137) % 4096 for index in range(lanes_count)]
    messages = [get_random_string(length=length).encode() for length in lengths]
    expected = [hashlib.sha256(message).digest() for message in messages]
    lanes = sz.Sha256s(lanes_count)
    assert len(lanes) == lanes_count
    offsets = [0] * lanes_count
    for call_index in range(3):
        chunks = []
        for lane_index in range(lanes_count):
            remaining = len(messages[lane_index]) - offsets[lane_index]
            take = remaining if call_index == 2 else remaining // 3
            chunks.append(messages[lane_index][offsets[lane_index] : offsets[lane_index] + take])
            offsets[lane_index] += take
        lanes.update(chunks)
    assert lanes.digest() == expected
    assert lanes.digest() == expected
    assert [lane.digest() for lane in lanes] == expected
    assert lanes.hexdigest() == [digest.hex() for digest in expected]
    clone = lanes.copy()
    clone.update([b"!"] * lanes_count)
    assert clone.digest() == [hashlib.sha256(message + b"!").digest() for message in messages]
    assert lanes.digest() == expected
    lanes.reset()
    assert lanes.digest() == [hashlib.sha256(b"").digest()] * lanes_count
@pytest.mark.parametrize("lanes_count", [0, 1, 8, 17])
@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_sha256_lanes_strs(lanes_count: int, seed_value: int):
    seed_random_generators(seed_value)
    messages = [get_random_string(length=(index * 137) % 4096) for index in range(lanes_count)]
    for texts in (sz.Strs(messages), sz.Strs(messages).shuffled()):
        expected = [hashlib.sha256(str(text).encode()).digest() for text in texts]
        assert sz.Sha256s(lanes_count).update(texts).digest() == expected
@pytest.mark.skipif("np" not in globals(), reason="NumPy is not installed")
@pytest.mark.parametrize("lanes_count", [0, 1, 8, 17])
def test_sha256_digests_into_matrix(lanes_count: int):
    messages = [f"message-{index}".encode() * (index + 1) for index in range(lanes_count)]
    expected = [hashlib.sha256(message).digest() for message in messages]
    lanes = sz.Sha256s(lanes_count)
    digests = np.empty((lanes_count, sz.Sha256.digest_length), np.uint8)
    assert lanes.update(messages).digest(out=digests) is digests
    assert [bytes(row) for row in digests] == expected
def test_sha256_lanes_recycling():
    lanes = sz.Sha256s(4)
    lanes.update([b"aa", b"bb", b"cc", b"dd"])
    lanes[1] = sz.Sha256()
    lanes.update([b"11", b"22", b"33", b"44"])
    digests = lanes.digest()
    assert digests[1] == hashlib.sha256(b"22").digest(), "the recycled lane kept its old data"
    assert digests[0] == hashlib.sha256(b"aa11").digest(), "a neighbouring lane was disturbed"
    assert digests[3] == hashlib.sha256(b"dd44").digest(), "a neighbouring lane was disturbed"
def test_sha256_lanes_errors():
    with pytest.raises(ValueError):
        sz.Sha256s(2).update([b"only-one-chunk"])
    with pytest.raises(ValueError):
        sz.Sha256s(2).update(sz.Strs(["only-one-chunk"]))
    with pytest.raises(ValueError):
        sz.Sha256s(2).digest(out=bytearray(63))
    with pytest.raises(TypeError):
        sz.Sha256s(2).digest(out=array("Q", [0] * 8))
    with pytest.raises(TypeError):
        sz.Sha256s(2)[0] = b"not-a-hasher"
@pytest.mark.parametrize("length", VECTOR_WIDTH_LENGTHS + [4096, 100000])
@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_bytesum_random(length: int, seed_value: int):
    def sum_bytes(body: str) -> int:
        return sum([ord(c) for c in body])
    seed_random_generators(seed_value)
    body = get_random_string(length=length)
    assert sum_bytes(body) == sz.bytesum(body)
@pytest.mark.parametrize("key_length", [0, 1, 16, 32, 64, 65, 128])
@pytest.mark.parametrize("message_length", [0, 1, 63, 64, 65, 127, 128, 1000])
@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_hmac_sha256(key_length: int, message_length: int, seed_value: int):
    seed_random_generators(seed_value)
    key = get_random_string(length=key_length).encode()
    message = get_random_string(length=message_length).encode()
    expected = hmac.new(key, message, hashlib.sha256).digest()
    result = sz.hmac_sha256(key, message)
    assert result == expected
    key_str = key.decode("latin1")
    message_str = message.decode("latin1")
    result_str = sz.hmac_sha256(key_str, message_str)
    assert result_str == expected
@pytest.mark.parametrize("messages_count", [0, 1, 8, 17, 129])
@pytest.mark.parametrize("key_length", [16, 65])
@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_hmac_sha256_many(messages_count: int, key_length: int, seed_value: int):
    seed_random_generators(seed_value)
    key = get_random_string(length=key_length).encode()
    messages = [get_random_string(length=(index * 37) % 512) for index in range(messages_count)]
    expected = [hmac.new(key, message.encode(), hashlib.sha256).digest() for message in messages]
    assert sz.hmac_sha256(key, [message.encode() for message in messages]) == expected
    assert sz.hmac_sha256(key, sz.Strs(messages)) == expected
@pytest.mark.skipif("np" not in globals(), reason="NumPy is not installed")
def test_hmac_sha256_into_matrix():
    key = b"secret"
    messages = [f"message-{index}".encode() for index in range(9)]
    expected = [hmac.new(key, message, hashlib.sha256).digest() for message in messages]
    tags = np.empty((len(messages), sz.Sha256.digest_length), np.uint8)
    assert sz.hmac_sha256(key, messages, out=tags) is tags
    assert [bytes(row) for row in tags] == expected
    one = bytearray(sz.Sha256.digest_length)
    sz.hmac_sha256(key, messages[0], out=one)
    assert bytes(one) == expected[0]
def test_hmac_sha256_kwargs():
    key = b"secret"
    message = b"Hello, world!"
    expected = hmac.new(key, message, hashlib.sha256).digest()
    result_positional = sz.hmac_sha256(key, message)
    assert result_positional == expected
    result_kwargs = sz.hmac_sha256(key=key, message=message)
    assert result_kwargs == expected
    result_mixed = sz.hmac_sha256(key, message=message)
    assert result_mixed == expected
    result_reversed = sz.hmac_sha256(message=message, key=key)
    assert result_reversed == expected
    with pytest.raises(TypeError, match="missing required arguments"):
        sz.hmac_sha256(key=key)
    with pytest.raises(TypeError, match="key specified twice"):
        sz.hmac_sha256(key, key=key)
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        sz.hmac_sha256(key=key, unknown=b"test")
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        sz.hmac_sha256(key=key, message=message, unknown=b"test")
@pytest.mark.parametrize("length", VECTOR_WIDTH_LENGTHS)
@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_unit_backend_differential_bytesum(length, seed_value):
    for body in differential_bodies(length, seed_value):
        oracle = sum(body.encode())
        assert_backends_agree(run_across_backends(lambda body=body: sz.bytesum(body)), oracle=oracle)
@pytest.mark.parametrize("length", VECTOR_WIDTH_LENGTHS)
@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_unit_backend_differential_hash(length, seed_value):
    for body in differential_bodies(length, seed_value):
        assert_backends_agree(run_across_backends(lambda body=body: sz.hash(body, seed=seed_value)))
        assert_backends_agree(run_across_backends(lambda body=body: Str(body).hash(seed=seed_value)))
@pytest.mark.parametrize("length", VECTOR_WIDTH_LENGTHS)
def test_unit_backend_differential_hash_multiseed(length):
    from array import array
    seeds_list = [0, 1, 42, 314159, 7, 8, 9, 10, 11]
    seeds = array("Q", seeds_list)
    for body in differential_bodies(length, seed_value=length):
        assert_backends_agree(run_across_backends(lambda body=body: sz.hash_multiseed(body, seeds)))
        def filled(body=body):
            out = array("Q", [0] * len(seeds_list))
            sz.hash_multiseed(body, seeds, out=out)
            return tuple(out)
        assert_backends_agree(run_across_backends(filled))
@pytest.mark.parametrize("length", VECTOR_WIDTH_LENGTHS)
@pytest.mark.parametrize("nonce", [0, 1, 42, 314159])
def test_unit_backend_differential_fill_random(length, nonce):
    def fill():
        buffer = bytearray(length)
        sz.fill_random(buffer, nonce=nonce)
        return bytes(buffer)
    assert_backends_agree(run_across_backends(fill))
@pytest.mark.parametrize("length", VECTOR_WIDTH_LENGTHS)
@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_unit_backend_differential_sha256(length, seed_value):
    seed_random_generators(seed_value)
    text = get_random_string(length=length)
    expected_digest = hashlib.sha256(text.encode()).digest()
    expected_hex = expected_digest.hex()
    for config in capability_sweep():
        with forced_capabilities(*config):
            assert sz.sha256(text) == expected_digest
            assert sz.sha256(text.encode()) == expected_digest
            assert Str(text).sha256() == expected_digest
            incremental = sz.Sha256()
            incremental.update(text)
            assert incremental.digest() == expected_digest
            assert incremental.hexdigest() == expected_hex
@pytest.mark.parametrize("length", [0, 1, 31, 32, 33, 64, 65, 128, 257, 1024])
@pytest.mark.parametrize("seed_value", SEED_VALUES)
def test_unit_backend_differential_sha256_chunked(length, seed_value):
    seed_random_generators(seed_value)
    text = get_random_string(length=length)
    expected_digest = hashlib.sha256(text.encode()).digest()
    for config in capability_sweep():
        with forced_capabilities(*config):
            chunk_size = max(1, length // 3)
            chunked = sz.Sha256()
            for chunk_start in range(0, length, chunk_size):
                chunked.update(text[chunk_start : chunk_start + chunk_size])
            assert chunked.digest() == expected_digest
            chunked.reset().update(text)
            assert chunked.digest() == expected_digest
            midpoint = length // 2
            first_half = sz.Sha256().update(text[:midpoint])
            second_half = first_half.copy()
            first_half.update(text[midpoint:])
            second_half.update(text[midpoint:])
            assert first_half.digest() == second_half.digest() == expected_digest
