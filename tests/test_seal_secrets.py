"""scripts/seal_secrets.py: the sealed format the controller decrypts.

The controller's decryption is reimplemented here from its documented
format (a 2-byte length, the RSA-OAEP encrypted session key, then the
AES-GCM sealed value under a zero nonce). The bootstrap in CI proves the
real controller opens the committed secrets; these tests pin the format and
the scope binding without a cluster.
"""

import base64
import importlib.util
import struct
import sys
from pathlib import Path

import pytest
import yaml
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "seal_secrets", ROOT / "scripts" / "seal_secrets.py"
)
seal = importlib.util.module_from_spec(_SPEC)
sys.modules["seal_secrets"] = seal
_SPEC.loader.exec_module(seal)


@pytest.fixture(scope="module")
def key():
    # 2048 bits keeps the test fast; the format does not depend on the size.
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


def unseal(key, ciphertext: str, label: str) -> str:
    raw = base64.b64decode(ciphertext)
    (length,) = struct.unpack(">H", raw[:2])
    session_key = key.decrypt(
        raw[2 : 2 + length],
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=label.encode(),
        ),
    )
    return AESGCM(session_key).decrypt(bytes(12), raw[2 + length :], None).decode()


def test_a_value_opens_with_its_own_namespace_and_name(key):
    sealed = seal._seal(
        key.public_key(), "s3cret", "vortex-staging/vortex-inference-jobs"
    )

    assert unseal(key, sealed, "vortex-staging/vortex-inference-jobs") == "s3cret"


@pytest.mark.parametrize(
    "label",
    ["vortex-prod/vortex-inference-jobs", "vortex-staging/vortex-inference-api-key"],
)
def test_a_value_does_not_open_in_another_namespace_or_secret(key, label):
    sealed = seal._seal(
        key.public_key(), "s3cret", "vortex-staging/vortex-inference-jobs"
    )

    with pytest.raises(ValueError):
        unseal(key, sealed, label)


def test_sealing_the_same_value_twice_gives_different_ciphertexts(key):
    # A fresh session key per value: equal secrets are not recognisable as
    # equal in the repository.
    first = seal._seal(key.public_key(), "same", "ns/name")
    second = seal._seal(key.public_key(), "same", "ns/name")

    assert first != second


def test_a_tampered_value_is_rejected(key):
    raw = bytearray(base64.b64decode(seal._seal(key.public_key(), "s3cret", "ns/name")))
    raw[-1] ^= 0x01

    with pytest.raises(InvalidTag):
        unseal(key, base64.b64encode(bytes(raw)).decode(), "ns/name")


def test_sealed_secret_document_targets_its_secret(key):
    document = seal._sealed_secret(
        key.public_key(), "vortex-inference-api-key", "vortex-prod", {"api-key": "k"}
    )

    assert document["kind"] == "SealedSecret"
    assert document["spec"]["template"]["metadata"] == {
        "name": "vortex-inference-api-key",
        "namespace": "vortex-prod",
    }
    assert (
        unseal(
            key,
            document["spec"]["encryptedData"]["api-key"],
            "vortex-prod/vortex-inference-api-key",
        )
        == "k"
    )


def test_committed_environments_are_sealed_to_the_committed_certificate():
    # Every SealedSecret in the overlays names its own environment's
    # namespace; a copy-paste between overlays would not decrypt.
    for env, namespace in seal.ENVIRONMENTS.items():
        path = ROOT / "k8s" / "overlays" / env / "sealed-secrets.yaml"
        documents = list(yaml.safe_load_all(path.read_text()))
        assert {d["metadata"]["name"] for d in documents} == {
            "vortex-inference-api-key",
            "vortex-inference-jobs",
        }
        for document in documents:
            assert document["metadata"]["namespace"] == namespace
            assert document["spec"]["template"]["metadata"]["namespace"] == namespace
