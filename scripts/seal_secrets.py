#!/usr/bin/env python3
"""Generate the sealing key and the sealed credentials of the GitOps environments.

Two subcommands:

  generate-key   Creates the RSA key pair the Sealed Secrets controller decrypts
                 with. The certificate goes to gitops/sealed-secrets/ and is
                 committed; the private key goes to .secrets/ (gitignored) and
                 to the SEALED_SECRETS_KEY repository secret, from which every
                 cluster bootstrap installs it. Run once; rotating it means
                 resealing every environment.

  environment    Generates fresh credentials for one environment (API key,
                 RabbitMQ and Redis passwords and the URLs built from them)
                 and writes them as SealedSecrets encrypted to the committed
                 certificate. Nothing in plain text touches the disk.

Sealing follows the controller's own format, which `kubeseal --scope strict`
also produces: per value, a random AES-256 session key encrypted with
RSA-OAEP (SHA-256) using "<namespace>/<name>" as the OAEP label, then the
value sealed with AES-GCM under that session key. The label binds each
ciphertext to its Secret's name and namespace, so a value copied into
another Secret or environment fails to decrypt. kubeseal produces
equivalent output, and either can be used:

  kubeseal --cert gitops/sealed-secrets/sealing-key.crt --scope strict ...

Usage:
  python scripts/seal_secrets.py generate-key
  python scripts/seal_secrets.py environment staging
"""

from __future__ import annotations

import argparse
import base64
import datetime
import os
import secrets
import struct
import sys
from pathlib import Path

import yaml
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.x509.oid import NameOID

ROOT = Path(__file__).resolve().parent.parent
CERT_PATH = ROOT / "gitops" / "sealed-secrets" / "sealing-key.crt"
KEY_PATH = ROOT / ".secrets" / "sealed-secrets.key"

# Namespace of each environment, as set in its overlay.
ENVIRONMENTS = {"staging": "vortex-staging", "prod": "vortex-prod"}


def generate_key() -> None:
    if KEY_PATH.exists():
        sys.exit(f"{KEY_PATH} already exists; delete it first to rotate the key")
    key = rsa.generate_private_key(public_exponent=65537, key_size=4096)
    name = x509.Name(
        [
            x509.NameAttribute(NameOID.COMMON_NAME, "sealed-secret"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "sealed-secret"),
        ]
    )
    now = datetime.datetime.now(datetime.UTC)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + datetime.timedelta(days=3650))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )
    CERT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CERT_PATH.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    KEY_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(KEY_PATH, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(
            key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            )
        )
    print(f"certificate: {CERT_PATH.relative_to(ROOT)}")
    print(f"private key: {KEY_PATH.relative_to(ROOT)} (keep it out of git)")


def _seal(public_key: rsa.RSAPublicKey, value: str, label: str) -> str:
    session_key = AESGCM.generate_key(bit_length=256)
    encrypted_key = public_key.encrypt(
        session_key,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=label.encode(),
        ),
    )
    # A zero nonce is safe here, and it is what the controller expects: each
    # session key is random and encrypts exactly one value.
    sealed = AESGCM(session_key).encrypt(bytes(12), value.encode(), None)
    return base64.b64encode(
        struct.pack(">H", len(encrypted_key)) + encrypted_key + sealed
    ).decode()


def _sealed_secret(
    public_key: rsa.RSAPublicKey, name: str, namespace: str, data: dict
) -> dict:
    label = f"{namespace}/{name}"
    return {
        "apiVersion": "bitnami.com/v1alpha1",
        "kind": "SealedSecret",
        "metadata": {"name": name, "namespace": namespace},
        "spec": {
            "encryptedData": {
                k: _seal(public_key, v, label) for k, v in sorted(data.items())
            },
            "template": {
                "metadata": {"name": name, "namespace": namespace},
                "type": "Opaque",
            },
        },
    }


def environment(env: str) -> None:
    namespace = ENVIRONMENTS[env]
    cert = x509.load_pem_x509_certificate(CERT_PATH.read_bytes())
    public_key = cert.public_key()
    rabbit = secrets.token_hex(24)
    redis = secrets.token_hex(24)
    # Same keys and URL shapes as `make api-key` and `make jobs-secrets`
    # create for the local profile.
    documents = [
        _sealed_secret(
            public_key,
            "vortex-inference-api-key",
            namespace,
            {"api-key": secrets.token_urlsafe(32)},
        ),
        _sealed_secret(
            public_key,
            "vortex-inference-jobs",
            namespace,
            {
                "rabbitmq-username": "vortex",
                "rabbitmq-password": rabbit,
                "rabbitmq-url": f"amqp://vortex:{rabbit}@vortex-rabbitmq:5672/",
                "rabbitmq-management-url": f"http://vortex:{rabbit}@vortex-rabbitmq.{namespace}.svc:15672",
                "redis-password": redis,
                "redis-url": f"redis://:{redis}@vortex-redis:6379/0",
                "redis-auth.conf": f"requirepass {redis}",
            },
        ),
    ]
    out = ROOT / "k8s" / "overlays" / env / "sealed-secrets.yaml"
    header = (
        f"# Credentials of the {env} environment, encrypted to the certificate in\n"
        "# gitops/sealed-secrets/. Only the Sealed Secrets controller holding the\n"
        "# matching private key can decrypt them, and each value only into the\n"
        "# Secret and namespace named here. Regenerate with:\n"
        f"#   python scripts/seal_secrets.py environment {env}\n"
    )
    out.write_text(
        header + yaml.safe_dump_all(documents, sort_keys=False, explicit_start=True)
    )
    print(f"wrote {out.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("generate-key")
    env_parser = sub.add_parser("environment")
    env_parser.add_argument("name", choices=sorted(ENVIRONMENTS))
    args = parser.parse_args()
    if args.command == "generate-key":
        generate_key()
    else:
        environment(args.name)


if __name__ == "__main__":
    main()
