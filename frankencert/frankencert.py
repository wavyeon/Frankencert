from __future__ import annotations

import argparse
import collections
import functools
import io
import os
import random
import string
import sys
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.backends.openssl.backend import Backend

# This is so much pain… :(
from cryptography.hazmat.backends.openssl.encode_asn1 import (
    _encode_asn1_int_gc,
    _encode_name_gc,
)
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, ed448, ed25519, rsa
from cryptography.hazmat.primitives.asymmetric.types import (
    PRIVATE_KEY_TYPES,
    PUBLIC_KEY_TYPES,
)
from cryptography.x509.extensions import Extension, ExtensionType
from cryptography.x509.name import Name
from cryptography.x509.oid import NameOID

log = functools.partial(print, file=sys.stderr, flush=True)
FRANKENCERT_T = tuple[PRIVATE_KEY_TYPES, list[x509.Certificate]]


def _convert_to_naive_utc_time(time: datetime.datetime) -> datetime:
    """Normalizes a datetime to a naive datetime in UTC.

    time -- datetime to normalize. Assumed to be in UTC if not timezone
            aware.
    """
    if time.tzinfo is not None:
        offset = time.utcoffset()
        offset = offset if offset else timedelta()
        return time.replace(tzinfo=None) - offset
    else:
        return time


class Version(Enum):
    v1 = 0
    v3 = 2


class FrankenBackend(Backend):
    def create_x509_certificate(
        self,
        builder: x509.CertificateBuilder,
        private_key: PRIVATE_KEY_TYPES,
        algorithm: hashes.HashAlgorithm | None,
    ) -> x509.Certificate:
        if builder._public_key is None:
            raise TypeError("Builder has no public key.")
        self._x509_check_signature_params(private_key, algorithm)

        # Resolve the signature algorithm.
        evp_md = self._evp_md_x509_null_if_eddsa(private_key, algorithm)

        # Create an empty certificate.
        x509_cert = self._lib.X509_new()
        x509_cert = self._ffi.gc(x509_cert, self._lib.X509_free)

        # Set the x509 version.
        res = self._lib.X509_set_version(x509_cert, builder._version.value)
        self.openssl_assert(res == 1)

        # Set the subject's name.
        res = self._lib.X509_set_subject_name(
            x509_cert, _encode_name_gc(self, builder._subject_name)
        )
        self.openssl_assert(res == 1)

        # Set the subject's public key.
        res = self._lib.X509_set_pubkey(
            x509_cert,
            builder._public_key._evp_pkey,  # type: ignore[union-attr]
        )
        self.openssl_assert(res == 1)

        # Set the certificate serial number.
        serial_number = _encode_asn1_int_gc(self, builder._serial_number)
        res = self._lib.X509_set_serialNumber(x509_cert, serial_number)
        self.openssl_assert(res == 1)

        # Set the "not before" time.
        self._set_asn1_time(
            self._lib.X509_getm_notBefore(x509_cert), builder._not_valid_before
        )

        # Set the "not after" time.
        self._set_asn1_time(
            self._lib.X509_getm_notAfter(x509_cert), builder._not_valid_after
        )

        # Add extensions.
        self._create_x509_extensions(
            extensions=builder._extensions,
            handlers=self._extension_encode_handlers,
            x509_obj=x509_cert,
            add_func=self._lib.X509_add_ext,
            gc=True,
        )

        # Set the issuer name.
        res = self._lib.X509_set_issuer_name(
            x509_cert, _encode_name_gc(self, builder._issuer_name)
        )
        self.openssl_assert(res == 1)

        # Sign the certificate with the issuer's private key.
        res = self._lib.X509_sign(
            x509_cert,
            private_key._evp_pkey,  # type: ignore[union-attr]
            evp_md,
        )
        if res == 0:
            errors = self._consume_errors_with_text()
            raise ValueError("Signing failed", errors)

        return self._ossl2cert(x509_cert)


_backend = FrankenBackend()


def _get_backend(backend: Backend | None) -> Backend:
    global _backend
    return _backend


class CertificateBuilder:
    def __init__(
        self,
        issuer_name=None,
        subject_name=None,
        public_key=None,
        serial_number=None,
        not_valid_before=None,
        not_valid_after=None,
        extensions=[],
    ) -> None:
        self._version = Version.v3
        self._issuer_name = issuer_name
        self._subject_name = subject_name
        self._public_key = public_key
        self._serial_number = serial_number
        self._not_valid_before = not_valid_before
        self._not_valid_after = not_valid_after
        self._extensions = extensions

    def issuer_name(self, name: Name) -> CertificateBuilder:
        """
        Sets the CA's distinguished name.
        """
        return CertificateBuilder(
            name,
            self._subject_name,
            self._public_key,
            self._serial_number,
            self._not_valid_before,
            self._not_valid_after,
            self._extensions,
        )

    def subject_name(self, name: Name) -> CertificateBuilder:
        """
        Sets the requestor's distinguished name.
        """
        return CertificateBuilder(
            self._issuer_name,
            name,
            self._public_key,
            self._serial_number,
            self._not_valid_before,
            self._not_valid_after,
            self._extensions,
        )

    def public_key(
        self,
        key: PUBLIC_KEY_TYPES,
    ) -> CertificateBuilder:
        """
        Sets the requestor's public key (as found in the signing request).
        """
        return CertificateBuilder(
            self._issuer_name,
            self._subject_name,
            key,
            self._serial_number,
            self._not_valid_before,
            self._not_valid_after,
            self._extensions,
        )

    def serial_number(self, number: int) -> CertificateBuilder:
        """
        Sets the certificate serial number.
        """
        return CertificateBuilder(
            self._issuer_name,
            self._subject_name,
            self._public_key,
            number,
            self._not_valid_before,
            self._not_valid_after,
            self._extensions,
        )

    def not_valid_before(self, time: datetime.datetime) -> CertificateBuilder:
        """
        Sets the certificate activation time.
        """
        time = _convert_to_naive_utc_time(time)
        return CertificateBuilder(
            self._issuer_name,
            self._subject_name,
            self._public_key,
            self._serial_number,
            time,
            self._not_valid_after,
            self._extensions,
        )

    def not_valid_after(self, time: datetime.datetime) -> CertificateBuilder:
        """
        Sets the certificate expiration time.
        """
        time = _convert_to_naive_utc_time(time)
        return CertificateBuilder(
            self._issuer_name,
            self._subject_name,
            self._public_key,
            self._serial_number,
            self._not_valid_before,
            time,
            self._extensions,
        )

    def add_extension(
        self, extval: ExtensionType, critical: bool
    ) -> CertificateBuilder:
        """
        Adds an X.509 extension to the certificate.
        """
        extension = Extension(extval.oid, critical, extval)
        return CertificateBuilder(
            self._issuer_name,
            self._subject_name,
            self._public_key,
            self._serial_number,
            self._not_valid_before,
            self._not_valid_after,
            self._extensions + [extension],
        )

    def sign(
        self,
        private_key: PRIVATE_KEY_TYPES,
        algorithm: hashes.HashAlgorithm,
        backend=None,
    ) -> x509.Certificate:
        """
        Signs the certificate using the CA's private key.
        """
        backend = _get_backend(backend)
        return backend.create_x509_certificate(self, private_key, algorithm)


def random_serial_number() -> int:
    return int.from_bytes(os.urandom(20), "big") >> 1


class FrankenCertGenerator:
    # 여러 가지 ECC (Elliptic Curve Cryptography) 키 타입을 정의합니다.
    ec_ciphers = {
        "ed25519": ed25519.Ed25519PrivateKey,
        "ed448": ed448.Ed448PrivateKey,
        "secp256r1": ec.SECP256R1,
        "secp384r1": ec.SECP384R1,
        "secp521r1": ec.SECP521R1,
    }
    
    # 여러 가지 해시 알고리즘을 정의합니다.
    hash_algos = {
        "md5": hashes.MD5,
        "sha1": hashes.SHA1,
        "sha224": hashes.SHA224,
        "sha256": hashes.SHA256,
        "sha384": hashes.SHA384,
        "sha512": hashes.SHA512,
        "sha512_224": hashes.SHA512_224,
        "sha512_256": hashes.SHA512_256,
        "blake2b": functools.partial(hashes.BLAKE2b, 64),
        "blake2s": functools.partial(hashes.BLAKE2s, 32),
        "sha3-224": hashes.SHA3_224,
        "sha3-256": hashes.SHA3_256,
        "sha3-384": hashes.SHA3_384,
        "sha3-512": hashes.SHA3_512,
    }

    def __init__(
        self,
        seed: list[x509.Certificate],
        ca_cert: x509.Certificate,
        ca_priv: PRIVATE_KEY_TYPES,
        config: dict,
    ) -> None:
        self.seed = seed
        self.ca_cert = ca_cert
        self.ca_priv = ca_priv
        self.digest = config["digest"]
        self.ext_mod_probability: float = config["ext_mod_probability"]
        self.flip_probability: float = config["flip_probability"]
        self.invalid: bool = config["invalid"]
        self.invalid_ts_probability: float = config["invalid_ts_probability"]
        self.keylen: int = config["keylen"]
        self.keytype: str = config["keytype"]
        self.max_depth: int = config["max_depth"]
        self.max_extensions: int = config["max_extensions"]
        self.randomize_hash: bool = config["randomize_hash"]
        self.randomize_serial: bool = config["randomize_serial"]
        self.randomize_keytype: bool = config["randomize_keytype"]
        self.self_signed_prob: float = config["self_signed_prob"]

    # keytype이 "rsa"인 경우 RSA 키를 생성합니다.
    # 그렇지 않은 경우 ECC 키를 생성합니다.
    # randomize_keytype이 True인 경우, ECC 키 타입을 무작위로 선택합니다.
    def _generate_priv(self) -> PRIVATE_KEY_TYPES:
        key: PRIVATE_KEY_TYPES | None = None
        t = self.keytype
        # TODO: Consider RSA in the randomized cert stuff as well.
        if t == "rsa":
            size = self.keylen
            key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=size,
            )
        else:
            if self.randomize_keytype:
                cipher = random.choice(list(self.ec_ciphers.values()))
            else:
                cipher = self.ec_ciphers[t]
            if isinstance(cipher, ed25519.Ed25519PrivateKey):
                key = cipher.generate()
            else:
                key = ec.generate_private_key(cipher)
        assert key
        return key

    # 인증서 생성
    def _generate_cert(
        self,
        issuer: x509.Name | None,
        signing_key: PRIVATE_KEY_TYPES,
        extensions: dict,
    ) -> tuple[PRIVATE_KEY_TYPES, x509.Certificate]:
        private_key = self._generate_priv()
        public_key = private_key.public_key()
        builder = CertificateBuilder()

        # 인증서의 유효기간 (not_before, not_after)을 설정합니다.
        # invalid_ts_probability에 따라 잘못된 유효 기간을 설정할지 결정합니다.
        # Set random not_before and not_after values.
        not_before = None
        not_after = None
        if random.random() < self.invalid_ts_probability:
            # 잘못된 타임스탬프를 생성하는 경우
            if random.random() < 0.5:
                # 아직 유효하지 않은 인증서로
                # Generate not yet valid cert.
                not_before = datetime.now() + timedelta(days=1)
            else:
                # 이미 만료된 인증서로
                # Generate expired cert.
                not_after = datetime.now() - timedelta(days=1)
        else:
            # 유효한 타임스탬프를 생성하는 경우
            # 랜덤으로 시드 인증서 선택하여 not_before 값 가져옵니다.
            # 랜덤으로 시드 인증서 선택하여 not_after 값 가져옵니다.
            # 각각의 값을 가져오는 인증서가 서로 다르면 not_after가 not_before보다 이른 경우가 생길 수도 있지 않나하는 의문...
            # 하나의 시드 인증서에서 두 값을 모두 가져오는 게 더 안정적이지 않을까?
            pick = random.choice(self.seed)
            not_before = pick.not_valid_before
            pick = random.choice(self.seed)
            not_after = pick.not_valid_after

        # None이 아닌지 검증
        assert not_before is not None
        assert not_after is not None

        # 인증서 빌더에 유효 기간 설정
        builder = builder.not_valid_before(not_before)
        builder = builder.not_valid_after(not_after)

        # serial number를 설정합니다.
        # Set serial number.
        
        # self.randomize_serial이 True인 경우, x509.random_serial_number()를 사용하여 무작위 일련번호를 생성합니다.
        if self.randomize_serial:
            builder = builder.serial_number(x509.random_serial_number())
        # self.randomize_serial이 False인 경우, 시드 인증서 중에서 무작위로 하나를 선택합니다.
        # 선택된 인증서의 serial number를 사용합니다.
        # serial number가 음수일 경우에는 양수로 변환하여 사용합니다.
        else:
            pick = random.choice(self.seed)
            s = pick.serial_number
            s = s if s > 0 else s * -1
            builder = builder.serial_number(s)

        # 주체 이름을 설정합니다.
        # 시드 인증서 중에서 무작위로 하나를 선택하여 선택된 인증서의 subject를 사용합니다.
        # Subject는 인증서의 소유자를 식별하는 정보를 포함하는 필드입니다.
        # Set subject.
        pick = random.choice(self.seed)
        builder = builder.subject_name(pick.subject)
        
        # 주요 속성
        # Subject 필드는 다음과 같은 주요 속성들로 구성될 수 있습니다:
        # CN (Common Name): 주로 도메인 이름 또는 개인의 이름.
        # O (Organization): 조직의 이름.
        # OU (Organizational Unit): 조직의 부서.
        # L (Locality): 도시 또는 지역.
        # ST (State or Province): 주 또는 도.
        # C (Country): 국가 코드 (2자리).
        # EmailAddress: 이메일 주소.

        # 인증서 검증에서 Subject의 역할
        # 인증서 검증 과정에서 Subject 필드는 다음과 같은 방법으로 사용됩니다:

        # 도메인 검증:
        # 웹 서버 인증서의 경우, 클라이언트는 서버가 제시하는 인증서의 Subject 필드에서 Common Name(CN) 또는 Subject Alternative Name(SAN) 필드가 요청한 도메인과 일치하는지 확인합니다.
        # 예를 들어, 클라이언트가 https://example.com에 접속할 때, 서버의 인증서 Subject 필드의 CN이 example.com과 일치해야 합니다.
        
        # 신뢰 체계 구축 (인증서 체인 관계):
        # 상위 인증서 (CA 인증서):
        # subject: CA의 이름 정보.
        # issuer: CA를 발급한 상위 기관의 이름 정보 (루트 CA의 경우 자기 자신).
        
        # 하위 인증서 (일반 인증서):
        # subject: 인증서 소유자의 이름 정보.
        # issuer: 인증서를 발급한 CA의 이름 정보.
        
        # 검증 과정
        # 루트 CA: 자기 자신을 발급했으므로 subject와 issuer가 동일합니다.
        # 중간 CA: 상위 인증서의 subject ("CN=Root CA")와 일치하는 issuer ("CN=Root CA")를 가집니다.
        # 엔드 엔티티: 상위 인증서의 subject ("CN=Intermediate CA")와 일치하는 issuer ("CN=Intermediate CA")를 가집니다.
        
        # 정책 준수:
        # 일부 보안 정책은 인증서의 Subject 필드에 특정 정보가 포함되어 있는지 확인하여 인증서를 승인하거나 거부할 수 있습니다.
        # 예를 들어, 조직의 내부 정책에 따라 특정 OU 또는 조직명으로 발급된 인증서만 신뢰할 수 있습니다.

        # 인증서의 발급자를 설정합니다.
        # issuer가 None인 경우, 시드 인증서 리스트에서 무작위로 하나의 인증서를 선택하고, 그 인증서의 발급자 이름(issuer)을 사용합니다.
        # builder.issuer_name(pick.issuer)를 사용하여 발급자 이름을 설정합니다.
        # Issuer와 Subject가 같은, 즉 자가 서명된 인증서를 만들어 볼 수도 있지 않나?
        if issuer is None:
            pick = random.choice(self.seed)
            builder = builder.issuer_name(pick.issuer)
            
        # self.invalid가 True인 경우, 잘못된 발급자 이름을 설정합니다.
        # x509.Name 객체를 생성하여 무작위 문자열을 포함하는 Common Name 속성을 설정합니다.
        elif self.invalid:
            builder = builder.issuer_name(
                x509.Name(
                    [
                        x509.NameAttribute(NameOID.COMMON_NAME, _random_str()),
                    ]
                )
            )
            # x509.SubjectAlternativeName 확장을 추가하여 무작위 도메인 이름을 포함하도록 설정합니다.
            # critical=False로 설정하여 이 확장이 중요하지 않음을 나타냅니다.
            
            # Critical 속성의 의미
            # critical=True:
            # 이 확장은 인증서를 검증하는 데 필수적입니다.
            # 인증서를 사용하는 애플리케이션(예: 웹 브라우저, 이메일 클라이언트 등)이 이 확장을 이해하지 못하거나 지원하지 않는 경우, 인증서는 검증에 실패해야 합니다.
            # 예를 들어, BasicConstraints 확장이 critical=True로 설정되면, CA 인증서의 유효성을 판단하는 데 필수적인 정보로 간주됩니다.
            # critical=False:
            # 이 확장은 인증서를 검증하는 데 필수적이지 않습니다.
            # 인증서를 사용하는 애플리케이션이 이 확장을 이해하지 못하더라도, 인증서의 검증에 실패하지 않습니다.
            # SAN(Subject Alternative Name) 확장과 같은 일부 확장은 critical=False로 설정되는 경우가 많습니다.
            # critical=False로 설정해도 특정 플러그인이나 애플리케이션에서는 해당 필드를 검증하여 실패할 수 있나? -> 있는 듯
            
            # Subject Alternative Name(SAN)은 X.509 인증서의 확장 필드 중 하나로, 하나의 인증서에 대해 여러 대체 이름을 지정할 수 있게 합니다. 
            # SAN 필드는 주로 웹 서버 인증서에서 사용되며, 단일 인증서로 여러 도메인 이름이나 IP 주소를 보호할 수 있도록 합니다.
            builder = builder.add_extension(
                x509.SubjectAlternativeName([x509.DNSName(_random_str())]),
                critical=False,
            )
        
        # issuer가 주어진 경우, 이를 사용하여 발급자 이름을 설정합니다.
        # builder.issuer_name(issuer)를 사용하여 발급자 이름을 설정합니다.
        # x509.SubjectAlternativeName 확장을 추가하여 발급자의 이름을 도메인 이름 형식으로 설정합니다.
        # critical=False로 설정하여 이 확장이 중요하지 않음을 나타냅니다.
        else:
            builder = builder.issuer_name(issuer)
            builder = builder.add_extension(
                x509.SubjectAlternativeName([x509.DNSName(issuer.rfc4514_string())]),
                critical=False,
            )

        # 정상적인 X.509 인증서에서 Subject 필드의 Common Name(CN)과 Subject Alternative Name(SAN) 확장의 DNS 이름이 반드시 일치해야 하는 것은 아닙니다.
        # 하지만, 많은 경우에서 CN은 SAN 확장에 포함된 이름 중 하나로 설정되는 것이 일반적입니다.
        
        # Subject의 CN:
        # Subject 필드의 CN은 인증서 소유자의 주된 식별자입니다.
        # 일반적으로 웹 서버 인증서의 경우 도메인 이름이 설정됩니다.
        # Subject Alternative Name(SAN):
        # SAN 확장은 여러 대체 이름을 포함할 수 있습니다. 이는 도메인 이름, IP 주소, 이메일 주소 등을 포함할 수 있습니다.
        
        # 예시  
        # 일반적인 웹 서버 인증서:
        # CN: www.example.com
        # SAN: example.com, www.example.com, mail.example.com
        # 이 경우, CN은 SAN에 포함된 이름 중 하나입니다. 
        # 이는 많은 브라우저와 클라이언트가 SAN 확장을 지원하지 않을 때, CN을 사용하여 도메인을 확인할 수 있도록 하기 위함입니다.
        
        # SAN 확장이 없는 경우
        # SAN 확장이 없을 경우, 클라이언트는 Subject 필드의 CN을 사용하여 인증서의 유효성을 확인합니다.
        # 그러나 현재는 SAN 확장이 널리 사용되며, 많은 인증 기관(CA)은 SAN 확장을 포함하지 않은 인증서를 발급하지 않습니다.
        
        # SAN 확장이 있는 경우
        # SAN 확장이 있는 경우, 클라이언트는 주로 SAN 확장의 내용을 사용하여 인증서의 유효성을 확인합니다.
        # 이 경우 CN은 보조적인 역할을 할 수 있습니다.
        
        # SAN 확장에 CN이 포함되지 않는 인증서 생성해볼 수 있지 않을까?
        # 대부분의 CA와 보안 전문가들은 CN이 SAN 확장에 포함되도록 인증서를 발급하는 것을 권장합니다. 이는 호환성과 보안성 모두를 보장하기 위함입니다.
        # SAN 확장이 있는 경우 주로 SAN 확장의 내용을 사용하기 때문에 의미가 없을 수도 있지만 기술적으로 생성은 가능함.
        
        # Basic Constraints 확장: 인증서의 유형과 사용 범위를 정의하는 확장 필드입니다.
        # ca=False: 이 인증서가 CA(인증 기관) 인증서가 아님을 나타냅니다. 즉, 이 인증서는 다른 인증서를 발급할 수 없습니다.
        # path_length=None: CA 인증서 체인의 최대 길이를 설정합니다. None은 제한이 없음을 의미합니다.
        # 하지만 ca=False로 설정되었기 때문에 이 값은 의미가 없습니다.
        # critical=True: 이 확장이 필수적임을 나타냅니다.
        # 인증서를 검증하는 모든 클라이언트는 이 확장을 이해하고 처리해야 합니다
        # 이 확장을 처리할 수 없는 클라이언트는 인증서를 신뢰하지 않습니다.
        builder = builder.public_key(public_key)
        builder = builder.add_extension(
            x509.BasicConstraints(ca=False, path_length=None),
            critical=True,
        )

        # 인증서에 확장을 추가합니다.
        # sample 변수에 0부터 self.max_extensions 사이의 무작위 정수를 할당합니다.
        # choices 변수에 extensions 딕셔너리의 키들 중에서 sample 개의 무작위 키를 선택합니다.
        # new_extensions 변수에 choices에 있는 각 키에 대해 extensions 딕셔너리에서 무작위 값을 선택하여 리스트를 만듭니다.
        sample = random.randint(0, self.max_extensions)
        choices = random.sample(extensions.keys(), sample)
        new_extensions = [random.choice(extensions[name]) for name in choices]
        # new_extensions 리스트의 각 확장을 순회하면서 확장을 추가합니다.
        for extension in new_extensions:
            # FIXME: How to implement this with python cryptography?
            # if random.random() < self.config["ext_mod_probability"]:
            #     randstr = "".join(chr(random.randint(0, 255)) for i in range(7))
            #     extension.set_data(randstr)
            
            try:
                # random.random() < self.flip_probability 조건에 따라 확장의 critical 속성을 반전시킵니다.
                if random.random() < self.flip_probability:
                    builder.add_extension(
                        extension.value, critical=not extension.critical
                    )
                # critical 속성이 반전되지 않으면, 원래의 critical 속성으로 확장을 추가합니다.
                else:
                    builder.add_extension(extension.value, critical=extension.critical)
                    
            # 확장을 추가하는 과정에서 중복된 확장은 ValueError가 발생할 수 있으므로, try-except 블록으로 이를 처리하여 중복 확장은 건너뜁니다.
            except ValueError:
                pass
        
        # 인증서를 서명하여 최종적으로 생성합니다.
        
        # cert 변수를 None으로 초기화합니다. 이후에 생성된 인증서를 이 변수에 저장할 것입니다.
        cert = None
        # ED25519 expects None here.
        
        # self.randomize_hash가 True이면, self.hash_algos에서 무작위로 해시 알고리즘을 선택합니다.
        if self.randomize_hash:
            digest = random.choice(list(self.hash_algos.values()))()
        # self.randomize_hash가 False이면, 설정된 self.digest 해시 알고리즘을 사용합니다.
        else:
            digest = self.hash_algos[self.digest]()
            
        # ED25519는 내부적으로 자체 해시를 사용하기 때문에 해시 알고리즘을 사용하지 않는다는 의미입니다.
        # ED25519 MUST NOT be used with a hash function, since
        # hashing is backed into the signing digest itself!
        
        # 인증 기관(CA) 인증서인 경우
        # issuer가 있는 경우, 이는 서명된 인증서 체인을 생성하는 것입니다.
        if issuer:
          # •	서명 키가 ED25519 타입이면 해시 알고리즘을 None으로 설정합니다.
            if isinstance(signing_key, ed25519.Ed25519PrivateKey):
                digest = None
            # 인증서 빌더(builder) 객체에 서명하여 인증서를 생성합니다.
            # Generate cert chain.
            cert = builder.sign(
                private_key=signing_key,
                algorithm=digest,
            )
        # 자가 서명된 인증서인 경우
        # issuer가 없는 경우, 이는 자가 서명된 인증서를 생성하는 것입니다.
        else:
            # 서명 키가 ED25519 타입이면 해시 알고리즘을 None으로 설정합니다.
            if isinstance(private_key, ed25519.Ed25519PrivateKey):
                digest = None
            # 인증서 빌더(builder) 객체에 서명하여 자가 서명된 인증서를 생성합니다.
            # Generate self signed cert.
            cert = builder.sign(
                private_key=private_key,
                algorithm=digest,
            )
        # cert가 None이 아님을 확인하기 위해 assert 문을 사용합니다.
	      # 최종적으로 생성된 개인 키(private_key)와 인증서(cert)를 반환합니다.
        assert cert is not None
        return private_key, cert

    # FRANKENCERT_T 형식의 인증서 체인을 생성합니다.
    # FRANKENCERT_T = tuple[PRIVATE_KEY_TYPES, list[x509.Certificate]]
    # number: 생성할 인증서 체인의 수.
	  # extensions: 확장 필드의 딕셔너리. 제공되지 않으면 self.seed를 사용하여 확장 필드를 생성합니다.
    def generate(
        self,
        number: int,
        extensions: dict | None = None,
    ) -> list[FRANKENCERT_T]:
        log("Generating frankencerts…")

        # 확장 필드가 제공되지 않으면 self.seed를 사용하여 확장 필드 딕셔너리를 생성합니다.
	      # 최대 확장 필드 수를 확장 필드 딕셔너리의 키 수와 self.max_extensions 중 작은 값으로 설정합니다.
        # 항상 None으로 넘어옴
        if extensions is None:
            extensions = get_extension_dict(self.seed)
        self.max_extensions = min(self.max_extensions, len(extensions.keys()))
        
        # self.max_depth 만큼의 개인 키를 생성하고 privs 리스트에 저장합니다.
        # generate the key pairs once and reuse them for faster
        # frankencert generation
        privs = []
        for _ in range(self.max_depth):
            priv = self._generate_priv()
            privs.append(priv)
            
        # 생성된 키 쌍의 수가 self.max_depth와 일치하는지 확인합니다.
        assert len(privs) == self.max_depth

        # number 만큼의 인증서 체인을 생성합니다.
        certs = []
        for i in range(number):
            log(f"\rProgress: {i+1}/{number}", end="")
            
            # 각 인증서 체인에 대해 초기 설정을 합니다.
            chain = []
            signing_key = self.ca_priv
            issuer = self.ca_cert.issuer
            priv = None
            length = random.randint(1, self.max_depth)
            
            # 체인의 길이를 1에서 self.max_depth 사이의 무작위 값으로 설정합니다.
	          # 코드에서 체인의 길이가 1이고 무작위 값이 self.self_signed_prob보다 작으면 자가 서명된 인증서를 생성합니다.
            # 그런데 체인의 길이가 1이면 최상위 인증서(self.ca_cert)가 있으므로 하위 인증서를 생성하지 않아야 하는거 아닌가?
            if length == 1 and random.random() < self.self_signed_prob:
                # _generate_cert에 issuer를 None으로 전달하면 issuer와 subject를 모두 랜덤으로 선정된 인증서에서 추출합니다.
                # 그럼 최상위 인증서 아래에 issuer와 subject가 랜덤하게 설정된 인증서가 생성됩니다.
                # 이게 맞는지?
                issuer = None

            for j in range(length):
                # issuer, siging_key, extensions 매개변수(상위 인증서의 발급자, 상위 인증서의 개인키, 확장) 전달하여 priv(개인키), cert(인증서) 생성
                priv, cert = self._generate_cert(
                    issuer,
                    signing_key,
                    extensions,
                )
                # issuer, sigining_key을 새로 생성된 인증서로 업데이트하여 다음 인증서 생성에 사용합니다.
                # 현재 코드에서는 생성된 모든 인증서가 같은 확장 필드를 가집니다. 
                # generate 함수에서 각 인증서를 생성할 때 extensions 딕셔너리를 매번 동일하게 전달하기 때문입니다. 
                # 이로 인해 생성된 인증서 체인에 속하는 모든 인증서가 동일한 확장 필드를 가지게 됩니다.
                # 인증서를 생성할 때 마다 다른 확장 필드를 가지게 할 순 없나?
                signing_key = priv
                issuer = cert.issuer
                # 새로 생성된 인증서를 chain 리스트에 추가합니다. 이 리스트는 하나의 인증서 체인을 형성합니다.
                chain.append(cert)
            # 루프가 완료되면 chain 리스트를 역순으로 저장합니다. 이는 인증서 체인이 올바른 순서로 저장되도록 하기 위함입니다.
	          # priv와 역순으로 정렬된 chain을 certs 리스트에 추가합니다. certs 리스트는 여러 인증서 체인을 포함합니다.
            certs.append((priv, list(reversed(chain))))
        log()
        assert len(certs) == number
        return certs


def _random_str() -> str:
  
    r = ""
    s = list(string.printable)
    for _ in range(random.randint(1, 128)):
        r += random.choice(s)
    return r


def _dump_certs_file(path: Path, franken_certs: list[FRANKENCERT_T]) -> None:
    for i, franken_cert in enumerate(franken_certs):
        key, cert_list = franken_cert
        p = path.joinpath(f"frankencert-{i}.pem")
        buf = io.BytesIO()
        pem = key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
        buf.write(pem)
        for cert in cert_list:
            pem = cert.public_bytes(encoding=serialization.Encoding.PEM)
            buf.write(pem)
        p.write_bytes(buf.getbuffer())


def _dump_certs_stdout(franken_certs: list[FRANKENCERT_T]) -> None:
    for franken_cert in franken_certs:
        key, cert_list = franken_cert
        pem = key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
        print(pem.decode())
        for cert in cert_list:
            pem = cert.public_bytes(encoding=serialization.Encoding.PEM)
            print(pem.decode())


def dump_certs(path: Path, certs: list[FRANKENCERT_T]) -> None:
    log(f"Writing frankencerts to {path}…")

    if str(path) == "-":
        _dump_certs_stdout(certs)
    else:
        base = Path(path)
        if not base.exists():
            base.mkdir(parents=True)
        _dump_certs_file(base, certs)

# load_seed 함수는 주어진 디렉토리에서 시드 인증서를 로드합니다.
# 각 파일을 읽어 PEM 형식의 X.509 인증서로 변환하고, 로드에 실패한 파일은 별도의 로그 파일에 기록합니다.
# 로드된 인증서들은 리스트에 저장되어 반환됩니다.
def load_seed(path: Path) -> list[x509.Certificate]:
    log("Loading seed certificates…")

    certs = []
    certsfiles = list(path.iterdir())

    with open("pyLoad-fails.txt", "w") as f:
        for i, infile in enumerate(certsfiles):
            log(f"\rProgress: {i+1}/{len(certsfiles)}", end="")
            data = infile.read_bytes()
            try:
                cert = x509.load_pem_x509_certificate(data)
                certs.append(cert)
            except Exception:
                f.write(f"failed: {infile}\n")
    return certs

# load_ca 함수는 주어진 파일 경로에서 루트 CA의 개인 키와 인증서를 로드합니다.
# 파일을 바이트 형태로 읽은 후, PEM 형식의 인증서와 개인 키를 각각 로드하여 반환합니다.
def load_ca(path: Path) -> tuple[PRIVATE_KEY_TYPES, x509.Certificate]:
    buf = path.read_bytes()
    ca_cert = x509.load_pem_x509_certificate(buf)
    ca_priv = serialization.load_pem_private_key(buf, password=None)
    return ca_priv, ca_cert

# get_extension_dict 함수는 주어진 인증서 리스트에서 각 인증서의 확장을 추출하여 정리한 중첩 사전을 생성합니다. 
# 이 사전은 확장의 OID(Object Identifier)를 키로, 해당 OID에 속하는 확장들의 리스트를 값으로 가지는 구조입니다.
def get_extension_dict(certs: list[x509.Certificate]) -> dict:
    d = collections.defaultdict(dict)
    
    # 각 인증서의 확장 리스트를 순회합니다.
    # 확장의 OID를 문자열로 변환한 값을 첫번째 키로, 확장의 값을 두번째 키로(중첩 사전) 사용합니다.
    # 확장 객체를 value로 사용하여 사전에 추가
    for cert in certs:
        for extension in cert.extensions:
            d[extension.oid.dotted_string][extension.value] = extension
    
    # 중첩 사전에 저장된 확장 객체들을 리스트로 변환
    for k in d.keys():
        d[k] = list(d[k].values())
    return d
  
  # 예시
  # 인증서 1:
  # OID: 2.5.29.14 (Subject Key Identifier)
  # 값: "abc123"
  # OID: 2.5.29.35 (Authority Key Identifier)
  # 값: "def456"
  
  # 인증서 2:
  # OID: 2.5.29.14 (Subject Key Identifier)
  # 값: "xyz789"
  
  # 인증서 1 처리
  # d["2.5.29.14"]["abc123"] = 확장1
  # d["2.5.29.35"]["def456"] = 확장2

  # 인증서 2 처리
  # d["2.5.29.14"]["xyz789"] = 확장3

  # 최종 사전 구조
  # {
  #     "2.5.29.14": [확장1, 확장3],
  #     "2.5.29.35": [확장2],
  # }

# parse_args 함수는 명령줄 인자를 파싱하여 프로그램 실행 시 필요한 설정과 옵션을 사용자로부터 입력받는 역할을 합니다.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate specially crafted SSL certificates for "
            "testing certificate validation code in SSL/TLS "
            "implementations"
        )
    )
    # 각 인자는 add_argument 메서드를 사용하여 정의됩니다.
    
    # 시드 인증서 경로
    parser.add_argument(
        "-s",
        "--seed",
        metavar="PATH",
        required=True,
        type=Path,
        help="Path to folder containing seed certificates",
    )

    # 루트 CA 경로
    parser.add_argument(
        "-c",
        "--ca",
        metavar="PATH",
        required=True,
        type=Path,
        help="Path to root ca file, containing priv key and certificate",
    )
    
    # 출력 경로
    parser.add_argument(
        "-o",
        "--out",
        metavar="PATH",
        default="-",
        type=Path,
        help="Out directory, or stdout with '-'",
    )
    
    # 로드 옵션
    parser.add_argument(
        "--load-only",
        action="store_true",
        help="Only load seeds and do not generate frankencerts",
    )
    
    # 키 타입
    parser.add_argument(
        "-k",
        "--keytype",
        default="secp256r1",
        help="Specify the keytype, e.g. secp256r1, see openssl",
    )
    
    # 키 길이
    parser.add_argument(
        "-l",
        "--keylen",
        metavar="INT",
        type=int,
        default=2048,
        help="Keylength, only for RSA keys",
    )
    
    # 해시 알고리즘
    parser.add_argument(
        "-d",
        "--digest",
        default="sha256",
        help="Hash algorithm to generate the signature",
    )
    
    #생성할 인증서 수
    parser.add_argument(
        "-n",
        "--number",
        type=int,
        metavar="INT",
        default=10,
        help="Quartity of generated certs",
    )
    
    # 잘못된 인증서 생성 옵션
    parser.add_argument(
        "-i",
        "--invalid",
        action="store_true",
        help="Introduce more brokenness",
    )
    
    # 최대 확장 수
    parser.add_argument(
        "--max-extensions",
        type=int,
        metavar="INT",
        default=20,
        help="Max X.509 extensions, currently not used",
    )
    
    # 최대 신뢰 체인 길이
    parser.add_argument(
        "--max-depth",
        type=int,
        metavar="INT",
        default=3,
        help="Maximum trust chain length",
    )
    
    # 확장 변경 확률
    parser.add_argument(
        "--ext-mod-prob",
        type=float,
        metavar="FLOAT",
        default=0.0,
    )
    
    #Critical 플래그 뒤집기 확률
    parser.add_argument(
        "--flip-critical-prob",
        type=float,
        metavar="FLOAT",
        default=0.25,
    )
    
    # 자가 서명 확률
    parser.add_argument(
        "--self-signed-prob",
        type=float,
        metavar="FLOAT",
        default=0.25,
    )
    
    # 잘못된 타임 스탬프 확률
    parser.add_argument(
        "--invalid-ts-prob",
        type=float,
        metavar="FLOAT",
        default=0.0,
    )
    
    # 일련번호 무작위화
    parser.add_argument(
        "--randomize-serial",
        action="store_true",
        help="Randomize the serial number of the generated certificates",
    )
    
    # 해시 함수 무작위화
    parser.add_argument(
        "--randomize-hash",
        action="store_true",
        help="Randomize the hash function generating the signatures [!BUGS!]",
    )
    
    # 키 타입 무작위화
    parser.add_argument(
        "--randomize-keytype",
        action="store_true",
        help="Use different keys: ec, ed25519, …",
    )
    
    # 파싱된 인자 반환
    return parser.parse_args()


def main() -> None:
    # parse_args 함수는 명령줄 인자를 파싱하여 args 객체에 저장합니다. 
    # 이 객체는 사용자가 제공한 옵션과 인자 값을 포함합니다.
    args = parse_args()
    
    # load_seed 함수는 주어진 경로(args.seed)에서 시드 인증서를 로드합니다.
    # 이 인증서들은 프랭큰서트 생성 시 참조됩니다.
    seed = load_seed(args.seed)
    
    # --load-only 옵션이 지정된 경우, 시드 인증서만 로드하고 프로그램을 종료합니다.
    if args.load_only:
        sys.exit(0)
        
    # load_ca 함수는 주어진 경로(args.ca)에서 루트 CA의 개인 키와 인증서를 로드합니다.
    # 루트 CA가 필요한 이유!!
    # 루트 CA는 다른 인증서를 발급하고 서명하는 권한을 가진 가장 신뢰할 수 있는 인증 기관입니다. 
    # 루트 CA의 개인 키는 이러한 인증서를 서명하는 데 사용되며, 루트 CA의 인증서는 이러한 서명을 검증하는 데 사용됩니다.
    ca_priv, ca_cert = load_ca(args.ca)
    
    # 설정을 담은 딕셔너리를 구성하여 프랭큰서트 생성 시 사용합니다.
    # 이 설정은 다양한 인증서 생성 옵션을 포함합니다.
    config = {
        "digest": args.digest,
        "ext_mod_probability": args.ext_mod_prob,
        "flip_probability": args.flip_critical_prob,
        "invalid": args.invalid,
        "invalid_ts_probability": args.invalid_ts_prob,
        "keylen": args.keylen,
        "keytype": args.keytype,
        "max_depth": args.max_depth,
        "max_extensions": args.max_extensions,
        "randomize_hash": args.randomize_hash,
        "randomize_serial": args.randomize_serial,
        "randomize_keytype": args.randomize_keytype,
        "self_signed_prob": args.self_signed_prob,
    }

    # FrankenCertGenerator 클래스를 초기화하여 프랭큰서트를 생성할 준비를 합니다.
    frankenstein = FrankenCertGenerator(seed, ca_cert, ca_priv, config)
    
    # generate 메서드를 호출하여 지정된 수(args.number)의 프랭큰서트를 생성합니다.
    frankencerts = frankenstein.generate(args.number)
    
    # dump_certs 함수는 생성된 프랭큰서트를 지정된 경로(args.out)에 출력합니다. 경로가 "-"인 경우, 표준 출력으로 출력합니다.
    dump_certs(args.out, frankencerts)
