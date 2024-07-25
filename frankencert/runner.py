from __future__ import annotations

import argparse
import json
import multiprocessing
import sqlite3
import subprocess
import sys
import warnings
from abc import ABC, abstractmethod
from argparse import ArgumentParser, Namespace
from collections.abc import Iterator
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import cryptography
import mbedtls
import OpenSSL.version
from cryptography.utils import CryptographyDeprecationWarning
from gallia.command import Script
from gallia.config import Config, load_config_file
from mbedtls.x509 import CRT as MBEDCertificate
from OpenSSL.crypto import FILETYPE_PEM
from OpenSSL.crypto import load_certificate as openssl_load_cert

from frankencert.asn1 import parse_asn1_json

warnings.filterwarnings("ignore", category=CryptographyDeprecationWarning)

from cryptography import x509  # noqa

SCHEMA = """
CREATE TABLE scan_run (
    id INTEGER PRIMARY KEY,
    command TEXT check(json_valid(command)),
    start_time REAL NOT NULL,
    end_time REAL,
    exit_code INTEGER
) STRICT;

CREATE TABLE plugin (
    id INTEGER PRIMARY KEY,
    name TEXT,
    description TEXT,
    version TEXT
) STRICT;

CREATE TABLE stdin (
    id INTEGER PRIMARY KEY,
    asn1_tree TEXT check(json_valid(asn1_tree)),
    zlint_result TEXT check(json_valid(zlint_result)),
    data BLOB
) STRICT;

CREATE TABLE scan_result (
    id INTEGER PRIMARY KEY,
    plugin_id INTEGER NOT NULL REFERENCES plugin(id) ON UPDATE CASCADE ON DELETE CASCADE,
    run_id INTEGER NOT NULL REFERENCES scan_run(id) ON UPDATE CASCADE ON DELETE CASCADE,
    loader TEXT NOT NULL,
    start_time REAL NOT NULL,
    end_time REAL NOT NULL,
    success INTEGER,
    stdin_id INTEGER REFERENCES stdin(id) ON UPDATE CASCADE ON DELETE CASCADE,
    stdout BLOB,
    stderr BLOB
) STRICT;

CREATE INDEX success_index ON scan_result(run, success);
CREATE INDEX loader_index ON scan_result(run, loader);
"""

# 주어진 인증서 데이터를 zlint라는 외부 도구를 사용하여 검증하고, 그 결과를 문자열로 반환하는 함수입니다.
# 출력은 JSON 형식으로 제공됩니다.

# 인증서 검증 결과 예시
# {
#   "certificate": {
#     "subject": "CN=example.com, O=Example Organization, C=US",
#     "issuer": "CN=Example CA, O=Example Organization, C=US",
#     "serialNumber": "123456789",
#     "notBefore": "2023-01-01T00:00:00Z",
#     "notAfter": "2024-01-01T00:00:00Z"
#   },
#   "lints": {
#     "w_basic_constraints_not_critical": {
#       "description": "Conforming CAs MUST mark this extension as critical",
#       "result": "warn",
#       "details": "The Basic Constraints extension is not marked as critical."
#     },
#     "e_sub_cert_not_is_ca": {
#       "description": "Subscriber certificates must not assert the cA boolean",
#       "result": "error",
#       "details": "The cA boolean is asserted in a subscriber certificate."
#     },
#     "e_sub_cert_key_usage_missing": {
#       "description": "Subscriber certificates must contain the Key Usage extension",
#       "result": "error",
#       "details": "The Key Usage extension is missing in the subscriber certificate."
#     },
#     "e_sub_cert_key_usage_cert_sign_bit_set": {
#       "description": "Subscriber certificates must not have the keyCertSign bit set",
#       "result": "error",
#       "details": "The keyCertSign bit is set in the Key Usage extension of the subscriber certificate."
#     }
#   }
# }

def zlint(cert: bytes) -> str:
    p = subprocess.run(["zlint"], input=cert, capture_output=True, check=True)
    return p.stdout.decode()

# 모든 플러그인이 상속해야 하는 추상 기본 클래스입니다.
# 이 클래스는 플러그인이 구현해야 할 기본 인터페이스를 정의합니다.
class BasePlugin(ABC):
    
    # __str__ 및 __repr__: 클래스의 이름을 문자열로 반환합니다.
    # run: 인증서를 처리하는 추상 메서드입니다. 하위 클래스에서 구현해야 합니다.
    # name: 플러그인의 이름을 반환합니다.
    # description: 플러그인의 설명을 반환하는 추상 속성입니다. 하위 클래스에서 구현해야 합니다.
    # version: 플러그인의 버전을 반환하는 추상 속성입니다. 하위 클래스에서 구현해야 합니다.
    def __str__(self) -> str:
        return self.__class__.__name__

    def __repr__(self) -> str:
        return str(self)

    @abstractmethod
    def run(self, cert: bytes) -> dict[str, Any]:
        ...

    @property
    def name(self) -> str:
        return repr(self)

    @property
    @abstractmethod
    def description(self) -> str:
        ...

    @property
    @abstractmethod
    def version(self) -> str:
        ...

# SubprocessPlugin 클래스는 외부 명령어를 서브프로세스로 실행하는 플러그인의 기본 클래스입니다.
# BasePlugin을 상속받아 구현되었습니다.
class SubprocessPlugin(BasePlugin):
    COMMAND: list[str] = []
    VERSION_COMMAND: list[str] = []
    
    # __init__: 플러그인의 명령어와 버전 명령어를 초기화합니다.
    # run: 인증서를 서브프로세스(병렬 처리 가능하게 함)로 실행하여 처리합니다. 오류가 발생하면 stderr, stdout, exitcode를 반환합니다.
    # description: 서브프로세스로 호출하는 명령어를 설명합니다.
    # version: 버전 명령어를 실행하여 플러그인의 버전을 반환합니다.
    def __init__(
        self,
        command: list[str] | None = None,
        version_command: list[str] | None = None,
    ) -> None:
        self.command = self.COMMAND if command is None else command
        self.version_command = (
            self.VERSION_COMMAND if version_command is None else version_command
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.command})"

    def run(self, cert: bytes) -> dict[str, Any]:
        p = subprocess.run(
            self.command,
            input=cert,
            capture_output=True,
        )

        if p.returncode != 0:
            return {
                "stderr": p.stderr,
                "stdout": p.stdout,
                "exitcode": p.returncode,
            }
        return {}

    @property
    def description(self) -> str:
        return f"calls {self.command} as a subprocess"

    @property
    def version(self) -> str:
        p = subprocess.run(
            self.version_command,
            capture_output=True,
            check=True,
        )
        return p.stdout.decode()

# GoPlugin 클래스는 Go 언어로 작성된 로더를 사용하여 인증서를 처리하는 플러그인입니다. SubprocessPlugin을 상속받아 구현되었습니다.
class GoPlugin(SubprocessPlugin):
    def __init__(
        self,
        command: list[str],
        version: str,
    ) -> None:
        self._version = version
        super().__init__(command)

    @property
    def version(self) -> str:
        return self._version

# GNUTLS_Plugin 클래스는 GnuTLS 도구를 사용하여 인증서를 처리하는 플러그인입니다. 
class GNUTLS_Plugin(SubprocessPlugin):
    COMMAND = [
        "certtool",
        "--certificate-info",
        "--load-certificate",
        "/dev/stdin",
    ]
    VERSION_COMMAND = [
        "certtool",
        "--version",
    ]

# MBED_TLS_Plugin 클래스는 MbedTLS 라이브러리를 사용하여 인증서를 처리하는 플러그인입니다.
class MBED_TLS_Plugin(BasePlugin):
    def run(self, cert: bytes) -> dict[str, Any]:
        out = {}
        try:
            MBEDCertificate.from_PEM(cert.decode())
        except Exception as e:
            out["stderr"] = str(e)

        return out

    @property
    def version(self) -> str:
        return mbedtls.version.version

    @property
    def description(self) -> str:
        return "mbedtls via a wrapper python library in process"

# OpenSSL_Plugin 클래스는 OpenSSL 라이브러리를 사용하여 인증서를 처리하는 플러그인입니다.
class OpenSSL_Plugin(BasePlugin):
    def run(self, cert: bytes) -> dict[str, Any]:
        out = {}
        try:
            openssl_load_cert(FILETYPE_PEM, cert)
        except Exception as e:
            out["stderr"] = str(e)

        return out

    @property
    def description(self) -> str:
        return OpenSSL.version.__summary__

    @property
    def version(self) -> str:
        return OpenSSL.version.__version__

# PythonPlugin 클래스는 Python의 cryptography 패키지를 사용하여 인증서를 처리하는 플러그인입니다.
class PythonPlugin(BasePlugin):
    def run(self, cert: bytes) -> dict[str, Any]:
        out = {}
        try:
            x509.load_pem_x509_certificate(cert)
        except Exception as e:
            out["stderr"] = str(e)

        return out

    @property
    def description(self) -> str:
        return "python cryptography package"

    @property
    def version(self) -> str:
        return cryptography.__version__


class DBHandler:
    def __init__(self, path: Path, db: sqlite3.Connection, cur: sqlite3.Cursor) -> None:
        self.path = path
        self.db = db
        self.cur = cur

    def commit(self) -> None:
        self.db.commit()

    def close(self) -> None:
        self.db.commit()
        self.db.close()

    def create(self) -> None:
        self.cur.executescript(SCHEMA)

    @classmethod
    def connect(cls, path: Path) -> DBHandler:
        create = False if path.exists() else True

        db = sqlite3.connect(path)
        cur = db.cursor()
        # https://phiresky.github.io/blog/2020/sqlite-performance-tuning/
        sql = [
            "PRAGMA foreign_keys = ON;",
            f"PRAGMA threads = {multiprocessing.cpu_count()};",
            "PRAGMA journal_mode = WAL;",
            "PRAGMA synchronous = normal;",
            "PRAGMA temp_store = memory;",
        ]
        for line in sql:
            cur.execute(line)

        if create:
            cur.executescript(SCHEMA)
            db.commit()

        return cls(path, db, cur)

    def add_plugin(self, name: str, description: str, version: str) -> None:
        self.cur.execute(
            """INSERT INTO plugin(name, description, version) VALUES(?, ?, ?)""",
            (name, description, version),
        )

    def run_add(self, command: list[str], start_time: datetime) -> None:
        self.cur.execute(
            "INSERT INTO scan_run(command, start_time) VALUES(?, ?)",
            (json.dumps(command), start_time.timestamp()),
        )
        self.run_id = self.cur.lastrowid

    def run_finish(self, end_time: datetime, exit_code: int) -> None:
        assert self.run_id, "run_id is not set"
        self.cur.execute(
            "UPDATE scan_run SET end_time=?, exit_code=? WHERE id==?",
            (end_time.timestamp(), exit_code, self.run_id),
        )

    def stdin_add(self, data: bytes) -> int:
        self.cur.execute(
            "INSERT INTO stdin(data) VALUES(?)",
            (data,),
        )
        assert self.cur.lastrowid
        return self.cur.lastrowid

    def get_certs(self) -> Iterator[tuple[int, bytes]]:
        cur = self.cur.execute(
            "SELECT id, data from stdin",
        )
        for row in cur.fetchall():
            yield (row[0], row[1])

    def zlint_result_add(self, cert_id: int, cert: bytes) -> None:
        try:
            zlint_result = json.dumps({"result": json.loads(zlint(cert)), "success": True})
        except subprocess.SubprocessError as e:
            zlint_result = json.dumps({"result": str(e), "success": False})

        self.cur.execute(
            """UPDATE stdin SET zlint_result=? WHERE id==?""", (zlint_result, cert_id)
        )

    def asn1_tree_add(self, cert_id: int, cert: bytes) -> None:
        try:
            asn1_tree = json.dumps(
                {"result": json.loads(parse_asn1_json(cert.decode())), "success": True}
            )
        except Exception as e:
            asn1_tree = json.dumps({"result": str(e), "success": False})

        self.cur.execute(
            """UPDATE stdin SET asn1_tree=? WHERE id==?""", (asn1_tree, cert_id)
        )

    def result_add(
        self,
        loader: str,
        start_time: datetime,
        end_time: datetime,
        success: bool,
        stdin_id: int,
        stdout: bytes | None,
        stderr: bytes | None,
    ) -> int:
        assert self.run_id, "run_id is not set"
        self.cur.execute(
            """INSERT INTO scan_result(
                    run_id,
                    loader,
                    start_time,
                    end_time,
                    success,
                    stdin_id,
                    stdout,
                    stderr
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                self.run_id,
                loader,
                start_time.timestamp(),
                end_time.timestamp(),
                success,
                stdin_id,
                stdout,
                stderr,
            ),
        )
        assert self.cur.lastrowid
        return self.cur.lastrowid

    def purge_unrefed_certs(self) -> None:
        cur = self.cur.execute(
            """
            SELECT stdin.id 
                FROM stdin 
                LEFT JOIN scan_result
                ON stdin.id = scan_result.stdin_id
                WHERE scan_result.stdin_id IS NULL"""
        )

        for row in cur.fetchall():
            row_id = row[0]
            self.cur.execute(f"DELETE FROM stdin WHERE stdin.id = {row_id}")


def parse_line(line: str, as_json: bool) -> bytes:
    # 줄이 JSON 형식인 경우
    # JSON 형식 예: '{"cert": "MIIDdzCCAl+gAwIBAgIEUQIDtzANBgkqhkiG9w0BAQUFADBoMQswCQYDVQQGEwJV..."}'
    if as_json:
        data = json.loads(line)
        return cast(bytes, data["cert"].encode())

    # 줄이 CSV 형식인 경우
    # CSV 형식 예: '1,MIIDdzCCAl+gAwIBAgIEUQIDtzANBgkqhkiG9w0BAQUFADBoMQswCQYDVQQGEwJV...'
    _, cert = line.split(",", maxsplit=1)
    out = (
        b"-----BEGIN CERTIFICATE-----\n"
        + cert.encode().strip()
        + b"\n-----END CERTIFICATE-----\n"
    )
    return out.strip()
    
    # PEM 형식 예
    # -----BEGIN CERTIFICATE-----
    # MIIDdzCCAl+gAwIBAgIEUQIDtzANBgkqhkiG9w0BAQUFADBoMQswCQYDVQQGEwJV
    # UzEVMBMGA1UEChMMRGlnaUNlcnQgSW5jMR8wHQYDVQQLExZEaWdpQ2VydCBUcnVz
    # dCBORVRX...
    # -----END CERTIFICATE-----
    
    # PEM 형식의 바이트 데이터 예
    # pem_data = (
    # b"-----BEGIN CERTIFICATE-----\n"
    # b"MIIDdzCCAl+gAwIBAgIEUQIDtzANBgkqhkiG9w0BAQUFADBoMQswCQYDVQQGEwJV\n"
    # b"UzEVMBMGA1UEChMMRGlnaUNlcnQgSW5jMR8wHQYDVQQLExZEaWdpQ2VydCBUcnVz\n"
    # b"dCBORVRX...\n"
    # b"-----END CERTIFICATE-----\n"
    # )




class Runner(Script):
    COMMAND = "frankencert"
    LOGGER_NAME = "frankenrunner"

    def __init__(self, parser: ArgumentParser, config: Config) -> None:
        super().__init__(parser, config)
        self.db: DBHandler

    def configure_parser(self) -> None:
        self.parser.add_argument(
            "--db",
            type=Path,
            help="path to sqlite database",
            default=self.config.get_value("frankencert.db_path", None),
        )
        self.parser.add_argument(
            "-i",
            "--indices",
            type=int,
            nargs="+",
            help="indices of plugins to run only",
        )
        self.parser.add_argument(
            "--start",
            type=int,
            help="start reading at this offset",
        )
        self.parser.add_argument(
            "--stop",
            type=int,
            help="stop reading at this offset",
        )
        self.parser.add_argument(
            "-j",
            "--json",
            action="store_true",
            help="read from stdin as json",
        )

    ## 플러그인 처리: 인증서를 플러그인으로 처리하고, 결과를 데이터베이스에 저장합니다.
    ## 예외 처리: 플러그인 처리 중 예외가 발생하면 로그에 기록하고 실패로 표시합니다.
    ## self: Runner 클래스의 인스턴스를 가리킵니다.
    ## plugin: BasePlugin 클래스를 상속받은 플러그인 객체입니다. 플러그인은 run 메서드를 가지고 있습니다.
    ## cert: 처리할 인증서 데이터를 나타내는 바이트 문자열입니다.
    ## stdin_id: 데이터베이스에 저장된 입력 데이터의 ID입니다.
    ## 반환 값: 실행 시간, 플러그인 정보, 입력 ID, 결과를 포함하는 딕셔너리를 반환합니다.
    def _process_plugin(
        self,
        plugin: BasePlugin,
        cert: bytes,
        stdin_id: int,
    ) -> dict[str, Any]:
        self.logger.trace(f"running plugin {plugin}")

        out = {
            "start_time": datetime.now(),
            "plugin": repr(plugin),
        }
        res = plugin.run(cert)
        out["end_time"] = datetime.now()
        
        # Do not log successfully parsed runs.        
        ## res가 빈 딕셔너리가 아닌 경우에만 out 딕셔너리에 기록
        if res != {}:
            out["stdin_id"] = stdin_id
            out["res"] = res
        else:
            out["stdin_id"] = None
            out["res"] = None

        return out

    # 비동기 작업의 결과 처리
    # futures: 비동기 작업의 결과를 나타내는 Future 객체의 리스트입니다. 각 Future 객체는 dict 형식의 결과를 반환합니다.
    def _flush_futures(self, futures: list[Future[dict[str, Any]]]) -> None:
        for future in as_completed(futures):
            res = future.result()
            
            # 성공 여부 판단: stdin_id가 None이 아닌 경우, 해당 작업이 실패한 것으로 간주
            # 로그에 플러그인 이름과 결과를 기록하고 success를 False로 설정합니다. 
            success = True
            if res["stdin_id"] is not None:
                self.logger.info(f"{res['plugin']}: {res['res']}")
                success = False

            # We only want failed certs in the database.
            # 작업이 성공한 경우(success가 True인 경우) 결과 처리를 건너뜁니다.
            if success is True:
                continue
            
            # 결과 확인: res["res"]가 None이 아닌 경우(작업이 실패한 경우), stdout과 stderr 값을 가져옵니다.
            # stdout과 stderr가 None이거나 존재하지 않으면 None으로 설정합니다.
            if (r := res["res"]) is not None:
                stdout = (
                    r["stdout"] if "stdout" in r and r["stdout"] is not None else None
                )
                stderr = (
                    r["stderr"] if "stderr" in r and r["stderr"] is not None else None
                )
            else:
                stdout = None
                stderr = None
            
            # result_add 호출하여 DB에 결과 추가
            self.db.result_add(
                loader=res["plugin"],
                start_time=res["start_time"],
                end_time=res["end_time"],
                success=success,
                stdin_id=res["stdin_id"],
                stderr=stderr.encode() if isinstance(stderr, str) else stderr,
                stdout=stdout.encode() if isinstance(stdout, str) else stdout,
            )
        # purge_unrefed_certs: 참조되지 않은 인증서를 데이터베이스에서 정리합니다.
        # commit: 데이터베이스 변경 사항을 커밋하여 영구 저장합니다.
        self.db.purge_unrefed_certs()
        self.db.commit()

    # 인증서를 검사(lint)하고, 검사 결과를 데이터베이스에 추가하는 역할을 합니다.
    # 인증서 가져오기: self.db.get_certs()를 호출하여 데이터베이스에서 인증서 목록을 가져옵니다.
    # 각 항목은 (id, cert) 형태로, 인증서의 ID와 인증서 데이터를 포함합니다.
    
    # ASN.1 트리 추가: self.db.asn1_tree_add(id, cert)를 호출하여 인증서의 ASN.1 트리를 데이터베이스에 추가합니다.
    # ASN.1(추상 구문 표기법 1)은 인증서의 구조를 표현하는 표준 형식입니다.
    
    # ZLint 결과 추가: self.db.zlint_result_add(id, cert)를 호출하여 인증서에 대한 ZLint 결과를 데이터베이스에 추가합니다.
    # ZLint는 X.509 인증서를 검사하여 여러 가지 규칙 위반 여부를 확인하는 도구입니다.
    def _lint_certs(self) -> None:
        for id, cert in self.db.get_certs():
            self.db.asn1_tree_add(id, cert)
            self.db.zlint_result_add(id, cert)

    # 전체 워크플로우
    # 인증서 파싱 및 데이터베이스 저장:
    # 표준 입력으로부터 인증서를 읽어 파싱하고 데이터베이스에 저장합니다.
    # 플러그인 적용:
    # 각 인증서에 대해 여러 플러그인을 병렬로 적용하여 결과를 처리하고 저장합니다.
    # Lint 검사:
    # 모든 플러그인 적용이 완료된 후에, lint 검사를 수행합니다.
    # (lint 검사를 왜 맨 마지막에 하는 지 잘 모르겠음)
    def main(self, args: Namespace) -> None:
        
        # DBHandler.connect: 데이터베이스에 연결합니다. args.db는 데이터베이스 파일 경로를 나타냅니다.
        # run_add: 현재 실행을 데이터베이스에 추가합니다. 실행 명령과 실행 시간을 기록합니다.
        self.db = DBHandler.connect(args.db)
        self.db.run_add(sys.argv, datetime.now())

        # 플러그인 목록: 여러 플러그인을 초기화하여 리스트에 저장합니다.
        plugins: list[BasePlugin] = [
            GNUTLS_Plugin(),
            MBED_TLS_Plugin(),
            OpenSSL_Plugin(),
            PythonPlugin(),
            GoPlugin(["loaders/go/go1.16.15-loader"], "1.16.15"),
            GoPlugin(["loaders/go/go1.17.13-loader"], "1.17.13"),
            GoPlugin(["loaders/go/go1.18.6-loader"], "1.18.6"),
            GoPlugin(["loaders/go/go1.19.1-loader"], "1.19.1"),
            GoPlugin(["loaders/go/go1.20.4-loader"], "1.20.4"),
            GoPlugin(["loaders/go/go1.21rc2-loader"], "1.21rc2"),
        ]

        # 각 플러그인을 데이터베이스에 추가하여 플러그인 이름, 설명, 버전을 저장합니다.
        for plugin in plugins:
            self.db.add_plugin(str(plugin), plugin.description, plugin.version)

        self.logger.info(f"loaded plugins: {plugins}")

        # 인증서 처리!
        # ThreadPoolExecutor: 스레드 풀을 생성하여 비동기 작업을 병렬로 실행합니다.
        # futures 리스트 초기화: 비동기 작업의 미래 객체를 저장할 리스트를 초기화합니다.
        with ThreadPoolExecutor() as executor:
            futures = []

            # 추측!
            # 표준 입력으로부터 각 줄 마다 서로 다른 인증서가 주어짐 
            for i, line in enumerate(sys.stdin):
                n = i + 1
                if n % 1000 == 0:
                    self.logger.info(f"parsing cert #{n}")

                # parse_line 함수는 입력된 인증서 데이터를 파싱하여 바이트 형식의 PEM 인코딩된 인증서로 변환합니다.
                # 표준 입력은 JSON 혹은 CSV 형식
                cert = parse_line(line, args.json)

                # PEM 인코딩된 인증서를 DB에 저장
                stdin_id = self.db.stdin_add(cert)

                # 각 플러그인에 대해 _process_plugin 메서드를 비동기로 실행하고, Future 객체를 futures 리스트에 추가합니다.
                # 인덱스 필터링: args.indices가 설정된 경우, 해당 인덱스의 플러그인은 건너뜁니다.
                for j, plugin in enumerate(plugins):
                    if args.indices is not None and j in args.indices:
                        continue

                    fut = executor.submit(
                        self._process_plugin,
                        plugin,
                        cert,
                        stdin_id,
                    )
                    futures.append(fut)
                    
                # futures 처리: futures 리스트의 길이가 1000이 될 때마다 _flush_futures 메서드를 호출하여 비동기 작업의 결과를 처리합니다.
                # futures 초기화: 처리 후 futures 리스트를 초기화합니다.
                if len(futures) % 1000 == 0:
                    self._flush_futures(futures)
                    futures = []

            self._flush_futures(futures)
            self._lint_certs()

    def entry_point(self, args: Namespace) -> int:
        exit_code = 0

        try:
            exit_code = super().entry_point(args)
        except Exception as e:
            self.logger.exception(f"exception occured: {e}")
            exit_code = 1
        finally:
            self.db.run_finish(datetime.now(), exit_code)
            self.db.commit()
            self.db.close()

        return exit_code


def main() -> None:
    ## 설정 파일 초기화
    config, _ = load_config_file()
    
    ## 파서 초기화
    parser = argparse.ArgumentParser()
    
    ## 설정 파일과 파서로 runner 초기화
    runner = Runner(parser, config)
    
    ## 커맨드에 입력된 인자 파싱해서 runner의 entry_point 메소드에 전달하여 실행
    ## sys.exit을 통해 entry_point가 반환한 종료코드를 이용해서 script 종료
    sys.exit(runner.entry_point(parser.parse_args()))

if __name__ == "__main__":
    main()
