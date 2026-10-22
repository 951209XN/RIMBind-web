import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from backend.hosted_install import extract_locked_zip, install_authorized_assets


class HostedInstall(unittest.TestCase):
    def archive(self, folder, name, body=b'frozen bytes', symlink=False):
        path = Path(folder)/'bundle.zip'
        with zipfile.ZipFile(path, 'w') as archive:
            info = zipfile.ZipInfo(name)
            if symlink:
                info.external_attr = 0o120777 << 16
            archive.writestr(info, body)
        return path, hashlib.sha256(path.read_bytes()).hexdigest(), {name: hashlib.sha256(body).hexdigest()}

    def test_installation_is_opt_in_and_requires_secret(self):
        for env in ({}, {'RIMBIND_INSTALL_ASSETS': '1'},
                    {'RIMBIND_INSTALL_ASSETS': '1', 'RIMBIND_THIRD_PARTY_TERMS_CONFIRMED': '1'}):
            with patch.dict(os.environ, env, clear=True):
                with self.assertRaises(RuntimeError):
                    install_authorized_assets()

    def test_locked_regular_files_extract(self):
        with tempfile.TemporaryDirectory() as folder:
            path, digest, files = self.archive(folder, 'nested/known.bin')
            target = Path(folder)/'output'
            extract_locked_zip(path, digest, target, files, 1000)
            self.assertEqual((target/'nested/known.bin').read_bytes(), b'frozen bytes')

    def test_wrong_archive_or_member_sha_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path, digest, files = self.archive(folder, 'known.bin')
            with self.assertRaises(RuntimeError):
                extract_locked_zip(path, 'wrong', Path(folder)/'output', files, 1000)
            with self.assertRaises(RuntimeError):
                extract_locked_zip(path, digest, Path(folder)/'output', {'known.bin': 'wrong'}, 1000)

    def test_traversal_symlink_and_oversize_rejected(self):
        for name, symlink, cap in (('../escape', False, 1000), ('/escape', False, 1000),
                                   ('known', True, 1000), ('known', False, 1)):
            with tempfile.TemporaryDirectory() as folder:
                path, digest, files = self.archive(folder, name, symlink=symlink)
                with self.assertRaises(RuntimeError):
                    extract_locked_zip(path, digest, Path(folder)/'output', files, cap)

    def test_unexpected_archive_member_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path, digest, files = self.archive(folder, 'known')
            with self.assertRaises(RuntimeError):
                extract_locked_zip(path, digest, Path(folder)/'output', {}, 1000)


if __name__ == '__main__':
    unittest.main()
