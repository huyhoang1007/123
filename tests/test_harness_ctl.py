"""Unit tests for scripts/harness_ctl.py."""
import pathlib
import sqlite3
import sys
import tempfile
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import harness_ctl  # noqa: E402


class HarnessCtlTests(unittest.TestCase):
    def test_check_syntax_clean_repo(self):
        self.assertEqual(harness_ctl.check_syntax(REPO_ROOT), [])

    def test_backup_sqlite_db_preserves_data_readonly(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            src_db = root / "memory.sqlite3"
            conn = sqlite3.connect(str(src_db))
            try:
                conn.execute("CREATE TABLE items (k TEXT PRIMARY KEY, v TEXT)")
                conn.execute("INSERT INTO items VALUES ('a', '1')")
                conn.commit()
            finally:
                conn.close()

            bak_dir = root / "backups"
            res = harness_ctl.backup_sqlite_db(src_db, out_dir=bak_dir)
            self.assertTrue(res["success"])
            bak_path = pathlib.Path(res["backup_path"])
            self.assertTrue(bak_path.is_file())
            self.assertGreater(res["size_bytes"], 0)

            check = sqlite3.connect(f"{bak_path.as_uri()}?mode=ro", uri=True)
            try:
                rows = check.execute("SELECT k, v FROM items").fetchall()
            finally:
                check.close()
            self.assertEqual(rows, [("a", "1")])

    def test_gather_doctor_report_and_clean_pycache(self):
        rep = harness_ctl.gather_doctor_report(REPO_ROOT)
        self.assertEqual(rep["syntax_errors"], [])
        self.assertTrue(rep["git"]["is_repo"])
        self.assertIn("mcp", rep["dependencies"])

        with tempfile.TemporaryDirectory() as tmp:
            fake_repo = pathlib.Path(tmp)
            pc = fake_repo / "pkg" / "__pycache__"
            pc.mkdir(parents=True)
            (pc / "mod.pyc").write_bytes(b"pyc")
            removed = harness_ctl.clean_pycache(fake_repo)
            self.assertEqual(removed, 1)
            self.assertFalse(pc.exists())

    def test_estimate_subcommand_succeeds(self):
        rc = harness_ctl.main(["estimate", "--text", "hello harness world", "--model", "gemini-2.5-flash"])
        self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()
