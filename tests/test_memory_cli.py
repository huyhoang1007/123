"""Unit tests for scripts/memory_cli.py and scripts/bench_retrieve.py."""
import json
import pathlib
import sys
import tempfile
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
for _p in (str(REPO_ROOT), str(SCRIPTS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bench_retrieve  # noqa: E402
import memory_cli  # noqa: E402


class MemoryCliAndBenchTests(unittest.TestCase):
    def test_memory_cli_save_stats_search_export_and_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            db_path = root / "mem.sqlite3"

            r1 = memory_cli.save_entry(
                db_path,
                mtype="semantic",
                key="ps51-utf8",
                value="Always set OutputEncoding UTF-8 in PowerShell 5.1",
                tags="windows,powershell",
            )
            self.assertEqual(r1["saved"]["key"], "ps51-utf8")
            self.assertIsNone(r1["backup"])

            # Second write creates an automatic backup snapshot
            r2 = memory_cli.save_entry(
                db_path,
                mtype="procedural",
                key="sqlite-backup",
                value="Use sqlite3.backup() instead of copying WAL files",
                tags="sqlite,safety",
            )
            self.assertTrue(r2["backup"]["success"])
            self.assertTrue(pathlib.Path(r2["backup"]["backup_path"]).is_file())

            st = memory_cli.stats_readonly(db_path)
            self.assertEqual(st["total"], 2)
            self.assertEqual(st["tiers"]["semantic"]["count"], 1)
            self.assertEqual(st["tiers"]["procedural"]["count"], 1)

            hits = memory_cli.search_entries(db_path, query="PowerShell UTF-8", top_k=3, min_sim=0.05)
            self.assertGreaterEqual(len(hits), 1)
            self.assertEqual(hits[0]["key"], "ps51-utf8")

            exported = memory_cli.export_readonly(db_path)
            self.assertEqual(len(exported), 2)

            db2 = root / "mem_copy.sqlite3"
            imp = memory_cli.import_entries(db2, exported)
            self.assertEqual(imp["imported"], 2)
            self.assertEqual(memory_cli.stats_readonly(db2)["total"], 2)

    def test_bench_retrieve_measures_corpus_and_modes(self):
        with tempfile.TemporaryDirectory() as tmp:
            proj = pathlib.Path(tmp)
            (proj / "auth.py").write_text(
                "def verify_token(token: str) -> bool:\n"
                "    return token.startswith('tok_')\n\n"
                "def rotate_secret(old: str) -> str:\n"
                "    return old + '_v2'\n",
                encoding="utf-8",
            )
            rep = bench_retrieve.run_benchmark(
                paths=[str(proj)],
                query="verify token rotation",
                modes=("concise", "minimal"),
                isolated=True,
            )
            self.assertEqual(rep["corpus_files"], 1)
            self.assertGreater(rep["corpus_raw_tokens"], 0)
            self.assertEqual(len(rep["modes"]), 2)
            self.assertEqual(rep["modes"][0]["output_mode"], "concise")
            self.assertGreater(rep["modes"][0]["output_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
