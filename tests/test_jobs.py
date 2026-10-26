"""Job cache, output isolation and tuning cleanup without model runs."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import io
import unittest

from experiments import engine, jobs, runner


class JobTests(unittest.TestCase):
    def test_tuning_outputs_are_removed_after_cache_commit(self):
        with TemporaryDirectory() as temporary:
            out = Path(temporary)
            folder = out / "tuning_raw" / "screen" / "trial"
            task = dict(
                group="no_rl",
                kind="random",
                map=dict(name="map"),
                hp=dict(name="c00"),
                seed=1,
                stage="screen",
            )

            def fake_case(job):
                folder.mkdir(parents=True)
                (folder / "temporary.dat").write_text("temporary trial output")
                return dict(task=job, path=str(folder), objective=0.2)

            with (
                patch.object(engine, "OUT", out),
                patch.object(engine, "case", side_effect=fake_case) as call,
            ):
                with (
                    patch.object(jobs, "ProcessPoolExecutor", ThreadPoolExecutor),
                    redirect_stdout(io.StringIO()),
                ):
                    first = jobs.run_pool([task], "screen", workers=1)
                    self.assertFalse(folder.exists())
                    self.assertTrue((out / "index/screen" / f"{jobs.task_id(task)}.json").exists())
                    again = jobs.run_pool([task], "screen", workers=1)
                self.assertEqual(first, again)
                self.assertEqual(call.call_count, 1)

    def test_run_and_tuning_cannot_share_output_directory(self):
        with TemporaryDirectory() as temporary, patch.object(engine, "OUT", Path(temporary)):
            initial = runner._configuration("run")
            self.assertEqual(runner._configuration("run"), initial)
            with self.assertRaises(AssertionError):
                runner._configuration("tune")


if __name__ == "__main__":
    unittest.main()
