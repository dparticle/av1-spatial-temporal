from __future__ import annotations

import unittest

from av1_spatial_temporal import cli, encoder, operations, workflow


class PublicApiTests(unittest.TestCase):
    def test_public_entry_points_are_importable(self) -> None:
        self.assertTrue(callable(cli.main))
        self.assertTrue(callable(encoder.encode_av1_svc))
        self.assertTrue(callable(operations.verify_reconstruction))
        self.assertTrue(callable(workflow.run_poc))


if __name__ == "__main__":
    unittest.main()
