from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]


class HeadlessPackagingTests(unittest.TestCase):
    def test_headless_requirements_are_exactly_pinned_and_exclude_desktop_stack(self):
        lines = [
            line.strip()
            for line in (ROOT / "requirements-headless.txt").read_text("utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        requirements = "\n".join(lines).lower()

        for package in (
            "pybioclip==2.1.6",
            "open-clip-torch==3.3.0",
            "torch==2.11.0+cpu",
            "torchvision==0.26.0+cpu",
            "numpy==2.1.3",
            "pillow==12.1.1",
            "rawpy==0.26.1",
        ):
            self.assertIn(package, lines)
        for forbidden in ("pyqt", "pywebview", "tensorflow", "onnxruntime", "msvc"):
            self.assertNotIn(forbidden, requirements)
        for line in lines:
            if line.startswith("--"):
                continue
            self.assertRegex(line, r"^[a-zA-Z0-9_.-]+==[a-zA-Z0-9_.+-]+$")

    def test_runtime_image_is_nonroot_and_contains_only_historical_modules(self):
        dockerfile = (ROOT / "Dockerfile.headless").read_text("utf-8")
        runtime = dockerfile.split(" AS runtime", 1)[1]

        self.assertIn("USER 65532:65532", runtime)
        self.assertIn(
            'ENTRYPOINT ["python", "-m", "analyzer.kestrel_analyzer.historical_cli"]',
            runtime,
        )
        self.assertNotIn("USER root", runtime)
        for forbidden in (
            "metadata_writer.py",
            "ratings.py",
            "pipeline.py",
            "api_bridge.py",
            "culling",
        ):
            self.assertNotIn(forbidden, dockerfile)

    def test_container_workflow_is_least_privilege_and_publishes_only_sha_image(self):
        workflow = (ROOT / ".github/workflows/headless-container.yml").read_text("utf-8")

        self.assertIn("pull_request:", workflow)
        self.assertRegex(workflow, r"push:\s*\n\s*branches: \[main\]")
        self.assertIn("permissions:\n  contents: read", workflow)
        self.assertEqual(1, workflow.count("packages: write"))
        self.assertIn("github.event_name == 'push'", workflow)
        self.assertIn("sha-${{ github.sha }}", workflow)
        self.assertNotIn(":latest", workflow)
        self.assertIn("platforms: linux/amd64", workflow)
        self.assertIn("--read-only", workflow)
        action_uses = re.findall(r"uses:\s+[^@\s]+@([^\s]+)", workflow)
        self.assertTrue(action_uses)
        self.assertTrue(all(re.fullmatch(r"[0-9a-f]{40}", pin) for pin in action_uses))


if __name__ == "__main__":
    unittest.main()
