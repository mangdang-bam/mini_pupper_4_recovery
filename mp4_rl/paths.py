"""Where this package's files live, and the file hash every record uses. Package-relative only, so a source checkout,
an installed wheel and the robot's deployed copy agree; the robot's model and measured data are ``mp4_sim``'s
(``mp4_sim.paths``). numpy-free: the robot's runner imports it."""
import hashlib
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
SKILLS = PACKAGE / "skills"             # the bank: one folder per policy (``mp4_rl.skills.REGISTRY``)
WALKER_POLICIES = SKILLS / "walker" / "policies"
GETUP_POLICIES = SKILLS / "getup" / "policies"
RUNS = Path("runs")                     # where the tools write by default: relative to the working directory

TEXT_SUFFIXES = {".xml", ".json", ".py", ".csv", ".md", ".txt"}


def sha256(path) -> str:
    """The file's digest; text sources are hashed with LF line endings, so a Windows checkout gives the same digest
    (the same rule as ``mp4_sim.paths.sha256``)."""
    p = Path(path)
    b = p.read_bytes()
    if p.suffix.lower() in TEXT_SUFFIXES:
        b = b.replace(b"\r\n", b"\n")
    return hashlib.sha256(b).hexdigest()
