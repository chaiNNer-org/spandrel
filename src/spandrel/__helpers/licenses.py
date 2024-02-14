from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class License:
    """
    A data class for software licensing information.
    """

    name: str
    """
    The name of the license.
    """
    spdx_id: str | None
    """
    The SPDX identifier of the license.
    """
    commercial: bool
    """
    Whether commercial use is allowed.
    """

    @staticmethod
    def from_known(spdx_id: KnownLicense) -> License:
        """
        Returns the license information for a known license.
        """
        return KNOWN_LICENSES[spdx_id]


KnownLicense = Literal[
    "MIT",
    "Apache-2.0",
    "BSD-3-Clause",
    "GPL-3.0",
    "CC-BY-4.0",
    "CC-BY-NC-4.0",
    "CC-BY-SA-4.0",
    "CC-BY-NC-SA-4.0",
    "CC0-1.0",
]
"""
A type alias for the SPDX license IDs of known licenses.
"""

KNOWN_LICENSES: dict[KnownLicense, License] = {
    "MIT": License(
        "MIT License",
        spdx_id="MIT",
        commercial=True,
    ),
    "Apache-2.0": License(
        "Apache License 2.0",
        spdx_id="Apache-2.0",
        commercial=True,
    ),
    "BSD-3-Clause": License(
        "BSD 3-Clause License",
        spdx_id="BSD-3-Clause",
        commercial=True,
    ),
    "GPL-3.0": License(
        "GNU General Public License v3.0",
        spdx_id="GPL-3.0",
        commercial=True,
    ),
    "CC-BY-4.0": License(
        "Creative Commons Attribution 4.0 International",
        spdx_id="CC-BY-4.0",
        commercial=True,
    ),
    "CC-BY-NC-4.0": License(
        "Creative Commons Attribution-NonCommercial 4.0 International",
        spdx_id="CC-BY-NC-4.0",
        commercial=False,
    ),
    "CC-BY-SA-4.0": License(
        "Creative Commons Attribution-ShareAlike 4.0 International",
        spdx_id="CC-BY-SA-4.0",
        commercial=True,
    ),
    "CC-BY-NC-SA-4.0": License(
        "Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International",
        spdx_id="CC-BY-NC-SA-4.0",
        commercial=False,
    ),
    "CC0-1.0": License(
        "Creative Commons Zero v1.0 Universal",
        spdx_id="CC0-1.0",
        commercial=True,
    ),
}
