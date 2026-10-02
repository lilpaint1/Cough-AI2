"""
librosa compatibility helper for Vercel.

Vercel's Python bundler may omit librosa's __init__.pyi.
librosa uses lazy_loader.attach_stub(), which requires that
stub file at runtime.

This helper patches lazy_loader only when importing librosa
without its adjacent .pyi file, while preserving the normal
librosa package and feature extraction behavior.
"""

import os


_PATCHED = False


def _patch_lazy_loader():
    global _PATCHED

    if _PATCHED:
        return

    import lazy_loader

    original_attach_stub = lazy_loader.attach_stub

    def safe_attach_stub(
        package_name,
        filename,
    ):
        if package_name == "librosa":

            stub_file = (
                os.path.splitext(filename)[0]
                + ".pyi"
            )

            # Vercel: __init__.pyi may be omitted
            if not os.path.exists(stub_file):

                return lazy_loader.attach(
                    package_name,
                    submodules=[
                        "core",
                        "feature",
                        "effects",
                        "util",
                    ],
                )

        return original_attach_stub(
            package_name,
            filename,
        )

    lazy_loader.attach_stub = (
        safe_attach_stub
    )

    _PATCHED = True


def get_librosa():
    """
    Return:
        librosa,
        librosa.core,
        librosa.feature,
        librosa.effects,
        librosa.util

    The top-level librosa API is avoided where the missing
    __init__.pyi would normally be required.
    """

    _patch_lazy_loader()

    import librosa

    from librosa import (
        core,
        feature,
        effects,
        util,
    )

    return (
        librosa,
        core,
        feature,
        effects,
        util,
    )
