"""
surveyor.registry — pluggable component registry.

Lets every surveyor capability be swapped for a newer implementation without
changing the orchestrator, the backend, or the frontend.  A component is just a
factory (callable) registered under a name; the orchestrator asks the registry
for the *current* implementation at run time.

Example
-------
    from surveyor.registry import register, get_component

    @register("damage_assessor")
    def _make():
        return MyNewDamageModel(weights="best.pt")

    assessor = get_component("damage_assessor")

Override at runtime (e.g. after training a better model)::

    register("damage_assessor", lambda: BetterModel(), override=True)

Components are created lazily and cached as singletons (one instance per name)
so heavy models load only once.  Call :func:`reset` to drop the cache and force
re-creation (useful after hot-swapping weights).
"""

from __future__ import annotations

import logging
import threading
from typing import Callable, Dict, Any

logger = logging.getLogger(__name__)


class ComponentRegistry:
    """Thread-safe name -> factory registry with singleton caching."""

    def __init__(self) -> None:
        self._factories: Dict[str, Callable[[], Any]] = {}
        self._instances: Dict[str, Any] = {}
        self._lock = threading.RLock()

    # ── registration ──────────────────────────────────────────────────────
    def register(
        self,
        name: str,
        factory: Callable[[], Any],
        *,
        override: bool = False,
    ) -> Callable[[], Any]:
        with self._lock:
            if name in self._factories and not override:
                raise ValueError(
                    f"Component {name!r} already registered; pass override=True "
                    "to replace it."
                )
            self._factories[name] = factory
            # Drop any cached instance so the new factory takes effect.
            self._instances.pop(name, None)
            logger.info("Registered surveyor component %r (override=%s)", name, override)
        return factory

    # ── resolution ────────────────────────────────────────────────────────
    def get(self, name: str) -> Any:
        with self._lock:
            if name in self._instances:
                return self._instances[name]
            if name not in self._factories:
                raise KeyError(
                    f"No surveyor component registered under {name!r}. "
                    f"Known: {sorted(self._factories)}"
                )
            instance = self._factories[name]()
            self._instances[name] = instance
            return instance

    def has(self, name: str) -> bool:
        return name in self._factories

    def reset(self, name: str | None = None) -> None:
        """Drop cached singleton(s) so the next get() rebuilds them."""
        with self._lock:
            if name is None:
                self._instances.clear()
            else:
                self._instances.pop(name, None)

    def names(self) -> list[str]:
        return sorted(self._factories)


# Module-level singleton registry.
registry = ComponentRegistry()


def register(name: str, factory: Callable[[], Any] | None = None, *, override: bool = False):
    """Register a component factory.

    Usable as a decorator (``@register("x")``) or a direct call
    (``register("x", factory)``).
    """
    if factory is None:
        def _decorator(f: Callable[[], Any]):
            registry.register(name, f, override=override)
            return f
        return _decorator
    return registry.register(name, factory, override=override)


def get_component(name: str) -> Any:
    return registry.get(name)


def register_default_components() -> None:
    """Register the built-in baseline implementations (idempotent).

    Called by the orchestrator on first use.  Each factory imports its module
    lazily so that simply importing the registry never pulls in torch/cv2.
    """
    if registry.has("ocr_engine"):
        return  # already initialised

    def _ocr():
        from surveyor.ocr_engine import OCREngine
        return OCREngine()

    def _seal():
        from surveyor.seal_inspector import SealInspector
        return SealInspector()

    def _damage():
        from surveyor.damage_assessor import DamageAssessor
        return DamageAssessor()

    def _detector():
        from surveyor.detection import ContainerTrackingDetector
        return ContainerTrackingDetector()

    def _small_object():
        from surveyor.small_object import SmallObjectPipeline
        return SmallObjectPipeline()

    registry.register("ocr_engine", _ocr)
    registry.register("seal_inspector", _seal)
    registry.register("damage_assessor", _damage)
    registry.register("container_detector", _detector)
    registry.register("small_object_pipeline", _small_object)
