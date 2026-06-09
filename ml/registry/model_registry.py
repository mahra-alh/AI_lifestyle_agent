"""
Firestore-backed model registry.

Tracks every trained model artifact with:
- version string  (e.g. "v2025.06.08")
- artifact paths  (GCS bucket path or local path)
- feature contract hash  (detects train/serve drift)
- training metadata  (rows, label source, eval metrics)
- status  ("staging" | "active" | "retired")

One Firestore collection: `model_registry`
One document per version, keyed by version string.
One special document keyed `__active__` that points to the current
production version — the serving layer reads only this document.

Usage
-----
From training (train_cells.py):
    from ml.registry.model_registry import ModelRegistry
    registry = ModelRegistry()
    registry.register(
        version="v2025.06.08",
        artifact_path="ml/models/lgbm_ranker_v2025.06.08.pkl",
        feature_set_hash=fc.feature_set_hash(),
        training_rows=4200,
        label_source="synthetic",
        metrics={"test_mse": 0.012},
    )

From serving (recommendation_service.py):
    registry = ModelRegistry()
    meta = registry.get_active()
    model = joblib.load(meta["artifact_path"])
    fc.assert_booster_matches(model)

Promoting to production:
    registry.promote("v2025.06.08")
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import firebase_admin
from firebase_admin import credentials, firestore

_CRED_PATH = os.getenv("FIREBASE_SERVICE_ACCOUNT_PATH", "config/serviceAccountKey.json")

_COLLECTION = "model_registry"
_ACTIVE_DOC = "__active__"


def _get_db():
    """Return the Firestore client, initializing Firebase on first call."""
    if not firebase_admin._apps:
        cred = credentials.Certificate(_CRED_PATH)
        firebase_admin.initialize_app(cred)
    return firestore.client()


class ModelRegistry:
    """
    Read/write interface for the Firestore model registry.

    All methods are synchronous (Firestore Python SDK blocking calls).
    """

    def __init__(self, collection: str = _COLLECTION) -> None:
        self._collection = collection

    @property
    def _col(self):
        return _get_db().collection(self._collection)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def register(
        self,
        version: str,
        artifact_path: str,
        feature_set_hash: str,
        *,
        training_rows: Optional[int] = None,
        label_source: str = "synthetic",
        metrics: Optional[Dict[str, float]] = None,
        notes: Optional[str] = None,
        status: str = "staging",
    ) -> Dict[str, Any]:
        """
        Register a newly trained model version.

        Does NOT promote it to active automatically — call promote() when
        you are satisfied with offline evaluation.
        """
        metadata: Dict[str, Any] = {
            "version": version,
            "artifact_path": artifact_path,
            "feature_set_hash": feature_set_hash,
            "label_source": label_source,
            "status": status,
            "registered_at": datetime.now(timezone.utc).isoformat(),
        }
        if training_rows is not None:
            metadata["training_rows"] = training_rows
        if metrics:
            metadata["metrics"] = metrics
        if notes:
            metadata["notes"] = notes

        self._col.document(version).set(metadata)
        return metadata

    def promote(self, version: str) -> None:
        """
        Promote a registered version to active production status.

        Retires the previously active version (if any) and updates
        the __active__ pointer document.

        Raises:
            ValueError: The version has not been registered yet.
        """
        doc = self._col.document(version).get()
        if not doc.exists:
            raise ValueError(
                f"Version '{version}' is not registered. "
                "Call register() before promote()."
            )

        active = self.get_active()
        if active and active.get("version") != version:
            self._col.document(active["version"]).update({"status": "retired"})

        self._col.document(version).update({
            "status": "active",
            "promoted_at": datetime.now(timezone.utc).isoformat(),
        })

        self._col.document(_ACTIVE_DOC).set({
            "version": version,
            "promoted_at": datetime.now(timezone.utc).isoformat(),
        })

    def retire(self, version: str) -> None:
        """Retire a model version without promoting another one."""
        self._col.document(version).update({"status": "retired"})

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_active(self) -> Optional[Dict[str, Any]]:
        """
        Return the metadata for the current active model version.
        Returns None if no version has been promoted yet.
        """
        pointer = self._col.document(_ACTIVE_DOC).get()
        if not pointer.exists:
            return None
        version = pointer.to_dict().get("version")
        if not version:
            return None
        return self.get(version)

    def get(self, version: str) -> Optional[Dict[str, Any]]:
        """Return metadata for a specific version. Returns None if not found."""
        doc = self._col.document(version).get()
        return doc.to_dict() if doc.exists else None

    def list_versions(
        self,
        status: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        List registered model versions, most recently registered first.

        Args:
            status: Filter by status ("staging", "active", "retired").
                    None returns all versions.
            limit:  Max results to return.
        """
        query = self._col
        if status:
            query = query.where("status", "==", status)

        docs = query.stream()
        results = [
            doc.to_dict()
            for doc in docs
            if doc.id != _ACTIVE_DOC
        ]
        results.sort(key=lambda d: d.get("registered_at", ""), reverse=True)
        return results[:limit]