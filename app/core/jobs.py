from typing import TypedDict, Optional, Literal, Dict, Any, Iterable
from datetime import datetime


# --- типы данных, хранимые в _store ---
JobStatus = Literal["queued", "processing", "done", "error", "cancelled"]


class JobInfo(TypedDict, total=False):
    # идентификаторы / времена
    job_id: str
    created_at: datetime
    started_at: datetime
    finished_at: datetime

    # статус
    status: JobStatus

    # исходные метаданные о файле
    filename: Optional[str]
    path: Optional[str]  # временный путь к файлу на диске
    doc_title: Optional[str]  # человекочитаемое имя документа
    namespace: Optional[str]  # например "book_<job_id>"
    index_name: Optional[str]

    # прогресс и метрики
    total_chunks: Optional[int]
    processed_chunks: Optional[int]
    progress: Optional[int]  # 0..100 (int)

    # коллбэк / ошибки / доп. данные
    callback_url: Optional[str]
    error: Optional[str]
    extra: Optional[Dict[str, Any]]


class JobStore:
    def __init__(self) -> None:
        # ключ -> JobInfo
        self._store: Dict[str, JobInfo] = {}

    def create(self, job_id: str, info: JobInfo) -> None:
        """Создать запись. Подставляем обязательные дефолты."""
        base: JobInfo = {
            "job_id": job_id,
            "created_at": datetime.utcnow(),
            "status": "queued",
            "progress": 0,
            "processed_chunks": 0,
            **info,
        }
        self._store[job_id] = base

    def get(self, job_id: str) -> Optional[JobInfo]:
        item = self._store.get(job_id)
        # возвращаем копию, чтобы не позволять внешнему коду мутировать напрямую
        return dict(item) if item is not None else None

    def get_listing(self) -> Iterable[JobInfo]:
        return list(self._store.values())

    def update(self, job_id: str, patch: Dict[str, Any]) -> None:
        """Обновить поля (patch). Без исключений — тихо не делает ничего, если job нет."""
        if job_id not in self._store:
            return
        self._store[job_id].update(patch)
        # время старта/финиша автоматически проставляем при смене статуса
        st = patch.get("status")
        if st == "processing" and "started_at" not in self._store[job_id]:
            self._store[job_id]["started_at"] = datetime.utcnow()
        if st in ("done", "error", "cancelled"):
            self._store[job_id].setdefault("finished_at", datetime.utcnow())

    def increment_processed(self, job_id: str, n: int = 1) -> None:
        """Увеличить processed_chunks и пересчитать progress, если известен total_chunks."""
        if job_id not in self._store:
            return
        cur = self._store[job_id].get("processed_chunks") or 0
        cur += n
        self._store[job_id]["processed_chunks"] = cur

        total = self._store[job_id].get("total_chunks")
        if total and total > 0:
            prog = int((cur / total) * 100)
            # cap
            self._store[job_id]["progress"] = max(0, min(100, prog))

    def set_progress(self, job_id: str, progress: int) -> None:
        if job_id not in self._store:
            return
        self._store[job_id]["progress"] = max(0, min(100, int(progress)))

    def finish(
        self, job_id: str, success: bool = True, error: Optional[str] = None
    ) -> None:
        if job_id not in self._store:
            return
        self._store[job_id]["status"] = "done" if success else "error"
        if error:
            self._store[job_id]["error"] = error
        self._store[job_id].setdefault("finished_at", datetime.utcnow())
        self._store[job_id]["progress"] = (
            100 if success else self._store[job_id].get("progress", 0)
        )

    def delete(self, job_id: str) -> None:
        self._store.pop(job_id, None)
